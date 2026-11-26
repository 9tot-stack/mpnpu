// SPDX-License-Identifier: Apache-2.0
// Layer sequencer + matrix unit + on-chip memories.  Runs one MNIST inference per `start` pulse.
//
// The network is described by a layer-descriptor ROM (layers.hex, generated from model_config.json).
// For every layer the sequencer loads the descriptor and writes the matrix unit's single
// `precision_mode` register from it, so mode 0 (1-bit) and mode 1 (INT8) alternate layer by layer.
//
// Descriptor (64 bit):
//   [0] mode  [1] src_sel  [2] pix16  [3] sink_cnt  [8:4] in_w  [12:9] pout_w  [16:13] pout_h
//   [23:17] out_c  [29:24] nchunk  [39:30] wbase  [47:40] tbase  [48] last
//   mode 0 : 3x3 valid conv + 2x2 max-pool (OR of thresholded bits). pix16=1 -> 16-bit channel
//            vectors per pixel, else 1 bit. sink_cnt=1 -> instead of writing the feature map, add each
//            pooled pixel into four overlapping region counters per channel (top/bottom x left/right,
//            split at pout>>1, so a 5x5 map gives 3x3 regions); counter index = region*64 + channel.
//   mode 1 : dense layer on the 256 counts as int8 + argmax. out_c = classes, nchunk = 32.
//
// Weight word addressing is oc-major / chunk-minor, so the issue pointer just increments.

`ifndef MEM_DIR
`define MEM_DIR "K:/FPGA/P1/mpnpu/rtl/mem/"
`endif

module mpnpu_core #(
    parameter MEM_DIR = `MEM_DIR
) (
    input  wire        clk,
    input  wire        rst_n,
    // image load port (1 pixel per write, row-major 28x28)
    input  wire        img_we,
    input  wire [9:0]  img_addr,
    input  wire        img_bit,
    input  wire        start,
    output reg         busy,
    output reg         done,
    output reg  [3:0]  result_class,
    output reg  signed [31:0] result_best,
    output reg  [15:0] cycles,
    output wire        precision_mode_o
);
    // ------------------------------------------------------------------ states
    localparam S_IDLE = 4'd0,  S_DESC  = 4'd1,  S_DESC2 = 4'd2,  S_CINIT = 4'd3,
               S_PBEG = 4'd4,  S_WBEG  = 4'd5,  S_GATH  = 4'd6,   S_MACP  = 4'd7,
               S_MACI = 4'd8,  S_MACW  = 4'd9,  S_POOLW = 4'd10,  S_NEXTL = 4'd11,
               S_DINIT = 4'd12, S_DONE = 4'd13;
    reg [3:0] st;

    // ------------------------------------------------------------------ descriptor
    reg [2:0]  layer_idx;
    wire [63:0] desc_rd;
    rom_sync #(.W(64), .D(8), .AW(3), .FILE({MEM_DIR, "layers.hex"})) u_desc (.clk(clk), .addr(layer_idx), .rdata(desc_rd));

    reg        d_mode, d_src, d_pix16, d_sink, d_last;
    reg [4:0]  d_in_w;
    reg [3:0]  d_pout_w, d_pout_h;
    reg [6:0]  d_out_c;
    reg [5:0]  d_nchunk;
    reg [9:0]  d_wbase;
    reg [7:0]  d_tbase;

    reg precision_mode;                       // the single runtime precision register
    assign precision_mode_o = precision_mode;

    // ------------------------------------------------------------------ memories
    reg  [9:0]  pa;                           // activation read address
    wire [15:0] rd0, rd1;
    wire        seq_we = (st == S_POOLW) && !d_sink;
    reg  [9:0]  oaddr;
    reg  [63:0] pool_bits;
    wire        we0 = img_we | (seq_we &  d_src);          // dst = ~src ; src=1 -> dst=0
    wire        we1 = seq_we & ~d_src;
    sdp_ram #(.W(16), .D(1024), .AW(10)) u_act0 (.clk(clk), .we(we0), .waddr(img_we ? img_addr : oaddr),
                                                 .wdata(img_we ? {15'd0, img_bit} : pool_bits[15:0]), .raddr(pa), .rdata(rd0));
    sdp_ram #(.W(16), .D(1024), .AW(10)) u_act1 (.clk(clk), .we(we1), .waddr(oaddr),
                                                 .wdata(pool_bits[15:0]), .raddr(pa), .rdata(rd1));
    wire [15:0] pix = d_src ? rd1 : rd0;

    // weights: two 32-bit BSRAM banks read in parallel = one 64-bit beat per cycle
    reg  [9:0]  wptr;
    wire [31:0] w0, w1;
    rom_sync #(.W(32), .D(1024), .AW(10), .FILE({MEM_DIR, "wmem0.hex"})) u_w0 (.clk(clk), .addr(wptr), .rdata(w0));
    rom_sync #(.W(32), .D(1024), .AW(10), .FILE({MEM_DIR, "wmem1.hex"})) u_w1 (.clk(clk), .addr(wptr), .rdata(w1));

    // thresholds / bias, read with look-ahead so the value is ready when a result arrives
    reg  [6:0]  res_oc;
    wire        mu_ov;
    wire signed [31:0] mu_acc;
    wire [7:0]  thr_rd;
    wire [6:0]  taddr = d_tbase[6:0] + (mu_ov ? res_oc + 7'd1 : res_oc);
    rom_sync #(.W(8), .D(128), .AW(7), .FILE({MEM_DIR, "thr.hex"})) u_thr (.clk(clk), .addr(taddr), .rdata(thr_rd));
    wire signed [31:0] bias_rd;
    wire [3:0]  baddr = mu_ov ? res_oc[3:0] + 4'd1 : res_oc[3:0];
    rom_sync #(.W(32), .D(16), .AW(4), .FILE({MEM_DIR, "bias.hex"})) u_bias (.clk(clk), .addr(baddr), .rdata(bias_rd));

    // ------------------------------------------------------------------ window / counts
    reg [191:0] win;
    reg [255:0][3:0] cnt;                     // 4 regions x 64 channels, 0..9 each (packed so every tool accepts NBA-in-loop)
    reg [3:0]   px, py;
    reg [9:0]   row_base;
    reg [1:0]   sub;
    reg [1:0]   kc;
    reg [3:0]   gi;
    reg [5:0]   ch_i;
    reg [6:0]   oc_i;
    reg         res_fin;
    reg signed [31:0] best_val;
    reg [3:0]   best_idx;

    // ------------------------------------------------------------------ MAC issue pipeline
    reg        m_v, m_first, m_last;
    reg [5:0]  m_ch;
    wire [63:0] a_win = (m_ch == 6'd0) ? win[63:0] : (m_ch == 6'd1) ? win[127:64] : win[191:128];
    reg  [63:0] a_cnt;
    integer     ci;
    always @* begin
        for (ci = 0; ci < 8; ci = ci + 1)
            a_cnt[8*ci +: 8] = {4'b0000, cnt[{m_ch[4:0], 3'b000} + ci[7:0]]};
    end
    wire [63:0] a_mux = precision_mode ? a_cnt : a_win;

    matrix_unit u_mu (
        .clk(clk), .rst_n(rst_n), .precision_mode(precision_mode),
        .in_valid(m_v), .in_first(m_first), .in_last(m_last),
        .a(a_mux), .w({w1, w0}),
        .out_valid(mu_ov), .out_acc(mu_acc)
    );

    // ------------------------------------------------------------------ main FSM
    integer c;
    wire last_chunk = (ch_i == d_nchunk - 6'd1);
    wire last_issue = last_chunk && (oc_i == d_out_c - 7'd1);
    wire in_top = (py <= (d_pout_h >> 1)), in_bot = (py >= (d_pout_h >> 1));
    wire in_lft = (px <= (d_pout_w >> 1)), in_rgt = (px >= (d_pout_w >> 1));
    wire last_px    = (px == d_pout_w - 4'd1) && (py == d_pout_h - 4'd1);

    always @(posedge clk) begin
        if (!rst_n) begin
            st <= S_IDLE; busy <= 1'b0; done <= 1'b0; layer_idx <= 3'd0; precision_mode <= 1'b0;
            m_v <= 1'b0; res_fin <= 1'b0; res_oc <= 7'd0; cycles <= 16'd0; pool_bits <= 64'd0;
            best_val <= 32'sd0; best_idx <= 4'd0; result_class <= 4'd0; result_best <= 32'sd0;
            wptr <= 10'd0; oaddr <= 10'd0; pa <= 10'd0; px <= 4'd0; py <= 4'd0; row_base <= 10'd0;
            sub <= 2'd0; kc <= 2'd0; gi <= 4'd0; ch_i <= 6'd0; oc_i <= 7'd0; win <= 192'd0;
            d_mode <= 1'b0; d_src <= 1'b0; d_pix16 <= 1'b0; d_sink <= 1'b0; d_last <= 1'b0; d_in_w <= 5'd0;
            d_pout_w <= 4'd0; d_pout_h <= 4'd0; d_out_c <= 7'd0; d_nchunk <= 6'd0; d_wbase <= 10'd0; d_tbase <= 8'd0;
            m_first <= 1'b0; m_last <= 1'b0; m_ch <= 6'd0;
            for (c = 0; c < 256; c = c + 1) cnt[c] <= 4'd0;
        end else begin
            m_v <= 1'b0;
            if (busy) cycles <= (cycles == 16'hFFFF) ? cycles : cycles + 16'd1;

            case (st)
                S_IDLE: begin
                    if (start) begin
                        busy <= 1'b1; done <= 1'b0; cycles <= 16'd0; layer_idx <= 3'd0; st <= S_DESC;
                    end
                end
                S_DESC: st <= S_DESC2;                         // descriptor ROM read latency
                S_DESC2: begin
                    d_mode <= desc_rd[0];  d_src <= desc_rd[1];  d_pix16 <= desc_rd[2];  d_sink <= desc_rd[3];
                    d_in_w <= desc_rd[8:4]; d_pout_w <= desc_rd[12:9]; d_pout_h <= desc_rd[16:13];
                    d_out_c <= desc_rd[23:17]; d_nchunk <= desc_rd[29:24]; d_wbase <= desc_rd[39:30];
                    d_tbase <= desc_rd[47:40]; d_last <= desc_rd[48];
                    precision_mode <= desc_rd[0];              // <-- runtime mode switch, per layer
                    st <= desc_rd[0] ? S_DINIT : S_CINIT;
                end
                // ---------------- 1-bit conv + pool layer
                S_CINIT: begin
                    px <= 4'd0; py <= 4'd0; row_base <= 10'd0; oaddr <= 10'd0;
                    if (d_sink) for (c = 0; c < 256; c = c + 1) cnt[c] <= 4'd0;
                    st <= S_PBEG;
                end
                S_PBEG: begin pool_bits <= 64'd0; sub <= 2'd0; st <= S_WBEG; end
                S_WBEG: begin
                    win <= 192'd0;
                    pa  <= row_base + {5'd0, px, 1'b0} + (sub[1] ? {5'd0, d_in_w} : 10'd0) + {9'd0, sub[0]};
                    kc <= 2'd0; gi <= 4'd0; st <= S_GATH;
                end
                S_GATH: begin
                    if (gi <= 4'd8) begin                                   // issue pixel gi (pa drives raddr)
                        pa <= (kc == 2'd2) ? pa + {5'd0, d_in_w} - 10'd2 : pa + 10'd1;
                        kc <= (kc == 2'd2) ? 2'd0 : kc + 2'd1;
                    end
                    if (gi >= 4'd1) begin                                   // capture pixel gi-1 (rdata is 1 cycle late)
                        if (d_pix16) win[16*(gi-4'd1) +: 16] <= pix;
                        else         win[{4'b0000, gi - 4'd1}] <= pix[0];
                    end
                    gi <= gi + 4'd1;
                    if (gi == 4'd9) st <= S_MACP;
                end
                S_MACP: begin
                    wptr <= d_wbase; oc_i <= 7'd0; ch_i <= 6'd0; res_oc <= 7'd0; res_fin <= 1'b0; st <= S_MACI;
                end
                S_MACI: begin
                    m_v <= 1'b1; m_first <= (ch_i == 6'd0); m_last <= last_chunk; m_ch <= ch_i;
                    wptr <= wptr + 10'd1;
                    if (last_chunk) begin ch_i <= 6'd0; oc_i <= oc_i + 7'd1; end else ch_i <= ch_i + 6'd1;
                    if (last_issue) st <= S_MACW;
                end
                S_MACW: begin
                    if (res_fin) begin
                        res_fin <= 1'b0;
                        if (d_mode) st <= S_DONE;
                        else if (sub == 2'd3) st <= S_POOLW;
                        else begin sub <= sub + 2'd1; st <= S_WBEG; end
                    end
                end
                S_POOLW: begin
                    if (d_sink) begin                                   // region counters (overlap on the centre row/col)
                        for (c = 0; c < 64; c = c + 1) if (c < d_out_c) begin
                            if (in_top && in_lft) cnt[c]       <= cnt[c]       + {3'd0, pool_bits[c]};
                            if (in_top && in_rgt) cnt[64 + c]  <= cnt[64 + c]  + {3'd0, pool_bits[c]};
                            if (in_bot && in_lft) cnt[128 + c] <= cnt[128 + c] + {3'd0, pool_bits[c]};
                            if (in_bot && in_rgt) cnt[192 + c] <= cnt[192 + c] + {3'd0, pool_bits[c]};
                        end
                    end
                    oaddr <= oaddr + 10'd1;
                    if (px == d_pout_w - 4'd1) begin px <= 4'd0; py <= py + 4'd1; row_base <= row_base + {4'd0, d_in_w, 1'b0}; end
                    else px <= px + 4'd1;
                    st <= last_px ? S_NEXTL : S_PBEG;
                end
                S_NEXTL: begin
                    layer_idx <= layer_idx + 3'd1; st <= S_DESC;
                end
                // ---------------- 8-bit dense layer + argmax
                S_DINIT: begin
                    best_val <= 32'sh80000000; best_idx <= 4'd0; st <= S_MACP;
                end
                S_DONE: begin
                    result_class <= best_idx; result_best <= best_val;
                    busy <= 1'b0; done <= 1'b1; st <= S_IDLE;
                end
                default: st <= S_IDLE;
            endcase

            // ---------------- result handling (runs concurrently with the issue side)
            if (mu_ov) begin
                if (!precision_mode) begin
                    if (mu_acc[7:0] >= thr_rd) pool_bits[res_oc[5:0]] <= 1'b1;   // maxpool = OR
                end else begin
                    if ((mu_acc + bias_rd) > best_val) begin best_val <= mu_acc + bias_rd; best_idx <= res_oc[3:0]; end
                end
                if (res_oc == d_out_c - 7'd1) begin res_oc <= 7'd0; res_fin <= 1'b1; end
                else res_oc <= res_oc + 7'd1;
            end
        end
    end
endmodule
