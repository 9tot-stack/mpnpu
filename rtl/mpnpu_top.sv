// SPDX-License-Identifier: Apache-2.0
// Tang Nano 9K top level: UART <-> mpnpu_core.
//
// Host protocol (8N1):
//   0xA5 + 98 bytes   packed 28x28 binary image (pixel i = bit (i%8) of byte i/8, row-major)
//                     -> reply 4 bytes: 0x5A, class, cycles[7:0], cycles[15:8]
//   0xA4              -> reply 4 bytes: 'N','P','U','1'   (device probe)
//
// Compute latency in clock cycles is reported so the host can compute on-device time exactly.

`ifndef MEM_DIR
`define MEM_DIR "K:/FPGA/P1/mpnpu/mpnpu/rtl/mem/"
`endif

module mpnpu_top #(
    parameter CPB     = 234,              // 27 MHz / 115200 baud
    parameter MEM_DIR = `MEM_DIR
) (
    input  wire       clk,                // 27 MHz
    input  wire       rst_btn_n,          // user button S1 (active low)
    input  wire       uart_rx,
    output wire       uart_tx,
    output wire [5:0] led_n               // active-low LEDs
);
    // ---- power-on reset + button
    reg [15:0] por = 16'd0;
    always @(posedge clk) if (por != 16'hFFFF) por <= por + 16'd1;
    wire rst_n = rst_btn_n & (por == 16'hFFFF);

    // ---- UART
    wire       rx_v;  wire [7:0] rx_d;
    reg        tx_send; reg [7:0] tx_d; wire tx_busy;
    uart_rx #(.CPB(CPB)) u_rx (.clk(clk), .rst_n(rst_n), .rxd(uart_rx), .valid(rx_v), .data(rx_d));
    uart_tx #(.CPB(CPB)) u_tx (.clk(clk), .rst_n(rst_n), .send(tx_send), .data(tx_d), .txd(uart_tx), .busy(tx_busy));

    // ---- core
    reg        img_we;  reg [9:0] img_addr;  reg img_bit;  reg start;
    wire       busy, done, pmode;
    wire [3:0] cls;     wire signed [31:0] best;  wire [15:0] cycles;
    mpnpu_core #(.MEM_DIR(MEM_DIR)) u_core (
        .clk(clk), .rst_n(rst_n), .img_we(img_we), .img_addr(img_addr), .img_bit(img_bit), .start(start),
        .busy(busy), .done(done), .result_class(cls), .result_best(best), .cycles(cycles), .precision_mode_o(pmode)
    );

    // ---- protocol FSM
    localparam T_CMD = 3'd0, T_RECV = 3'd1, T_LOAD = 3'd2, T_WAIT = 3'd3, T_RUN = 3'd4, T_TX = 3'd5;
    reg [2:0]  ts;
    reg [6:0]  nbytes;
    reg [7:0]  ld_byte; reg [3:0] ld_i; reg [9:0] ld_base; reg ld_act;
    reg [31:0] resp;    reg [2:0] tx_n;  reg tx_wait;

    always @(posedge clk) begin
        if (!rst_n) begin
            ts <= T_CMD; nbytes <= 7'd0; ld_act <= 1'b0; img_we <= 1'b0; start <= 1'b0; tx_send <= 1'b0;
            ld_byte <= 8'd0; ld_i <= 4'd0; ld_base <= 10'd0; resp <= 32'd0; tx_n <= 3'd0; tx_wait <= 1'b0;
            img_addr <= 10'd0; img_bit <= 1'b0; tx_d <= 8'd0;
        end else begin
            img_we <= 1'b0; start <= 1'b0; tx_send <= 1'b0;
            // pixel loader: 8 writes per received byte
            if (ld_act) begin
                img_we <= 1'b1; img_addr <= ld_base + {6'd0, ld_i}; img_bit <= ld_byte[ld_i[2:0]];
                if (ld_i == 4'd7) ld_act <= 1'b0; else ld_i <= ld_i + 4'd1;
            end
            case (ts)
                T_CMD: if (rx_v) begin
                    if (rx_d == 8'hA5) begin ts <= T_RECV; nbytes <= 7'd0; end
                    else if (rx_d == 8'hA4) begin resp <= {8'h31, 8'h55, 8'h50, 8'h4E}; tx_n <= 3'd0; tx_wait <= 1'b0; ts <= T_TX; end
                end
                T_RECV: if (rx_v) begin
                    ld_byte <= rx_d; ld_i <= 4'd0; ld_base <= {nbytes, 3'b000}; ld_act <= 1'b1;
                    nbytes <= nbytes + 7'd1;
                    if (nbytes == 7'd97) ts <= T_LOAD;
                end
                T_LOAD: if (!ld_act && !img_we) begin start <= 1'b1; ts <= T_WAIT; end
                T_WAIT: if (busy) ts <= T_RUN;          // core clears `done` as it raises `busy`; never trust a stale done
                T_RUN: if (done && !busy) begin
                    resp <= {cycles[15:8], cycles[7:0], 4'd0, cls, 8'h5A}; tx_n <= 3'd0; tx_wait <= 1'b0; ts <= T_TX;
                end
                T_TX: begin
                    if (tx_wait) begin
                        if (tx_busy) tx_wait <= 1'b0;                      // UART accepted the byte
                    end else if (!tx_busy && !tx_send) begin
                        if (tx_n == 3'd4) ts <= T_CMD;                     // all 4 bytes fully shifted out
                        else begin
                            tx_d <= resp[7:0]; tx_send <= 1'b1; resp <= {8'd0, resp[31:8]};
                            tx_n <= tx_n + 3'd1; tx_wait <= 1'b1;
                        end
                    end
                end
                default: ts <= T_CMD;
            endcase
        end
    end

    // ---- LEDs (active low): 0 busy, 1 precision_mode, 2 done-seen, 3..5 class[2:0]
    reg seen;
    always @(posedge clk) if (!rst_n) seen <= 1'b0; else if (done) seen <= 1'b1; else if (start) seen <= 1'b0;
    assign led_n = ~{cls[2:0], seen, pmode, busy};
endmodule
