// SPDX-License-Identifier: Apache-2.0
// Switchable 1-bit / 8-bit matrix execution unit for Gowin GW1NR-9 (Tang Nano 9K).
//
//   precision_mode = 0 : 64 x 1-bit MACs  -> XNOR + popcount, LUTs only, no DSP.
//                        a[i], w[i] are single bits (1 = +1, 0 = -1). The result accumulates the
//                        number of MATCHING lanes (0..64 per beat). The signed +-1 dot product is
//                        2*matches - lanes, and the consumer folds that affine term into its
//                        threshold (see sw/train_export.py), which is cheaper than doing it here.
//   precision_mode = 1 : 8 x INT8 MACs    -> 8 signed 8x8 multipliers (DSP) + adder tree.
//                        a[8i+:8], w[8i+:8] are two's-complement int8, result is the signed sum.
//
// One beat = one 64-bit activation word and one 64-bit weight word per cycle. A dot product of any
// length is streamed as consecutive beats; in_first clears the accumulator, in_last emits the result.
// Latency: in_valid -> out_valid = 3 cycles (3-stage pipeline). Throughput: 1 beat/cycle.
//
// precision_mode must only change while the pipeline is empty (the sequencer guarantees this: it
// waits for the last out_valid of a layer before loading the next layer descriptor).
// Operands of the inactive datapath are forced to zero (operand isolation) so it does not toggle.

// Popcount split in two halves so the pipeline register sits in the middle of the tree.
module popcount64_front (
    input  wire [63:0] x,
    output wire [31:0] y          // eight 4-bit partial counts
);
    wire [1:0] l1 [0:31];
    wire [2:0] l2 [0:15];
    genvar i;
    generate
        for (i = 0; i < 32; i = i + 1) begin : g1 assign l1[i] = {1'b0, x[2*i]} + {1'b0, x[2*i+1]}; end
        for (i = 0; i < 16; i = i + 1) begin : g2 assign l2[i] = {1'b0, l1[2*i]} + {1'b0, l1[2*i+1]}; end
        for (i = 0; i < 8;  i = i + 1) begin : g3 assign y[4*i +: 4] = {1'b0, l2[2*i]} + {1'b0, l2[2*i+1]}; end
    endgenerate
endmodule

module popcount64_back (
    input  wire [31:0] x,
    output wire [6:0]  y
);
    wire [4:0] l4 [0:3];
    wire [5:0] l5 [0:1];
    genvar i;
    generate
        for (i = 0; i < 4; i = i + 1) begin : g4 assign l4[i] = {1'b0, x[8*i +: 4]} + {1'b0, x[8*i+4 +: 4]}; end
        for (i = 0; i < 2; i = i + 1) begin : g5 assign l5[i] = {1'b0, l4[2*i]} + {1'b0, l4[2*i+1]}; end
    endgenerate
    assign y = {1'b0, l5[0]} + {1'b0, l5[1]};
endmodule

module matrix_unit (
    input  wire               clk,
    input  wire               rst_n,
    input  wire               precision_mode,   // 0 = 1-bit BNN, 1 = INT8
    input  wire               in_valid,
    input  wire               in_first,
    input  wire               in_last,
    input  wire [63:0]        a,
    input  wire [63:0]        w,
    output reg                out_valid,
    output reg signed [31:0]  out_acc
);
    // ---- operand isolation: the inactive datapath sees constant zeros and does not toggle
    wire [63:0] a_bnn = precision_mode ? 64'd0 : a;
    wire [63:0] w_bnn = precision_mode ? 64'd0 : w;
    wire [63:0] a_i8  = precision_mode ? a : 64'd0;
    wire [63:0] w_i8  = precision_mode ? w : 64'd0;

    // ---- stage 0 (combinational): 8 signed 8x8 products (DSP) and first 3 popcount levels (LUT)
    wire signed [15:0] p [0:7];
    wire [31:0]        pc_front;
    genvar i;
    generate
        for (i = 0; i < 8; i = i + 1) begin : g_mul
            assign p[i] = $signed(a_i8[8*i +: 8]) * $signed(w_i8[8*i +: 8]);
        end
    endgenerate
    popcount64_front u_pcf (.x(~(a_bnn ^ w_bnn)), .y(pc_front));

    // ---- stage 1 registers
    reg signed [15:0] p_r [0:7];
    reg [31:0]        pc_r;
    reg               v1, f1, l1, m1;
    integer k;
    always @(posedge clk) begin
        if (!rst_n) begin
            v1 <= 1'b0; f1 <= 1'b0; l1 <= 1'b0; m1 <= 1'b0; pc_r <= 32'd0;
            for (k = 0; k < 8; k = k + 1) p_r[k] <= 16'sd0;
        end else begin
            v1 <= in_valid; f1 <= in_first; l1 <= in_last; m1 <= precision_mode; pc_r <= pc_front;
            for (k = 0; k < 8; k = k + 1) p_r[k] <= p[k];
        end
    end

    // ---- stage 1 (combinational): adder tree / popcount tail
    wire signed [16:0] s0 = p_r[0] + p_r[1];
    wire signed [16:0] s1 = p_r[2] + p_r[3];
    wire signed [16:0] s2 = p_r[4] + p_r[5];
    wire signed [16:0] s3 = p_r[6] + p_r[7];
    wire signed [17:0] s4 = s0 + s1;
    wire signed [17:0] s5 = s2 + s3;
    wire signed [18:0] mac = s4 + s5;
    wire [6:0]         pop;
    popcount64_back u_pcb (.x(pc_r), .y(pop));

    // ---- stage 2 registers
    reg                v2, f2, l2;
    reg signed [31:0]  part;
    always @(posedge clk) begin
        if (!rst_n) begin
            v2 <= 1'b0; f2 <= 1'b0; l2 <= 1'b0; part <= 32'sd0;
        end else begin
            v2 <= v1; f2 <= f1; l2 <= l1;
            part <= m1 ? {{13{mac[18]}}, mac} : {25'd0, pop};
        end
    end

    // ---- stage 2 (combinational) + accumulator register
    reg signed [31:0]  acc;
    wire signed [31:0] base = f2 ? 32'sd0 : acc;
    wire signed [31:0] nxt  = base + part;
    always @(posedge clk) begin
        if (!rst_n) begin
            acc <= 32'sd0; out_valid <= 1'b0; out_acc <= 32'sd0;
        end else begin
            out_valid <= 1'b0;
            if (v2) begin
                acc <= nxt;
                if (l2) begin out_valid <= 1'b1; out_acc <= nxt; end
            end
        end
    end
endmodule
