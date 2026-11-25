// SPDX-License-Identifier: Apache-2.0
// Drives mpnpu_mmio the way a RISC-V core would (register writes / reads) and checks both modes.
`timescale 1ns/1ps
module tb_mmio;
    reg clk = 0; always #5 clk = ~clk;
    reg rst_n = 0, we = 0, re = 0; reg [4:0] addr = 0; reg [31:0] wd = 0; wire [31:0] rd;
    mpnpu_mmio dut (.clk(clk), .rst_n(rst_n), .bus_we(we), .bus_re(re), .bus_addr(addr), .bus_wdata(wd), .bus_rdata(rd));

    integer errors = 0;
    task wr(input [4:0] a, input [31:0] d); begin @(posedge clk); we <= 1; addr <= a; wd <= d; @(posedge clk); we <= 0; end endtask
    task rdreg(input [4:0] a, output [31:0] d); begin @(posedge clk); re <= 1; addr <= a; #1 d = rd; @(posedge clk); re <= 0; end endtask
    task wait_valid; reg [31:0] s; integer t; begin s = 0; t = 0; while (!s[0] && t < 50) begin rdreg(5'h1C, s); t = t + 1; end end endtask

    reg [31:0] r; reg [63:0] A, W; integer i, exp_v;
    initial begin
        repeat (3) @(posedge clk); rst_n <= 1;
        // ---- 1-bit mode: two beats, expect number of matching lanes
        wr(5'h00, 0);
        A = 64'hF0F0_F0F0_0123_4567; W = 64'hFF00_FF00_89AB_CDEF; exp_v = 0;
        for (i = 0; i < 64; i = i + 1) exp_v = exp_v + (A[i] == W[i]);
        wr(5'h04, A[31:0]); wr(5'h08, A[63:32]); wr(5'h0C, W[31:0]); wr(5'h10, W[63:32]); wr(5'h14, 32'b01);   // first
        A = 64'h0123_4567_89AB_CDEF; W = 64'hFEDC_BA98_7654_3210;
        for (i = 0; i < 64; i = i + 1) exp_v = exp_v + (A[i] == W[i]);
        wr(5'h04, A[31:0]); wr(5'h08, A[63:32]); wr(5'h0C, W[31:0]); wr(5'h10, W[63:32]); wr(5'h14, 32'b10);   // last
        wait_valid; rdreg(5'h18, r);
        if (r !== exp_v) begin errors = errors + 1; $display("1-bit FAIL got %0d exp %0d", r, exp_v); end else $display("1-bit OK: %0d matches", r);
        // ---- INT8 mode: one beat, 8 signed products
        wr(5'h00, 1);
        A = {8'd127, 8'h80, 8'd3, 8'hFD, 8'd10, 8'd0, 8'h81, 8'd1}; W = {8'd127, 8'h80, 8'hFC, 8'd5, 8'hF6, 8'd9, 8'h7F, 8'd2};
        exp_v = 127*127 + (-128*-128) + (3*-4) + (-3*5) + (10*-10) + 0 + (-127*127) + 2;
        wr(5'h04, A[31:0]); wr(5'h08, A[63:32]); wr(5'h0C, W[31:0]); wr(5'h10, W[63:32]); wr(5'h14, 32'b11);
        wait_valid; rdreg(5'h18, r);
        if ($signed(r) !== exp_v) begin errors = errors + 1; $display("INT8 FAIL got %0d exp %0d", $signed(r), exp_v); end else $display("INT8 OK: %0d", $signed(r));
        rdreg(5'h1C, r); if (r[0] !== 0) begin errors = errors + 1; $display("VALID not cleared by RESULT read"); end
        if (errors == 0) $display("PASS"); else $display("FAIL");
        $finish;
    end
endmodule
