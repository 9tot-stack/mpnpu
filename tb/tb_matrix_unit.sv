// SPDX-License-Identifier: Apache-2.0
// Randomised self-checking test of matrix_unit in both precision modes, including a
// mid-stream mode switch (drained between layers, as the sequencer does).
`timescale 1ns/1ps
module tb_matrix_unit;
    reg clk = 0; always #5 clk = ~clk;
    reg rst_n = 0;
    reg mode = 0, v = 0, f = 0, l = 0;
    reg [63:0] a = 0, w = 0;
    wire ov; wire signed [31:0] acc;
    matrix_unit dut (.clk(clk), .rst_n(rst_n), .precision_mode(mode), .in_valid(v), .in_first(f), .in_last(l),
                     .a(a), .w(w), .out_valid(ov), .out_acc(acc));

    integer errors = 0, checks = 0;
    reg signed [31:0] exp_q [0:63];
    integer qh = 0, qt = 0, n_exp = 0;

    // reference model
    function automatic integer ref_beat(input m, input [63:0] aa, input [63:0] ww);
        integer i, s; reg signed [7:0] x, y;
        begin
            s = 0;
            if (!m) begin for (i = 0; i < 64; i = i + 1) s = s + ((aa[i] == ww[i]) ? 1 : 0); end
            else    begin for (i = 0; i < 8; i = i + 1) begin x = aa[8*i +: 8]; y = ww[8*i +: 8]; s = s + x * y; end end
            ref_beat = s;
        end
    endfunction

    always @(posedge clk) if (ov) begin
        checks = checks + 1;
        if (acc !== exp_q[qh]) begin errors = errors + 1; $display("MISMATCH got %0d exp %0d", acc, exp_q[qh]); end
        qh = (qh + 1) % 64;
    end

    task run_dot(input m, input integer beats, input bit extreme);
        integer b; integer s; reg [63:0] aa, ww;
        begin
            s = 0;
            for (b = 0; b < beats; b = b + 1) begin
                aa = {$urandom, $urandom}; ww = {$urandom, $urandom};
                if (extreme) begin aa = (b % 2) ? 64'h8080808080808080 : 64'h7F7F7F7F7F7F7F7F;
                                   ww = (b % 3) ? 64'h8080808080808080 : 64'h7F7F7F7F7F7F7F7F; end
                a <= aa; w <= ww; v <= 1; f <= (b == 0); l <= (b == beats - 1);
                s = s + ref_beat(m, aa, ww);
                @(posedge clk);
            end
            exp_q[qt] = s; qt = (qt + 1) % 64; n_exp = n_exp + 1;
            v <= 0; f <= 0; l <= 0;
            if ($urandom % 2) repeat ($urandom % 3) @(posedge clk);   // random bubbles between dot products
        end
    endtask

    integer i;
    initial begin
        repeat (4) @(posedge clk); rst_n <= 1; repeat (2) @(posedge clk);
        // mode 0: single beats, 3-beat (conv2-like), all-ones / all-zeros edges
        mode <= 0; @(posedge clk);
        for (i = 0; i < 200; i = i + 1) run_dot(0, 1 + ($urandom % 4), 0);
        a <= 64'hFFFF_FFFF_FFFF_FFFF; w <= 64'hFFFF_FFFF_FFFF_FFFF; v <= 1; f <= 1; l <= 1; exp_q[qt] = 64; qt = (qt + 1) % 64; n_exp = n_exp + 1; @(posedge clk); v <= 0;
        a <= 0; w <= 64'hFFFF_FFFF_FFFF_FFFF; v <= 1; f <= 1; l <= 1; exp_q[qt] = 0; qt = (qt + 1) % 64; n_exp = n_exp + 1; @(posedge clk); v <= 0;
        repeat (6) @(posedge clk);                      // drain, then switch precision
        mode <= 1; @(posedge clk);
        for (i = 0; i < 200; i = i + 1) run_dot(1, 1 + ($urandom % 8), 0);
        for (i = 0; i < 20; i = i + 1)  run_dot(1, 8, 1);   // extreme values: -128*-128 and 127*127 corners
        repeat (6) @(posedge clk);
        mode <= 0; @(posedge clk);
        for (i = 0; i < 100; i = i + 1) run_dot(0, 3, 0);
        repeat (6) @(posedge clk);
        $display("matrix_unit: %0d dot products checked, %0d errors", checks, errors);
        if (errors == 0 && checks == n_exp) $display("PASS"); else $display("FAIL");
        $finish;
    end
endmodule
