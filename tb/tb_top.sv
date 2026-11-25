// SPDX-License-Identifier: Apache-2.0
// End-to-end test: bit-banged UART host -> mpnpu_top -> compare with the Python golden model.
// For every image it checks (1) the returned class, (2) the 256 region counts left by the two
// 1-bit conv layers, (3) the winning int8-dense logit, and (4) that precision_mode really toggled
// 0 -> 1 during the run. Run from the project root so the $readmemh paths resolve.
`timescale 1ns/1ps
module tb_top;
    parameter CPB = 8;
    parameter NV  = 64;
    reg clk = 0; always #5 clk = ~clk;     // 100 MHz in sim; real board is 27 MHz (cycle counts are what matter)
    reg rst_btn_n = 0;
    reg rxd = 1;
    wire txd; wire [5:0] led_n;
    mpnpu_top #(.CPB(CPB)) dut (.clk(clk), .rst_btn_n(rst_btn_n), .uart_rx(rxd), .uart_tx(txd), .led_n(led_n));

    reg [7:0]  img   [0:NV*98-1];
    reg [7:0]  expc  [0:NV-1];
    reg [31:0] expb  [0:NV-1];
    reg [7:0]  expn  [0:NV*256-1];
    reg [7:0]  label [0:NV-1];

    task send_byte(input [7:0] b);
        integer i;
        begin
            rxd <= 0; repeat (CPB) @(posedge clk);
            for (i = 0; i < 8; i = i + 1) begin rxd <= b[i]; repeat (CPB) @(posedge clk); end
            rxd <= 1; repeat (CPB) @(posedge clk);
        end
    endtask

    task recv_byte(output [7:0] b);
        integer i;
        begin
            @(negedge txd);
            repeat (CPB + CPB / 2) @(posedge clk);
            for (i = 0; i < 8; i = i + 1) begin b[i] = txd; repeat (CPB) @(posedge clk); end
        end
    endtask

    integer errors = 0, correct = 0, n, k, c, mode1_cycles, mode0_cycles, maxcyc = 0, mintog;
    reg [7:0] r0, r1, r2, r3, cmp_n;
    reg [15:0] cyc;
    reg saw0, saw1;
    always @(posedge clk) if (dut.u_core.busy) begin
        if (dut.u_core.precision_mode) begin saw1 <= 1; mode1_cycles <= mode1_cycles + 1; end
        else begin saw0 <= 1; mode0_cycles <= mode0_cycles + 1; end
    end

    initial begin
        $readmemh("tb/vectors/tv_img.hex",   img);
        $readmemh("tb/vectors/tv_class.hex", expc);
        $readmemh("tb/vectors/tv_best.hex",  expb);
        $readmemh("tb/vectors/tv_cnt.hex",   expn);
        $readmemh("tb/vectors/tv_label.hex", label);
        mode0_cycles = 0; mode1_cycles = 0; saw0 = 0; saw1 = 0;
        repeat (10) @(posedge clk); rst_btn_n <= 1;
        wait (dut.rst_n === 1'b1); repeat (10) @(posedge clk);

        // device probe
        send_byte(8'hA4); recv_byte(r0); recv_byte(r1); recv_byte(r2); recv_byte(r3);
        if ({r0, r1, r2, r3} !== "NPU1") begin errors = errors + 1; $display("probe FAIL: %h %h %h %h", r0, r1, r2, r3); end
        else $display("probe OK: NPU1");

        for (n = 0; n < NV; n = n + 1) begin
            send_byte(8'hA5);
            for (k = 0; k < 98; k = k + 1) send_byte(img[n * 98 + k]);
            recv_byte(r0); recv_byte(r1); recv_byte(r2); recv_byte(r3);
            cyc = {r3, r2};
            if (cyc > maxcyc) maxcyc = cyc;
            cmp_n = 0;
            for (c = 0; c < 256; c = c + 1) if ({4'b0, dut.u_core.cnt[c]} !== expn[n * 256 + c]) cmp_n = cmp_n + 1;
            if (r0 !== 8'h5A || r1 !== expc[n] || cmp_n != 0 || dut.u_core.result_best !== expb[n]) begin
                errors = errors + 1;
                $display("img %0d MISMATCH: hdr %h class %0d (exp %0d) count-diffs %0d best %0d (exp %0d)",
                         n, r0, r1, expc[n], cmp_n, $signed(dut.u_core.result_best), $signed(expb[n]));
            end
            if (r1 === label[n]) correct = correct + 1;
        end
        $display("images: %0d  errors vs golden: %0d  accuracy vs labels: %0d/%0d", NV, errors, correct, NV);
        $display("precision_mode seen 0:%0d 1:%0d ; busy cycles in mode0 (1-bit conv) = %0d, mode1 (int8 dense) = %0d (summed over images)",
                 saw0, saw1, mode0_cycles, mode1_cycles);
        $display("compute latency (cycles @ 27 MHz): last %0d, max %0d  -> %0.3f ms", cyc, maxcyc, cyc / 27000.0);
        if (errors == 0 && saw0 && saw1) $display("PASS"); else $display("FAIL");
        $finish;
    end

    initial begin #2000000000; $display("TIMEOUT"); $finish; end
endmodule
