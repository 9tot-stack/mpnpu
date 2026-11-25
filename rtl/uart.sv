// SPDX-License-Identifier: Apache-2.0
// Minimal 8N1 UART. CPB = clock cycles per bit (27e6/115200 = 234 on the Tang Nano 9K).

module uart_rx #(parameter CPB = 234) (
    input  wire       clk,
    input  wire       rst_n,
    input  wire       rxd,
    output reg        valid,
    output reg  [7:0] data
);
    reg [1:0]  sync;
    reg        busy;
    reg [15:0] cnt;
    reg [3:0]  bitn;
    reg [7:0]  sh;
    always @(posedge clk) begin
        if (!rst_n) begin
            sync <= 2'b11; busy <= 1'b0; valid <= 1'b0; cnt <= 16'd0; bitn <= 4'd0; sh <= 8'd0; data <= 8'd0;
        end else begin
            sync <= {sync[0], rxd};
            valid <= 1'b0;
            if (!busy) begin
                if (!sync[1]) begin busy <= 1'b1; cnt <= CPB / 2; bitn <= 4'd0; end   // start bit: sample at its middle
            end else if (cnt != 0) begin
                cnt <= cnt - 16'd1;
            end else begin
                cnt <= CPB - 1;
                if (bitn == 4'd0) begin
                    if (sync[1]) busy <= 1'b0;                  // false start
                    bitn <= 4'd1;
                end else if (bitn <= 4'd8) begin
                    sh <= {sync[1], sh[7:1]}; bitn <= bitn + 4'd1;
                end else begin                                  // stop bit
                    busy <= 1'b0;
                    if (sync[1]) begin valid <= 1'b1; data <= sh; end
                end
            end
        end
    end
endmodule

module uart_tx #(parameter CPB = 234) (
    input  wire       clk,
    input  wire       rst_n,
    input  wire       send,
    input  wire [7:0] data,
    output reg        txd,
    output wire       busy
);
    reg [9:0]  sh;
    reg [3:0]  bitn;
    reg [15:0] cnt;
    reg        act;
    assign busy = act;
    always @(posedge clk) begin
        if (!rst_n) begin
            txd <= 1'b1; act <= 1'b0; sh <= 10'h3FF; bitn <= 4'd0; cnt <= 16'd0;
        end else if (!act) begin
            if (send) begin sh <= {1'b1, data, 1'b0}; act <= 1'b1; bitn <= 4'd0; cnt <= CPB - 1; txd <= 1'b0; end
        end else if (cnt != 0) begin
            cnt <= cnt - 16'd1;
        end else begin
            cnt <= CPB - 1;
            if (bitn == 4'd9) begin act <= 1'b0; txd <= 1'b1; end
            else begin bitn <= bitn + 4'd1; txd <= sh[bitn + 4'd1]; end
        end
    end
endmodule
