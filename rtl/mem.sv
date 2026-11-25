// SPDX-License-Identifier: Apache-2.0
// Synchronous memories written to infer Gowin BSRAM (registered read, one cycle latency).

module sdp_ram #(parameter W = 16, parameter D = 1024, parameter AW = 10) (
    input  wire          clk,
    input  wire          we,
    input  wire [AW-1:0] waddr,
    input  wire [W-1:0]  wdata,
    input  wire [AW-1:0] raddr,
    output reg  [W-1:0]  rdata
);
    reg [W-1:0] mem [0:D-1];
    always @(posedge clk) begin
        if (we) mem[waddr] <= wdata;
        rdata <= mem[raddr];
    end
endmodule

// Read-only memory initialised from a $readmemh file (model weights, thresholds, layer table).
module rom_sync #(parameter W = 32, parameter D = 512, parameter AW = 9, parameter FILE = "") (
    input  wire          clk,
    input  wire [AW-1:0] addr,
    output reg  [W-1:0]  rdata
);
    reg [W-1:0] mem [0:D-1];
    generate if (FILE != "") begin : g_init
        initial $readmemh(FILE, mem);
    end endgenerate
    always @(posedge clk) rdata <= mem[addr];
endmodule
