// SPDX-License-Identifier: Apache-2.0
// Memory-mapped wrapper that lets a CoralNPU scalar core drive the switchable matrix unit as an
// ordinary SoC peripheral (the same way Coral's own `gpio` / `spi_master` devices are attached).
// The register bus is a trivial single-cycle write / combinational-read interface; a TL-UL or AXI-Lite
// bridge (e.g. the ones in coralnpu/hdl/chisel/src/bus) sits in front of it. See coral_adapter/README.md.
//
// Register map (32-bit, word addressed by bus_addr[4:2]):
//   0x00 CTRL    RW  [0] precision_mode   0 = 1-bit XNOR/popcount, 1 = INT8   (the single mode register)
//   0x04 A_LO    RW  activation word [31:0]      (mode 0: 32 x 1-bit lanes ; mode 1: 4 x int8)
//   0x08 A_HI    RW  activation word [63:32]
//   0x0C W_LO    RW  weight word     [31:0]
//   0x10 W_HI    RW  weight word     [63:32]
//   0x14 PUSH    WO  [0] first  [1] last : writing pushes one beat (A,W) into the unit with these flags
//   0x18 RESULT  RO  accumulator of the most recent completed dot product (reading clears VALID)
//   0x1C STATUS  RO  [0] VALID (a result is waiting)   [1] BUSY (beats still in the pipeline)
// Software contract: change CTRL.precision_mode only when STATUS.BUSY == 0.

module mpnpu_mmio (
    input  wire        clk,
    input  wire        rst_n,
    input  wire        bus_we,
    input  wire        bus_re,
    input  wire [4:0]  bus_addr,        // byte address [4:0]
    input  wire [31:0] bus_wdata,
    output reg  [31:0] bus_rdata
);
    reg        precision_mode;
    reg [63:0] a_reg, w_reg;
    reg        push, push_first, push_last;
    reg [3:0]  inflight;                 // beats issued whose dot product has not completed (max 15 beats)
    reg        res_valid;
    reg signed [31:0] res;
    wire       mu_ov;
    wire signed [31:0] mu_acc;

    matrix_unit u_mu (
        .clk(clk), .rst_n(rst_n), .precision_mode(precision_mode),
        .in_valid(push), .in_first(push_first), .in_last(push_last),
        .a(a_reg), .w(w_reg), .out_valid(mu_ov), .out_acc(mu_acc)
    );

    wire [2:0] widx = bus_addr[4:2];
    wire       do_push = bus_we && (widx == 3'd5);

    always @(posedge clk) begin
        if (!rst_n) begin
            precision_mode <= 1'b0; a_reg <= 64'd0; w_reg <= 64'd0;
            push <= 1'b0; push_first <= 1'b0; push_last <= 1'b0; inflight <= 4'd0; res_valid <= 1'b0; res <= 32'sd0;
        end else begin
            push <= do_push;
            if (do_push) begin push_first <= bus_wdata[0]; push_last <= bus_wdata[1]; end
            if (bus_we) case (widx)
                3'd0: precision_mode <= bus_wdata[0];
                3'd1: a_reg[31:0]  <= bus_wdata;
                3'd2: a_reg[63:32] <= bus_wdata;
                3'd3: w_reg[31:0]  <= bus_wdata;
                3'd4: w_reg[63:32] <= bus_wdata;
                default: ;
            endcase
            // in-flight accounting: +1 per pushed beat, cleared when a dot product completes
            if (do_push && !mu_ov)       inflight <= inflight + 4'd1;
            else if (!do_push && mu_ov)  inflight <= 4'd0;
            if (mu_ov) begin res <= mu_acc; res_valid <= 1'b1; end
            else if (bus_re && widx == 3'd6) res_valid <= 1'b0;
        end
    end

    always @* begin
        case (widx)
            3'd0:    bus_rdata = {31'd0, precision_mode};
            3'd1:    bus_rdata = a_reg[31:0];
            3'd2:    bus_rdata = a_reg[63:32];
            3'd3:    bus_rdata = w_reg[31:0];
            3'd4:    bus_rdata = w_reg[63:32];
            3'd6:    bus_rdata = res;
            3'd7:    bus_rdata = {30'd0, (inflight != 4'd0) || push, res_valid};
            default: bus_rdata = 32'd0;
        endcase
    end
endmodule
