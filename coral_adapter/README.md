# Coral NPU integration notes

Status: `mpnpu_mmio.sv` is **tested in isolation** (`make sim-mmio`: both modes through the register
interface). It has **not** been built into a Coral SoC; the Coral-side snippets below come from reading the
repository, not from a successful Coral build.

## Option A (recommended): attach as a SoC peripheral

Coral's SoC attaches devices such as `gpio` and `spi_master` as TL-UL peripherals. The same pattern fits here.

1. **Bridge.** Put a TL-UL (or AXI-Lite) to register-bus bridge in front of `mpnpu_mmio` (Coral's `hdl/chisel/src/bus`
   has TL-UL pieces to build on) and wrap `mpnpu_mmio.sv` + `matrix_unit.sv` as a Chisel `BlackBox`.
2. **Declare the device** in `hdl/chisel/src/soc/SoCChiselConfig.scala`, next to the `gpio` entry:
   ```scala
   ChiselModuleConfig(
     name = "mpnpu",
     moduleClass = "bus.MpnpuPeripheral",          // your wrapper from step 1
     params = ...,
     deviceConnections = Map("io.tl" -> "mpnpu"),
     externalPorts = Seq()
   ),
   ```
3. **Give it an address range** in `hdl/chisel/src/soc/CrossbarConfig.scala`, next to
   `DeviceConfig("gpio", Seq(AddressRange(0x40030000, 0x1000)))`, for example
   `DeviceConfig("mpnpu", Seq(AddressRange(0x40040000, 0x1000)))`, and add `"mpnpu"` to the host-to-device
   connection lists where `"gpio"` appears.
4. **Drive it from C** on the Coral scalar core with plain loads and stores (register map in the header of
   `mpnpu_mmio.sv`): write `CTRL` (precision), write the A and W words, write `PUSH` with first/last flags,
   poll `STATUS.VALID`, read `RESULT`.

## Option B: replace the INT8 lane inside the Zvt matrix engine

This is the "modify the Coral matrix unit" route. Files that would change, found by reading the repo:

| File | What to change |
|---|---|
| `hdl/verilog/rvv/design/Zvt/zvt_pe_mulbulk_int_lane.sv` | The INT8 multiplier lane (32-bit operands, 4x4 partial-product array, `use_dsp`). This is where a mode-switched datapath would go. |
| `Zvt/zvt_pe_adder_int_lane.sv`, `zvt_pe_adder.sv` | Accumulation; a popcount path for 1-bit mode. |
| `Zvt/zvt_pe_array.sv`, `zvt_pe_block.sv`, `zvt_pe_mulbulk.sv` | PE array structure and interfaces. |
| `Zvt/zvt_ctrl.sv`, `Zvt/zvt.sv` | Command decode: add the precision field (a new `mtype` value or a CSR). |
| `hdl/verilog/rvv/inc/rvv_backend_define.svh` | Array dimensions (`NUM_BLK`, `TE`, `SUBTILE_SIZE`). |
| `hdl/chisel/src/coralnpu/scalar/Csr.scala` | A custom `precision_mode` CSR. |
| `hdl/chisel/src/coralnpu/Parameters.scala` | Memory sizes and variant flags. |

Why `matrix_unit` is **not** a drop-in for the Zvt lane: `zvt_pe_mulbulk_int_lane` is an element-wise lane that
produces a matrix of partial products for outer-product style GEMM, with its own operand format, masking and
status ports. `matrix_unit` is a streaming dot-product engine (64 binary lanes or 8 INT8 lanes per beat). Bridging
them means mapping Zvt's tile operands onto beats and reworking the accumulate stage, and that needs the full
Coral toolchain (Bazel, Verilator, the Zvt testbenches) to validate. That work is not done here.

Not recommended on this FPGA: the full `VmeCoreMini` including the RVV backend is far beyond 8,640 LUT4s.
