# Gowin EDA project (Tang Nano 9K)

`mpnpu.gprj` was written by hand from the Gowin project format; **it has not been opened in the Gowin IDE**
(no Gowin tools in the build environment). If the IDE rejects it, create a new project instead:

1. Device: **GW1NR-LV9QN88PC6/I5** (family GW1N-9C, "GW1NR-9C" in the IDE list).
2. Add `../rtl/*.sv` (all five files), `mpnpu.cst`, `mpnpu.sdc`. Top module: `mpnpu_top`.
3. Project > Configuration > Synthesize > Verilog Language: **System Verilog 2017**.
4. The weight/threshold/layer images are loaded with `$readmemh`. They are looked up as
   `rtl/mem/<file>.hex` relative to the *project working directory*. If the IDE cannot find them, set the
   global macro `MEM_DIR` (Project > Configuration > Synthesize > Global Macros / Verilog defines) to the
   absolute or project-relative folder of `rtl/mem/`, with a trailing slash, e.g. `MEM_DIR="C:/work/mpnpu/rtl/mem/"`.
5. Process > Place & Route, then Programmer (SRAM for tests, embedded flash to persist).

Pin map (cross-checked against Sipeed/Lushay Labs examples, verify against the schematic):
clk=52 (27 MHz), S1 reset=4, UART TX=17 / RX=18 (via the on-board BL702), LEDs=10,11,13,14,15,16.
