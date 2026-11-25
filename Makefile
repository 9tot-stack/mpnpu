# SPDX-License-Identifier: Apache-2.0
# Run from the project root (the $readmemh paths are relative to it).
RTL  = rtl/mem.sv rtl/matrix_unit.sv rtl/uart.sv rtl/mpnpu_core.sv rtl/mpnpu_top.sv
PY  ?= python3

.PHONY: all train test sim-unit sim-mmio sim-top synth synth-nodsp clean
all: test

train:                     ## train + fold + export weights/thresholds/layer table + test vectors (~8 min on 2 CPU cores)
	$(PY) -W ignore sw/train_export.py --epochs 24 --lr 2e-3 --logit-scale 10 --bs 128

sim-unit:                  ## randomised check of the 1-bit / INT8 matrix unit
	iverilog -g2012 -o build/tb_mu rtl/matrix_unit.sv tb/tb_matrix_unit.sv && vvp build/tb_mu | tail -3

sim-mmio:                  ## Coral-style register wrapper
	iverilog -g2012 -o build/tb_mmio rtl/matrix_unit.sv coral_adapter/mpnpu_mmio.sv tb/tb_mmio.sv && vvp build/tb_mmio | tail -3

sim-top:                   ## full UART->inference->UART run, bit-exact against sw/golden.py (about 1 min)
	iverilog -g2012 -o build/tb_top $(RTL) tb/tb_top.sv && vvp build/tb_top | tail -8

test: sim-unit sim-mmio sim-top

synth:                     ## Yosys synth_gowin (infers MULT9X9 DSPs + BSRAM) -> resource report in build/yosys.log
	yowasp-yosys -q -l build/yosys.log -p "read_verilog -sv $(RTL); synth_gowin -top mpnpu_top -json build/mpnpu.json; stat"
	@grep -E "^ +[0-9]+ +(MULT9X9|SPX9|DPX9B|ALU|LUT[1-4]|DFF)" build/yosys.log | tail -12

synth-nodsp:               ## same, multipliers in LUTs
	yowasp-yosys -q -l build/yosys_nodsp.log -p "read_verilog -sv $(RTL); synth_gowin -nodsp -top mpnpu_top -json build/mpnpu_nodsp.json; stat"

# NOTE: the open-source nextpnr-gowin build tested here cannot place the DSP (MULT9X9) or the SPX9
# BSRAM primitives on GW1N-9C, so a bitstream is produced with the Gowin EDA project in gowin/.
clean:
	rm -f build/*.json build/tb_* build/*.log
