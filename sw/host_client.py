#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""PC side of the USB-UART accelerator protocol. Needs only pyserial + numpy (no GPU, no ML framework).

    python3 host_client.py --port /dev/ttyUSB1 --probe
    python3 host_client.py --port /dev/ttyUSB1 --mnist data/t10k-images-idx3-ubyte.gz \
                           --labels data/t10k-labels-idx1-ubyte.gz -n 200

Protocol (8N1): 0xA5 + 98 packed-bit bytes -> 4 bytes [0x5A, class, cycles_lo, cycles_hi]; 0xA4 -> "NPU1".
The FPGA reports its own compute time in 27 MHz clock cycles; the script also measures wall-clock
round-trip time so you can see how much of the latency is UART transfer.
"""
import argparse, gzip, struct, time
import numpy as np

CLK_HZ = 27_000_000


def pack_image(img_u8, thr=128):
    bits = (np.asarray(img_u8).reshape(-1) >= thr).astype(np.uint8)
    out = np.zeros(98, dtype=np.uint8)
    for i, b in enumerate(bits):
        out[i // 8] |= int(b) << (i % 8)
    return bytes(out)


class NpuDevice:
    def __init__(self, port, baud=115200):
        import serial
        self.s = serial.Serial(port, baud, timeout=1.0)

    def probe(self):
        self.s.reset_input_buffer(); self.s.write(b"\xA4")
        return self.s.read(4) == b"NPU1"

    def infer(self, img_u8):
        self.s.write(b"\xA5" + pack_image(img_u8))
        r = self.s.read(4)
        if len(r) != 4 or r[0] != 0x5A:
            raise IOError(f"bad reply {r!r}")
        return r[1], r[2] | (r[3] << 8)          # class, compute cycles


def load_idx(path):
    d = gzip.open(path).read()
    nd = d[3]
    dims = struct.unpack(">" + "I" * nd, d[4:4 + 4 * nd])
    return np.frombuffer(d, np.uint8, offset=4 + 4 * nd).reshape(dims)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", required=True)
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--mnist"); ap.add_argument("--labels"); ap.add_argument("-n", type=int, default=100)
    a = ap.parse_args()
    dev = NpuDevice(a.port, a.baud)
    if a.probe:
        print("device answered NPU1" if dev.probe() else "no / wrong answer"); return
    imgs, labels = load_idx(a.mnist), load_idx(a.labels)
    ok, rt, cyc = 0, [], []
    for i in range(a.n):
        t0 = time.perf_counter()
        c, cy = dev.infer(imgs[i])
        rt.append(time.perf_counter() - t0); cyc.append(cy); ok += int(c == labels[i])
    print(f"accuracy {ok}/{a.n} = {ok / a.n:.3f}")
    print(f"on-chip compute: {np.mean(cyc):.0f} cycles = {np.mean(cyc) / CLK_HZ * 1e3:.3f} ms")
    print(f"host round trip: mean {np.mean(rt) * 1e3:.2f} ms, max {np.max(rt) * 1e3:.2f} ms  (baud {a.baud})")


if __name__ == "__main__":
    main()
