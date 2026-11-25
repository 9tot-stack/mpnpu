# SPDX-License-Identifier: Apache-2.0
"""Bit-exact integer reference of the FPGA inference, decoded from the packed memory images.

It reads the same hex files the RTL loads (wmem0/1, thr, bias, layers), so a packing bug in
the exporter shows up as a mismatch against the RTL or the float model.
"""
import os
import numpy as np


def _rd(path, width_bits, signed=False):
    vals = [int(l.strip(), 16) for l in open(path) if l.strip()]
    if signed:
        s = 1 << (width_bits - 1)
        vals = [(v ^ s) - s for v in vals]
    return vals


def _field(d, lo, n):
    return (d >> lo) & ((1 << n) - 1)


class Params:
    def __init__(self, w64, thr, bias, descs):
        self.w64, self.thr, self.bias, self.descs = w64, thr, bias, descs
        self.layers = []
        for d in descs:
            L = dict(mode=_field(d, 0, 1), src=_field(d, 1, 1), pix16=_field(d, 2, 1), sink=_field(d, 3, 1),
                     in_w=_field(d, 4, 5), pout_w=_field(d, 9, 4), pout_h=_field(d, 13, 4), out_c=_field(d, 17, 7),
                     nchunk=_field(d, 24, 6), wbase=_field(d, 30, 10), tbase=_field(d, 40, 8), last=_field(d, 48, 1))
            self.layers.append(L)
            if L["last"]:
                break

    @classmethod
    def from_files(cls, d):
        lo = _rd(os.path.join(d, "wmem0.hex"), 32)
        hi = _rd(os.path.join(d, "wmem1.hex"), 32)
        w64 = [(h << 32) | l for l, h in zip(lo, hi)]
        return cls(w64, _rd(os.path.join(d, "thr.hex"), 8), _rd(os.path.join(d, "bias.hex"), 32, True),
                   _rd(os.path.join(d, "layers.hex"), 64))

    def conv_lanes(self, L):
        n = L["out_c"] * L["nchunk"]
        out = np.zeros((L["out_c"], L["nchunk"] * 64), dtype=np.float32)
        for oc in range(L["out_c"]):
            for ch in range(L["nchunk"]):
                v = self.w64[L["wbase"] + oc * L["nchunk"] + ch]
                out[oc, ch * 64:(ch + 1) * 64] = [(v >> i) & 1 for i in range(64)]
        return out

    def dense_w(self, L):
        W = np.zeros((L["out_c"], L["nchunk"] * 8), dtype=np.int64)
        for j in range(L["out_c"]):
            for k in range(L["nchunk"]):
                v = self.w64[L["wbase"] + j * L["nchunk"] + k]
                for i in range(8):
                    b = (v >> (8 * i)) & 0xFF
                    W[j, 8 * k + i] = b - 256 if b >= 128 else b
        return W


def _conv_pool(act, lanes_w, thr, L):
    """act: [N, H, W, C] bits. Returns pooled bits [N, ph, pw, out_c] (OR of thresholded 2x2 conv outputs)."""
    N, H, W, C = act.shape
    oh, ow = H - 2, W - 2
    win = np.zeros((N, oh, ow, L["nchunk"] * 64), dtype=np.float32)       # pad lanes stay 0 (act=0, weight pad=1 -> mismatch)
    for kr in range(3):
        for kc in range(3):
            p = kr * 3 + kc
            win[..., p * C:(p + 1) * C] = act[:, kr:kr + oh, kc:kc + ow, :]
    # matches = #lanes where act == weight bit  (XNOR + popcount)
    m = win.reshape(-1, win.shape[-1]) @ lanes_w.T + (1 - win.reshape(-1, win.shape[-1])) @ (1 - lanes_w).T
    m = np.rint(m).astype(np.int64).reshape(N, oh, ow, L["out_c"])
    bit = (m >= np.asarray(thr[L["tbase"]:L["tbase"] + L["out_c"]])[None, None, None, :]).astype(np.uint8)
    ph, pw = L["pout_h"], L["pout_w"]
    bit = bit[:, :2 * ph, :2 * pw, :].reshape(N, ph, 2, pw, 2, L["out_c"])
    return bit.max(axis=(2, 4))


def infer_batch(P, img_bits, chunk=500):
    """img_bits: [N, 28, 28] uint8 in {0,1}. Returns (class[N], counts[N,256], best_logit[N])."""
    l1, l2, l3 = P.layers
    w1, w2, W3 = P.conv_lanes(l1), P.conv_lanes(l2), P.dense_w(l3)
    bias = np.asarray(P.bias[:l3["out_c"]], dtype=np.int64)
    cls, cnts, best = [], [], []
    for s in range(0, len(img_bits), chunk):
        x = img_bits[s:s + chunk, :, :, None].astype(np.float32)
        a1 = _conv_pool(x, w1, P.thr, l1)
        a2 = _conv_pool(a1.astype(np.float32), w2, P.thr, l2)
        # four overlapping regions, same rule as the RTL: top=py<=s, bottom=py>=s, left=px<=s, right=px>=s (s=pout>>1)
        sr, sc = l2["pout_h"] >> 1, l2["pout_w"] >> 1
        rows = [(0, sr + 1), (sr, l2["pout_h"])]; cols = [(0, sc + 1), (sc, l2["pout_w"])]
        cnt = np.concatenate([a2[:, r0:r1, c0:c1, :].sum(axis=(1, 2)) for r0, r1 in rows for c0, c1 in cols], axis=1).astype(np.int64)  # [N, 256], idx=region*64+c
        logit = cnt @ W3.T + bias[None, :]
        cls.append(logit.argmax(1))                                            # first max wins, same as RTL strict '>'
        best.append(logit.max(1)); cnts.append(cnt)
    return np.concatenate(cls), np.concatenate(cnts), np.concatenate(best)
