#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Train a mixed-precision MNIST model with Brevitas and export it for the FPGA.

Network (all integer semantics are mirrored bit-exactly by sw/golden.py and the RTL):

    28x28 binarized image (bit=1 -> +1)
      conv3x3 1->16   (1-bit weights, Brevitas binary quantizer)  -> BN -> sign -> maxpool2   13x13x16
      conv3x3 16->64  (1-bit weights)                              -> BN -> sign -> maxpool2   5x5x64
      per-channel count of +1 over four overlapping 3x3 regions of the 5x5 map (0..9 each)
                                                                   -> 256 int8 activations
      dense   256->10 (8-bit weights, Brevitas Int8 quantizer)     -> argmax

The four region counts keep coarse spatial layout. (A single global count per channel was tried
first and plateaued at ~88% on MNIST; the regions lift it to ~97% for +2.5 KB of int8 weights.)

BN + sign is folded into one integer threshold per output channel on the XNOR-popcount
match count; a negative BN scale is handled by inverting that channel's weight bits.

Usage:  python3 train_export.py --epochs 24
"""
import argparse, gzip, json, math, os, struct, sys, time
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from brevitas.nn import QuantConv2d, QuantLinear
from brevitas.quant.binary import SignedBinaryWeightPerTensorConst
from brevitas.quant import Int8WeightPerTensorFloat

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


# ----------------------------------------------------------------------------- data
def load_idx(path):
    with gzip.open(path) as f:
        d = f.read()
    magic = struct.unpack(">I", d[:4])[0]
    nd = magic & 0xFF
    dims = struct.unpack(">" + "I" * nd, d[4:4 + 4 * nd])
    return np.frombuffer(d, dtype=np.uint8, offset=4 + 4 * nd).reshape(dims)


def load_mnist(data_dir):
    p = lambda n: os.path.join(data_dir, n)
    return (load_idx(p("train-images-idx3-ubyte.gz")), load_idx(p("train-labels-idx1-ubyte.gz")),
            load_idx(p("t10k-images-idx3-ubyte.gz")), load_idx(p("t10k-labels-idx1-ubyte.gz")))


# ----------------------------------------------------------------------------- model
class SignSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return torch.where(x >= 0, torch.ones_like(x), -torch.ones_like(x))

    @staticmethod
    def backward(ctx, g):
        (x,) = ctx.saved_tensors
        return g * (x.abs() <= 1).to(g.dtype)


class Net(nn.Module):
    # (row0,row1,col0,col1) slices of the 5x5 pooled map; mirrored by the RTL rule top=py<=2, bottom=py>=2, ...
    REGIONS = ((0, 3, 0, 3), (0, 3, 2, 5), (2, 5, 0, 3), (2, 5, 2, 5))

    def __init__(self, c1=16, c2=64, logit_scale=1.0):
        super().__init__()
        self.logit_scale = logit_scale   # training-only; a positive scale never changes argmax
        bq = SignedBinaryWeightPerTensorConst
        self.conv1 = QuantConv2d(1, c1, 3, bias=False, weight_quant=bq)
        self.bn1 = nn.BatchNorm2d(c1)
        self.conv2 = QuantConv2d(c1, c2, 3, bias=False, weight_quant=bq)
        self.bn2 = nn.BatchNorm2d(c2)
        self.fc = QuantLinear(c2 * len(self.REGIONS), 10, bias=True, weight_quant=Int8WeightPerTensorFloat)

    def counts(self, x):
        y = F.max_pool2d(SignSTE.apply(self.bn1(self.conv1(x))), 2)
        y = F.max_pool2d(SignSTE.apply(self.bn2(self.conv2(y))), 2)
        b = (y + 1) * 0.5
        return torch.cat([b[:, :, r0:r1, c0:c1].sum((2, 3)) for r0, r1, c0, c1 in self.REGIONS], 1)  # idx = region*c2 + c, 0..9

    def forward(self, x):
        return self.fc((self.counts(x) - 4.5) / 4.5) * self.logit_scale


def to_pm1(imgs_u8, thr=128):
    return torch.from_numpy((imgs_u8 >= thr).astype(np.float32) * 2 - 1).unsqueeze(1)


def evaluate(model, X, y, bs=1000):
    model.eval()
    correct = 0
    with torch.no_grad():
        for i in range(0, len(X), bs):
            correct += (model(X[i:i + bs]).argmax(1).numpy() == y[i:i + bs]).sum()
    return correct / len(X)


def train(model, Xtr, ytr, Xte, yte, epochs, bs=256, lr=4e-3, log=print):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    steps = epochs * math.ceil(len(Xtr) / bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps)
    yt = torch.from_numpy(ytr.astype(np.int64))
    for ep in range(epochs):
        model.train(); t0 = time.time()
        perm = torch.randperm(len(Xtr)); tot = 0.0
        for i in range(0, len(Xtr), bs):
            idx = perm[i:i + bs]
            loss = F.cross_entropy(model(Xtr[idx]), yt[idx])
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            with torch.no_grad():                    # standard BNN latent-weight clipping
                model.conv1.weight.clamp_(-1, 1); model.conv2.weight.clamp_(-1, 1)
            tot += loss.item() * len(idx)
        log(f"epoch {ep + 1}/{epochs} loss {tot / len(Xtr):.4f} test_acc {evaluate(model, Xte, yte):.4f} ({time.time() - t0:.0f}s)")


# ----------------------------------------------------------------------------- BN fold + packing
def fold_conv(conv, bn):
    """Return (w_bits[oc, ic, 3, 3] in {0,1} after flip, thr[oc] int, K) for match-count thresholds."""
    qw = conv.quant_weight()
    w = qw.int().detach().numpy().astype(np.int64)            # +-1
    scale = float(qw.scale.detach().reshape(-1)[0])
    g = bn.weight.detach().double().numpy(); b = bn.bias.detach().double().numpy()
    mu = bn.running_mean.detach().double().numpy(); var = bn.running_var.detach().double().numpy()
    sig = np.sqrt(var + bn.eps)
    A = g * scale / sig
    B = b - g * mu / sig
    K = w.shape[1] * 9
    thr = np.zeros(len(A), dtype=np.int64)
    for c in range(len(A)):
        if A[c] > 0:
            thr[c] = math.ceil((-B[c] / A[c] + K) / 2 - 1e-9)
        elif A[c] < 0:
            w[c] = -w[c]                                       # invert weights: dot' = -dot
            thr[c] = math.ceil((B[c] / A[c] + K) / 2 - 1e-9)
        else:
            thr[c] = 0 if B[c] >= 0 else 255
    thr = np.clip(thr, 0, 255)                                 # m is 0..K<=144 so 0 == always, 255 == never
    return (w > 0).astype(np.uint8), thr, K


def conv_lanes(wbits):
    """[oc, ic, 3, 3] -> [oc, nchunk*64] lane vector, lane = (kr*3+kc)*ic + c, pad lanes = 1."""
    oc, ic = wbits.shape[:2]
    lanes = wbits.transpose(0, 2, 3, 1).reshape(oc, 9 * ic)   # (kr,kc,c) order
    nch = math.ceil(9 * ic / 64)
    out = np.ones((oc, nch * 64), dtype=np.uint8)
    out[:, :9 * ic] = lanes
    return out, nch


def lanes_to_words(lanes):
    """[oc, nch*64] bits -> list of 64-bit ints, oc-major / chunk-minor."""
    words = []
    for row in lanes:
        for ch in range(len(row) // 64):
            v = 0
            for i in range(64):
                v |= int(row[ch * 64 + i]) << i
            words.append(v)
    return words


def dense_words(Wint):
    words = []
    for j in range(Wint.shape[0]):
        for k in range(Wint.shape[1] // 8):
            v = 0
            for i in range(8):
                v |= (int(Wint[j, 8 * k + i]) & 0xFF) << (8 * i)
            words.append(v)
    return words


def make_desc(mode, src_sel, pix16, sink_cnt, in_w, pout_w, pout_h, out_c, nchunk, wbase, tbase, last):
    d = (mode & 1) | (src_sel & 1) << 1 | (pix16 & 1) << 2 | (sink_cnt & 1) << 3
    d |= (in_w & 31) << 4 | (pout_w & 15) << 9 | (pout_h & 15) << 13 | (out_c & 127) << 17
    d |= (nchunk & 63) << 24 | (wbase & 1023) << 30 | (tbase & 255) << 40 | (last & 1) << 48
    return d


def pack_image_bits(bits28):
    flat = bits28.reshape(-1).astype(np.uint8)
    out = np.zeros(98, dtype=np.uint8)
    for i, b in enumerate(flat):
        out[i // 8] |= int(b) << (i % 8)
    return out


def write_hex(path, vals, width_bits):
    digits = (width_bits + 3) // 4
    with open(path, "w") as f:
        for v in vals:
            f.write(f"{int(v) & ((1 << width_bits) - 1):0{digits}x}\n")


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=24)
    ap.add_argument("--config", default=os.path.join(HERE, "model_config.json"))
    ap.add_argument("--data", default=os.path.join(ROOT, "data"))
    ap.add_argument("--mem-out", default=os.path.join(ROOT, "rtl", "mem"))
    ap.add_argument("--vec-out", default=os.path.join(ROOT, "tb", "vectors"))
    ap.add_argument("--ckpt", default=os.path.join(ROOT, "sw", "out", "model.pt"))
    ap.add_argument("--n-vectors", type=int, default=64)
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--logit-scale", type=float, default=10.0)
    ap.add_argument("--bs", type=int, default=128)
    a = ap.parse_args()
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    cfg = json.load(open(a.config))
    L = cfg["layers"]
    # ---- validate the layer config against what the sequencer supports
    assert [l["type"] for l in L] == ["conv3x3_pool2", "conv3x3_pool2", "dense"], "sequencer supports conv,conv,dense"
    for l in L:
        want = "8bit" if l["type"] == "dense" else "1bit"
        if l["precision"] != want:
            sys.exit(f"layer {l['name']}: precision {l['precision']} unsupported for type {l['type']} (need {want})")
    os.makedirs(a.mem_out, exist_ok=True); os.makedirs(a.vec_out, exist_ok=True)
    os.makedirs(os.path.dirname(a.ckpt), exist_ok=True)

    Xtr_u8, ytr, Xte_u8, yte = load_mnist(a.data)
    thr_px = cfg["input"]["binarize_threshold"]
    Xtr, Xte = to_pm1(Xtr_u8, thr_px), to_pm1(Xte_u8, thr_px)
    model = Net(L[0]["out_c"], L[1]["out_c"], a.logit_scale)
    if a.skip_train and os.path.exists(a.ckpt):
        model.load_state_dict(torch.load(a.ckpt))
    else:
        train(model, Xtr, ytr, Xte, yte, a.epochs, bs=a.bs, lr=a.lr)
        torch.save(model.state_dict(), a.ckpt)
    model.eval()
    float_pred = np.concatenate([model(Xte[i:i + 1000]).argmax(1).detach().numpy() for i in range(0, 10000, 1000)])
    acc_float = float((float_pred == yte).mean())
    print(f"float (fake-quant) test accuracy: {acc_float:.4f}")

    # ---- fold + pack
    w1, t1, K1 = fold_conv(model.conv1, model.bn1)
    w2, t2, K2 = fold_conv(model.conv2, model.bn2)
    lanes1, nch1 = conv_lanes(w1)
    lanes2, nch2 = conv_lanes(w2)
    qfc = model.fc.quant_weight()
    Wint = qfc.int().detach().numpy().astype(np.int64)         # [10, 64], int8 range
    s_w = float(qfc.scale.detach().reshape(-1)[0])
    bfc = model.fc.bias.detach().double().numpy()
    a_coef = s_w / 4.5
    bias_int = np.round(bfc / a_coef - 4.5 * Wint.sum(1)).astype(np.int64)

    words = lanes_to_words(lanes1) + lanes_to_words(lanes2) + dense_words(Wint)
    wb1, wb2, wb3 = 0, len(lanes1) * nch1, len(lanes1) * nch1 + len(lanes2) * nch2
    assert len(words) <= 1024
    words += [0] * (1024 - len(words))
    write_hex(os.path.join(a.mem_out, "wmem0.hex"), [w & 0xFFFFFFFF for w in words], 32)
    write_hex(os.path.join(a.mem_out, "wmem1.hex"), [w >> 32 for w in words], 32)
    thr_all = list(t1) + list(t2); thr_all += [0] * (128 - len(thr_all))
    write_hex(os.path.join(a.mem_out, "thr.hex"), thr_all, 8)
    write_hex(os.path.join(a.mem_out, "bias.hex"), list(bias_int) + [0] * (16 - len(bias_int)), 32)

    c1, c2, fc = L
    descs = [
        make_desc(0, 0, 0, 0, 28, 13, 13, c1["out_c"], nch1, wb1, 0, 0),
        make_desc(0, 1, 1, 1, 13, 5, 5, c2["out_c"], nch2, wb2, len(t1), 0),
        make_desc(1, 0, 0, 0, 0, 0, 0, fc["out_c"], fc["in_c"] // 8, wb3, 0, 1),
    ]
    write_hex(os.path.join(a.mem_out, "layers.hex"), descs + [0] * (8 - len(descs)), 64)

    # ---- golden integer model decoded from the PACKED words (covers the packing too)
    import golden
    P = golden.Params.from_files(a.mem_out)
    pred_g, cnt_g, best_g = golden.infer_batch(P, (Xte_u8 >= thr_px).astype(np.uint8))
    acc_int = float((pred_g == yte).mean())
    agree = float((pred_g == float_pred).mean())
    print(f"integer golden model test accuracy: {acc_int:.4f}   agreement with float model: {agree:.4f}")

    # ---- test vectors for the RTL testbench
    n = a.n_vectors
    bits = (Xte_u8[:n] >= thr_px).astype(np.uint8)
    img_bytes = np.concatenate([pack_image_bits(b) for b in bits])
    write_hex(os.path.join(a.vec_out, "tv_img.hex"), img_bytes, 8)
    write_hex(os.path.join(a.vec_out, "tv_class.hex"), pred_g[:n], 8)
    write_hex(os.path.join(a.vec_out, "tv_best.hex"), best_g[:n], 32)
    write_hex(os.path.join(a.vec_out, "tv_cnt.hex"), cnt_g[:n].reshape(-1), 8)   # 256 per image
    write_hex(os.path.join(a.vec_out, "tv_label.hex"), yte[:n], 8)

    # ---- size report
    nfc = Wint.shape[1]
    base8 = 16 * 9 + 64 * 144 + nfc * 10                                    # all-8-bit weights, unpadded
    mixed_unpadded = (16 * 9 + 64 * 144) // 8 + nfc * 10
    mixed_stored = 8 * (len(lanes1) * nch1 + len(lanes2) * nch2 + 10 * (nfc // 8))   # 64-bit words actually stored
    rep = dict(acc_float=acc_float, acc_int=acc_int, float_int_agreement=agree,
               weight_bytes_all8bit_unpadded=base8, weight_bytes_mixed_unpadded=mixed_unpadded,
               weight_bytes_mixed_as_stored=mixed_stored,
               reduction_vs_all8_unpadded=1 - mixed_unpadded / base8, reduction_vs_all8_as_stored=1 - mixed_stored / base8,
               thr_bytes=len(t1) + len(t2), bias_bytes=4 * len(bias_int), thresholds_conv1=t1.tolist(), thresholds_conv2=t2.tolist(),
               fc_weight_scale=s_w, binary_weight_scale=float(model.conv1.quant_weight().scale.reshape(-1)[0]))
    os.makedirs(os.path.join(ROOT, "docs"), exist_ok=True)
    json.dump(rep, open(os.path.join(ROOT, "docs", "model_report.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in rep.items() if not k.startswith("thresholds")}, indent=1))


if __name__ == "__main__":
    sys.path.insert(0, HERE)
    main()
