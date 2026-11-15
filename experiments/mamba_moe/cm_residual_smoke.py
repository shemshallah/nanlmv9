#!/usr/bin/env python3
"""Real CM-prior residual smoke.

Pipeline:
  v9 CM dump (top-k byte distributions)
    + token model logits
    + token->byte trie
    -> byte-level neural/CM logits
    -> ContextGate trained on true bytes
    -> report neural-only, CM-only, and gated bpc + learned gate weights.

This is the real-logit replacement for the n-gram prior proxy.
"""
import argparse
import json
import math
import os
import re
import sys

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "v9"))

from bpe import BPE  # noqa: E402
from byte_layer import ByteTrie, CMBytePrior  # noqa: E402
from hybrid import ContextGate  # noqa: E402
from model import MambaMoE  # noqa: E402

TOKEN_RE = re.compile(r" ?\w+|\s+|[^\w\s]", re.UNICODE)
WORD_RE = re.compile(r"\w+", re.UNICODE)


def load_window_bytes(path):
    data = open(path, "rb").read()
    nw, skip = np.frombuffer(data[:8], dtype="<u4")
    out = []
    o = 8
    for _ in range(nw):
        (ln,) = np.frombuffer(data[o : o + 4], dtype="<u4")
        o += 4
        out.append(data[o + skip : o + ln])
        o += ln
    return b"".join(out)


def build_word_map(meta, bpe):
    vocab = [bytes.fromhex(h) for h in meta["vocab_hex"]]
    word_to_id = {}
    for i, b in enumerate(vocab):
        if i >= 256 + meta["bpe_merges"] and len(b) > 1:
            word_to_id[b] = i
    return vocab, word_to_id


def encode_text(text, bpe, word_to_id):
    ids = []
    for tok in TOKEN_RE.findall(text):
        b = tok.encode("utf-8", "surrogateescape")
        i = word_to_id.get(b)
        if i is not None:
            ids.append(i)
        elif WORD_RE.fullmatch(tok.lstrip(" ")):
            if tok.startswith(" "):
                ids.append(32)
                tok = tok[1:]
            ids.extend(bpe.word(tok.encode("utf-8", "surrogateescape")))
        else:
            ids.extend(b)
    return ids


def load_model(ckpt_path):
    ck = torch.load(ckpt_path, map_location="cpu")
    c = ck["config"]
    model = MambaMoE(
        ck["meta"]["vocab_size"],
        d_model=c.get("d_model", 128),
        n_layers=c.get("layers", 2),
        d_state=c.get("d_state", 16),
        n_experts=c.get("experts", 64),
        rank=c.get("rank", 8),
        top_k=c.get("top_k", 2),
        shared_hidden=c.get("shared_hidden", 256),
        moe_position=c.get("moe_position", "bottom"),
        use_shared=not c.get("no_shared", False),
        pair_matrix=c.get("pair_matrix", False),
        pair_symmetric=not c.get("pair_asymmetric", False),
        ffn_hidden=c.get("ffn_hidden"),
        local_context=c.get("local_context", 0),
        sandwich=c.get("sandwich", False),
        mirror_points=c.get("mirror_points", 256),
    )
    model.load_state_dict(ck["model"], strict=False)
    model.eval()
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=os.path.expanduser("~/nanlm-data/tok32k_mixed"))
    ap.add_argument("--ckpt", default=os.path.expanduser("~/nanlm-data/runs/quick50/last.pt"))
    ap.add_argument("--cm-dump", default=os.path.expanduser("~/nanlm-data/cm_valid8.bin"))
    ap.add_argument("--window", default=os.path.join(HERE, "..", "..", "v9", "win_valid8_4k.bin"))
    ap.add_argument("--tokens", type=int, default=512)
    ap.add_argument("--max-bytes", type=int, default=4000)
    ap.add_argument("--gate-steps", type=int, default=200)
    ap.add_argument("--gate-lr", type=float, default=0.05)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    meta = json.load(open(os.path.join(args.data_dir, "meta.json")))
    trie = ByteTrie([bytes.fromhex(h) for h in meta["vocab_hex"]])
    bpe = BPE.load(os.path.join(HERE, "..", "..", "v9", "bpe_merges.txt"))
    vocab, word_to_id = build_word_map(meta, bpe)

    raw = load_window_bytes(args.window)
    text = raw.decode("utf-8", "surrogateescape")
    ids = encode_text(text, bpe, word_to_id)
    ids = ids[: args.tokens]
    # Exact byte reconstruction check.
    recon = b"".join(vocab[i] for i in ids)
    nbytes = min(len(recon), args.max_bytes)
    print(f"raw bytes={len(raw):,} tokens={len(ids):,} recon={len(recon):,}")

    model = load_model(args.ckpt)
    with torch.no_grad():
        x = torch.tensor(ids, dtype=torch.long)[None]
        logits, _ = model(x)
        token_logits = logits[0]  # [L, V]
    prior = CMBytePrior.load(args.cm_dump)
    print(f"CM prior: n={prior.n:,} k={prior.k}")

    nn_logits, cm_logits, true_bytes, feats = [], [], [], []
    byte_pos = len(vocab[ids[0]]) if ids else 0
    for i in range(len(ids) - 1):
        target = ids[i + 1]
        tbytes = vocab[target]
        if byte_pos + len(tbytes) > nbytes:
            break
        prefix = b""
        for j, b in enumerate(tbytes):
            d, mass = trie.byte_distribution(token_logits[i], prefix)
            if float(mass) <= 0:
                break
            cm = prior.logits(byte_pos + j)
            nn_logits.append(torch.log(d.clamp_min(1e-12)))
            cm_logits.append(cm)
            true_bytes.append(b)
            feats.append([
                float(d.max()), float(-(d * torch.log(d.clamp_min(1e-12))).sum()),
                float(torch.exp(cm).max()), float(-(torch.exp(cm) * cm).sum()),
                j / max(1, len(tbytes)),
            ])
            prefix += bytes([b])
        byte_pos += len(tbytes)
        if len(true_bytes) >= args.max_bytes:
            break

    Xn = torch.stack(nn_logits)
    Xc = torch.stack(cm_logits)
    y = torch.tensor(true_bytes, dtype=torch.long)
    feats = torch.tensor(feats)
    extra = feats[:, -1:]
    print(f"collected bytes={len(y):,} nn={Xn.shape} cm={Xc.shape}")

    def bpc(logits):
        return float(F.cross_entropy(logits, y) / math.log(2))

    print(f"neural-only bpc={bpc(Xn):.4f}  cm-only bpc={bpc(Xc):.4f}")

    gate = ContextGate(n_sources=2, hidden=64, extra_features=1)
    opt = torch.optim.Adam(gate.parameters(), lr=args.gate_lr)
    for step in range(args.gate_steps):
        mixed, w = gate([Xn, Xc], extra=extra)
        loss = F.cross_entropy(mixed, y)
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 50 == 0:
            print(f"gate step {step:3d} bpc={bpc(mixed.detach()):.4f} w_cm={w[:,1].mean():.4f}")
    with torch.no_grad():
        mixed, w = gate([Xn, Xc], extra=extra)
        print(f"gated bpc={bpc(mixed):.4f} mean weights nn={w[:,0].mean():.3f} cm={w[:,1].mean():.3f}")
        print(f"w_cm min={w[:,1].min():.3f} max={w[:,1].max():.3f}")
    if args.out:
        json.dump({
            "bytes": len(y),
            "neural_bpc": bpc(Xn),
            "cm_bpc": bpc(Xc),
            "gated_bpc": bpc(mixed),
            "mean_w_nn": float(w[:,0].mean()),
            "mean_w_cm": float(w[:,1].mean()),
        }, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
