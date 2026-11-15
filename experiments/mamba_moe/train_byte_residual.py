#!/usr/bin/env python3
"""Train a Mamba/MoE token model through the real CM byte residual.

This is the full-model version of cm_residual_smoke.py: token logits ->
token->byte trie -> gated CM prior -> byte NLL -> backward into the model.
"""
import argparse, json, math, os, sys
import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "v9"))
from bpe import BPE
from byte_layer import ByteTrie, CMBytePrior
from experimental import ByteResidualLoss
from hybrid import ContextGate
from model import MambaMoE, count_params
from cm_residual_smoke import load_window_bytes, build_word_map, encode_text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=os.path.expanduser("~/nanlm-data/tok32k_mixed"))
    ap.add_argument("--cm-dump", required=True)
    ap.add_argument("--window", default=os.path.join(HERE, "..", "..", "v9", "win_valid8_4k.bin"))
    ap.add_argument("--out", default=os.path.expanduser("~/nanlm-data/runs/byte_residual"))
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--seq", type=int, default=32)
    ap.add_argument("--d-model", type=int, default=64)
    ap.add_argument("--layers", type=int, default=1)
    ap.add_argument("--experts", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    meta = json.load(open(os.path.join(args.data_dir, "meta.json")))
    vocab = [bytes.fromhex(h) for h in meta["vocab_hex"]]
    trie = ByteTrie(vocab)
    prior = CMBytePrior.load(args.cm_dump)
    bpe = BPE.load(os.path.join(HERE, "..", "..", "v9", "bpe_merges.txt"))
    _, word_to_id = build_word_map(meta, bpe)
    raw = load_window_bytes(args.window)
    ids = encode_text(raw.decode("utf-8", "surrogateescape"), bpe, word_to_id)
    recon = b"".join(vocab[i] for i in ids)
    offsets = []
    pos = 0
    for tid in ids:
        offsets.append(pos)
        pos += len(vocab[tid])
    assert recon == raw[:len(recon)]
    seq = min(args.seq, len(ids) - 1)
    print(f"tokens={len(ids):,} bytes={len(recon):,} seq={seq} cm={prior.n:,}")

    model = MambaMoE(meta["vocab_size"], d_model=args.d_model, n_layers=args.layers,
                     d_state=8, n_experts=args.experts, rank=4, top_k=2,
                     shared_hidden=2 * args.d_model, moe_position="ffn",
                     pair_matrix=True, sandwich=False)
    gate = ContextGate(n_sources=2, hidden=64, extra_features=1)
    loss_fn = ByteResidualLoss(trie, prior, gate, vocab)
    params = list(model.parameters()) + list(gate.parameters())
    opt = torch.optim.AdamW(params, lr=args.lr)
    print("params", f"{count_params(model):,}", "gate", f"{count_params(gate):,}")

    for step in range(args.steps):
        start = int(torch.randint(0, len(ids) - seq - 1, (1,)))
        x = torch.tensor(ids[start:start + seq], dtype=torch.long)[None]
        y = torch.tensor(ids[start + 1:start + seq + 1], dtype=torch.long)[None]
        logits, aux = model(x)
        # byte offsets for target tokens (shifted by one)
        bo = torch.tensor(offsets[start + 1:start + seq + 1], dtype=torch.long)
        loss, nbytes = loss_fn(logits[0], y[0], bo, extra=torch.zeros(seq, 1))
        total = loss + 0.01 * (aux if torch.is_tensor(aux) else 0.0)
        opt.zero_grad(); total.backward(); torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step()
        if step % 5 == 0:
            print(f"step {step:3d} byte_bpc={float(loss)/math.log(2):.4f} bytes={nbytes}")
    torch.save({"model": model.state_dict(), "gate": gate.state_dict(),
                "config": vars(args), "meta": {"vocab_size": meta["vocab_size"]}},
               os.path.join(args.out, "last.pt"))
    print("saved", os.path.join(args.out, "last.pt"))


if __name__ == "__main__":
    main()
