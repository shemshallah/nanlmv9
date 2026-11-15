#!/usr/bin/env python3
"""Train/evaluate the 256-expert MoE + selective-SSM prototype."""
import argparse
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from model import MambaMoE, count_params


def load_meta(data_dir):
    meta = json.load(open(os.path.join(data_dir, "meta.json")))
    return meta, torch.tensor(meta["bytes_per_token"], dtype=torch.float32)


def get_batch(arr, bs, seq, device):
    n = len(arr)
    if n <= seq + 1:
        raise SystemExit("corpus too small for seq_len")
    starts = np.random.randint(0, n - seq - 1, size=bs)
    x = np.stack([np.asarray(arr[i : i + seq]) for i in starts])
    y = np.stack([np.asarray(arr[i + 1 : i + seq + 1]) for i in starts])
    return (
        torch.from_numpy(x.astype(np.int64)).to(device),
        torch.from_numpy(y.astype(np.int64)).to(device),
    )


@torch.no_grad()
def evaluate(model, arr, bpt, seq, batch, device, max_batches=None):
    model.eval()
    ce_sum = 0.0
    byte_sum = 0.0
    tok_sum = 0
    n = len(arr)
    starts = list(range(0, n - seq - 1, seq))
    if max_batches:
        starts = starts[:max_batches]
    for start in starts:
        x = torch.from_numpy(np.asarray(arr[start : start + seq]).astype(np.int64)).to(device)[None]
        y = torch.from_numpy(np.asarray(arr[start + 1 : start + seq + 1]).astype(np.int64)).to(device)[None]
        logits, _ = model(x)
        ce = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1), reduction="sum")
        ce_sum += float(ce)
        byte_sum += float(bpt[y.reshape(-1)].sum())
        tok_sum += y.numel()
    model.train()
    if tok_sum == 0:
        return {"bpc": float("nan"), "nats_per_token": float("nan"), "tokens": 0}
    nats = ce_sum / tok_sum
    return {
        "bpc": ce_sum / math.log(2) / byte_sum,
        "nats_per_token": nats,
        "ppl": math.exp(nats),
        "tokens": tok_sum,
        "bytes": byte_sum,
        "bytes_per_token": byte_sum / tok_sum,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=os.path.expanduser("~/nanlm-data/tok32k_mixed"))
    ap.add_argument("--out", default=os.path.expanduser("~/nanlm-data/runs/mamba_moe"))
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--seq-len", type=int, default=128)
    ap.add_argument("--d-model", type=int, default=256)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--d-state", type=int, default=16)
    ap.add_argument("--experts", type=int, default=256)
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--top-k", type=int, default=2)
    ap.add_argument("--shared-hidden", type=int, default=512)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--min-lr", type=float, default=3e-5)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--aux-weight", type=float, default=0.01)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--log-interval", type=int, default=10)
    ap.add_argument("--eval-interval", type=int, default=100)
    ap.add_argument("--eval-batches", type=int, default=32)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--resume", default=None)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    if args.smoke:
        args.steps = min(args.steps, 3)
        args.d_model = min(args.d_model, 64)
        args.layers = min(args.layers, 1)
        args.experts = min(args.experts, 16)
        args.rank = min(args.rank, 4)
        args.seq_len = min(args.seq_len, 32)
        args.batch_size = min(args.batch_size, 2)
        args.eval_interval = 1
        args.eval_batches = 1
        args.log_interval = 1

    if args.threads:
        torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.out, exist_ok=True)

    meta, bpt = load_meta(args.data_dir)
    vocab_size = meta["vocab_size"]
    train_arr = np.memmap(os.path.join(args.data_dir, "train.bin"), dtype=np.uint16, mode="r")
    valid_arr = np.memmap(os.path.join(args.data_dir, "valid.bin"), dtype=np.uint16, mode="r")
    print(f"vocab={vocab_size:,} train_tokens={len(train_arr):,} valid_tokens={len(valid_arr):,}")

    model = MambaMoE(
        vocab_size,
        d_model=args.d_model,
        n_layers=args.layers,
        d_state=args.d_state,
        n_experts=args.experts,
        rank=args.rank,
        top_k=args.top_k,
        shared_hidden=args.shared_hidden,
    )
    print(f"params={count_params(model):,}")
    device = torch.device("cpu")
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    start_step = 0
    if args.resume:
        ck = torch.load(args.resume, map_location=device)
        model.load_state_dict(ck["model"])
        if "opt" in ck:
            opt.load_state_dict(ck["opt"])
        start_step = ck.get("step", 0)
        print(f"resumed at step {start_step}")

    t0 = time.time()
    model.train()
    for step in range(start_step, args.steps):
        lr = args.lr
        if step < args.warmup:
            lr = args.lr * (step + 1) / max(1, args.warmup)
        else:
            progress = (step - args.warmup) / max(1, args.steps - args.warmup)
            lr = args.min_lr + (args.lr - args.min_lr) * 0.5 * (1 + math.cos(math.pi * progress))
        for g in opt.param_groups:
            g["lr"] = lr

        opt.zero_grad(set_to_none=True)
        loss_val = 0.0
        aux_val = 0.0
        byte_count = 0
        tok_count = 0
        for _ in range(args.grad_accum):
            x, y = get_batch(train_arr, args.batch_size, args.seq_len, device)
            logits, aux = model(x)
            ce = F.cross_entropy(logits.reshape(-1, vocab_size), y.reshape(-1))
            loss = ce + args.aux_weight * aux
            (loss / args.grad_accum).backward()
            loss_val += float(ce.detach())
            aux_val += float(aux.detach())
            byte_count += float(bpt[y.reshape(-1)].sum())
            tok_count += y.numel()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        opt.step()

        if step % args.log_interval == 0 or step == args.steps - 1:
            bpc = (loss_val / args.grad_accum) / math.log(2) / (byte_count / tok_count)
            dt = time.time() - t0
            print(
                f"step {step:5d} lr {lr:.2e} ce {loss_val/args.grad_accum:.4f} "
                f"aux {aux_val/args.grad_accum:.3f} bpc {bpc:.4f} "
                f"{tok_count*args.grad_accum/dt:.0f} tok/s"
            )
        if args.eval_interval and (step + 1) % args.eval_interval == 0:
            ev = evaluate(model, valid_arr, bpt, args.seq_len, args.batch_size, device, args.eval_batches)
            print(
                f"  eval step {step+1}: bpc={ev['bpc']:.4f} nats/tok={ev['nats_per_token']:.4f} "
                f"ppl={ev['ppl']:.2f} bytes/tok={ev['bytes_per_token']:.3f}"
            )
            ck = {
                "model": model.state_dict(),
                "opt": opt.state_dict(),
                "step": step + 1,
                "config": vars(args),
                "meta": {"vocab_size": vocab_size},
                "eval": ev,
            }
            torch.save(ck, os.path.join(args.out, "last.pt"))

    ev = evaluate(model, valid_arr, bpt, args.seq_len, args.batch_size, device, args.eval_batches)
    print(f"final eval: {ev}")
    ck = {
        "model": model.state_dict(),
        "opt": opt.state_dict(),
        "step": args.steps,
        "config": vars(args),
        "meta": {"vocab_size": vocab_size},
        "eval": ev,
    }
    torch.save(ck, os.path.join(args.out, "last.pt"))
    print(f"saved {args.out}/last.pt")


if __name__ == "__main__":
    main()
