#!/usr/bin/env python3
"""Train/evaluate the 256-expert MoE + selective-SSM prototype."""
import argparse
import json
import math
import os
import time

import numpy as np
import subprocess
import torch
import torch.nn.functional as F

from model import MambaMoE, count_params
from hybrid import NGramPrior, ContextGate, mixture_logits
from presets import PRESETS
from retrieval import build_retriever, KNNRetriever


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


def fuse_retrieval(logits, rl, mode="add", weight=0.5):
    if mode == "confidence":
        ent = -(torch.softmax(logits, -1) * torch.log_softmax(logits, -1)).sum(-1)
        conf = torch.softmax(rl, -1).max(-1).values
        w = torch.sigmoid(2.0 * (conf - ent)) * weight
        return logits + w[:, None] * rl
    return logits + weight * rl


def retrieval_contexts(x, order):
    b, l = x.shape
    ctx = torch.zeros(b, l, order, dtype=torch.long, device=x.device)
    for t in range(l):
        s = max(0, t - order + 1)
        ctx[:, t, order - (t - s + 1) :] = x[:, s : t + 1]
        if s == 0 and t + 1 < order:
            ctx[:, t, : order - (t + 1)] = x[:, :1]
    return ctx.reshape(-1, order)


@torch.no_grad()
def evaluate(model, arr, bpt, seq, batch, device, max_batches=None, prior=None, gate=None,
             prior_weight=1.0, retriever=None, retrieval_weight=0.5, retrieval_order=4,
             retrieval_fusion="add"):
    model.eval()
    bpt = bpt.to(device)
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
        logits = logits.reshape(-1, logits.shape[-1])
        if prior is not None:
            pl = prior.logits(x.reshape(-1)).to(device)
            extra = torch.linspace(0, 1, logits.shape[0], device=device)[:, None]
            mixed, _ = mixture_logits(logits, pl, gate, prior_weight, extra=extra)
            logits = mixed
        if retriever is not None:
            if isinstance(retriever, KNNRetriever):
                h = model.encode(x)
                rl = retriever.logits(h).to(device)
            else:
                rl = retriever.logits(retrieval_contexts(x, retrieval_order)).to(device)
            logits = fuse_retrieval(logits, rl, retrieval_fusion, retrieval_weight)
        ce = F.cross_entropy(logits, y.reshape(-1), reduction="sum")
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


def build_knn(retriever, model, arr, seq, device, max_items=50_000):
    hs, ys = [], []
    n = len(arr)
    for start in range(0, n - seq - 1, seq):
        x = torch.from_numpy(np.asarray(arr[start : start + seq]).astype(np.int64)).to(device)[None]
        y = torch.from_numpy(np.asarray(arr[start + 1 : start + seq + 1]).astype(np.int64)).to(device)[None]
        h = model.encode(x)
        hs.append(h.reshape(-1, h.shape[-1]).cpu())
        ys.append(y.reshape(-1).cpu())
        if sum(len(v) for v in hs) >= max_items:
            break
    retriever.build(torch.cat(hs), torch.cat(ys))
    print(f"kNN datastore: {retriever.keys.shape[0]:,} states")


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
    ap.add_argument("--moe-position", choices=("none", "bottom", "ffn", "both"), default="bottom")
    ap.add_argument("--no-shared", action="store_true", help="disable the dense shared expert")
    ap.add_argument("--pair-matrix", action="store_true", help="enable E x E expert-pair adjacency")
    ap.add_argument("--pair-asymmetric", action="store_true", help="do not symmetrize the pair matrix")
    ap.add_argument("--ffn-hidden", type=int, default=None)
    ap.add_argument("--local-context", type=int, default=0)
    ap.add_argument("--sandwich", action="store_true")
    ap.add_argument("--mirror-points", type=int, default=256)
    ap.add_argument("--scan", choices=("sequential", "chunked"), default="sequential")
    ap.add_argument("--scan-chunk", type=int, default=16)
    ap.add_argument("--prior", choices=("none", "unigram", "bigram"), default="none")
    ap.add_argument("--prior-weight", type=float, default=1.0)
    ap.add_argument("--prior-tokens", type=int, default=2_000_000)
    ap.add_argument("--context-gate", action="store_true", help="learn neural mixture weights for prior/neural")
    ap.add_argument("--gate-hidden", type=int, default=64)
    ap.add_argument("--pair-reg", type=float, default=1e-4)
    ap.add_argument("--retrieval", choices=("none", "multimatch", "multimatch_multi", "suffix", "knn"), default="none")
    ap.add_argument("--retrieval-tokens", type=int, default=500_000)
    ap.add_argument("--retrieval-items", type=int, default=50_000)
    ap.add_argument("--retrieval-weight", type=float, default=0.5)
    ap.add_argument("--retrieval-build-step", type=int, default=1)
    ap.add_argument("--retrieval-order", type=int, default=4)
    ap.add_argument("--retrieval-fusion", choices=("add", "confidence"), default="add")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--min-lr", type=float, default=3e-5)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--aux-weight", type=float, default=0.01)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--device", default="cpu", help="cpu, cuda, or auto")
    ap.add_argument("--log-interval", type=int, default=10)
    ap.add_argument("--eval-interval", type=int, default=100)
    ap.add_argument("--eval-batches", type=int, default=32)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--resume", default=None)
    ap.add_argument("--result-json", default=None)
    ap.add_argument("--preset", choices=("10m", "100m"), default=None)
    ap.add_argument("--preset-smoke", action="store_true")
    ap.add_argument("--max-seconds", type=float, default=None)
    ap.add_argument("--cm-dump", default=None, help="run gated CM residual eval after training")
    ap.add_argument("--cm-window", default=os.path.join(HERE, "..", "..", "v9", "win_valid8_4k.bin"))
    ap.add_argument("--cm-bytes", type=int, default=1000)
    ap.add_argument("--cm-gate-steps", type=int, default=100)
    ap.add_argument("--amp", action="store_true", help="CUDA mixed precision")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    if args.preset:
        for k, v in PRESETS[args.preset].items():
            setattr(args, k, v)
        print(f"preset={args.preset}")
        if args.preset_smoke:
            args.steps = 1
            args.batch_size = 1
            args.seq_len = 16
            args.eval_interval = 1
            args.eval_batches = 1

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
        moe_position=args.moe_position,
        use_shared=not args.no_shared,
        pair_matrix=args.pair_matrix,
        pair_symmetric=not args.pair_asymmetric,
        ffn_hidden=args.ffn_hidden,
        local_context=args.local_context,
        sandwich=args.sandwich,
        mirror_points=args.mirror_points,
        scan_mode=args.scan,
        scan_chunk=args.scan_chunk,
    )
    print(f"params={count_params(model):,}")
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"device={device}")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
    model.to(device)
    bpt = bpt.to(device)

    prior = None
    if args.prior != "none":
        print(f"building {args.prior} prior over {args.prior_tokens:,} tokens")
        prior = NGramPrior.from_tokens(
            train_arr,
            order=1 if args.prior == "unigram" else 2,
            vocab_size=vocab_size,
            max_tokens=args.prior_tokens,
        )
    gate = None
    if args.context_gate:
        gate = ContextGate(n_sources=2, hidden=args.gate_hidden, extra_features=1).to(device)
        print(f"context gate params={count_params(gate):,}")

    retriever = build_retriever(args.retrieval, train_arr, vocab_size, args.retrieval_tokens)
    if retriever is not None:
        print(f"retrieval={args.retrieval}")
    if isinstance(retriever, KNNRetriever) and args.retrieval_build_step <= 0:
        build_knn(retriever, model, train_arr, args.seq_len, device, args.retrieval_items)

    params = list(model.parameters()) + (list(gate.parameters()) if gate is not None else [])
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)

    start_step = 0
    if args.resume:
        ck = torch.load(args.resume, map_location=device)
        model.load_state_dict(ck["model"])
        if gate is not None and "gate" in ck:
            gate.load_state_dict(ck["gate"])
        if "opt" in ck:
            opt.load_state_dict(ck["opt"])
        start_step = ck.get("step", 0)
        print(f"resumed at step {start_step}")

    t0 = time.time()
    model.train()
    for step in range(start_step, args.steps):
        if args.max_seconds and (time.time() - t0) >= args.max_seconds:
            print(f"time budget reached at step {step}")
            break
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
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(args.amp and device.type == "cuda")):
                logits, aux = model(x)
            flat_logits = logits.reshape(-1, vocab_size).float()
            flat_y = y.reshape(-1)
            if prior is not None:
                pl = prior.logits(x.reshape(-1)).to(device)
                extra = torch.linspace(0, 1, flat_logits.shape[0], device=device)[:, None]
                mixed, _ = mixture_logits(flat_logits, pl, gate, args.prior_weight, extra=extra)
                ce = F.cross_entropy(mixed, flat_y)
            else:
                ce = F.cross_entropy(flat_logits, flat_y)
            pair_reg = 0.0
            for m in model.modules():
                if hasattr(m, "pair_reg"):
                    pair_reg = pair_reg + m.pair_reg
            loss = ce + args.aux_weight * aux + args.pair_reg * pair_reg
            (loss / args.grad_accum).backward()
            loss_val += float(ce.detach())
            aux_val += float(aux.detach()) if torch.is_tensor(aux) else float(aux)
            byte_count += float(bpt[flat_y].sum())
            tok_count += flat_y.numel()
        torch.nn.utils.clip_grad_norm_(params, args.grad_clip)
        opt.step()
        if (
            isinstance(retriever, KNNRetriever)
            and retriever.keys is None
            and step >= args.retrieval_build_step
        ):
            build_knn(retriever, model, train_arr, args.seq_len, device, args.retrieval_items)

        if step % args.log_interval == 0 or step == args.steps - 1:
            bpc = (loss_val / args.grad_accum) / math.log(2) / (byte_count / tok_count)
            dt = time.time() - t0
            print(
                f"step {step:5d} lr {lr:.2e} ce {loss_val/args.grad_accum:.4f} "
                f"aux {aux_val/args.grad_accum:.3f} bpc {bpc:.4f} "
                f"{tok_count*args.grad_accum/dt:.0f} tok/s"
            )
        if args.eval_interval and (step + 1) % args.eval_interval == 0:
            ev = evaluate(model, valid_arr, bpt, args.seq_len, args.batch_size, device, args.eval_batches,
                          prior, gate, args.prior_weight, retriever, args.retrieval_weight, args.retrieval_order, args.retrieval_fusion)
            print(
                f"  eval step {step+1}: bpc={ev['bpc']:.4f} nats/tok={ev['nats_per_token']:.4f} "
                f"ppl={ev['ppl']:.2f} bytes/tok={ev['bytes_per_token']:.3f}"
            )
            ck = {
                "model": model.state_dict(),
                "gate": gate.state_dict() if gate is not None else None,
                "opt": opt.state_dict(),
                "step": step + 1,
                "config": vars(args),
                "meta": {"vocab_size": vocab_size},
                "eval": ev,
            }
            torch.save(ck, os.path.join(args.out, "last.pt"))

    ev = evaluate(model, valid_arr, bpt, args.seq_len, args.batch_size, device, args.eval_batches,
                  prior, gate, args.prior_weight, retriever, args.retrieval_weight, args.retrieval_order, args.retrieval_fusion)
    print(f"final eval: {ev}")
    ck = {
        "model": model.state_dict(),
        "gate": gate.state_dict() if gate is not None else None,
        "opt": opt.state_dict(),
        "step": args.steps,
        "config": vars(args),
        "meta": {"vocab_size": vocab_size},
        "eval": ev,
    }
    torch.save(ck, os.path.join(args.out, "last.pt"))
    if args.result_json:
        import json as _json
        _json.dump(
            {
                "eval": ev,
                "params": count_params(model),
                "config": vars(args),
            },
            open(args.result_json, "w"),
            indent=1,
        )
    print(f"saved {args.out}/last.pt")
    if args.cm_dump:
        cmd = [sys.executable, os.path.join(HERE, "cm_residual_smoke.py"),
               "--ckpt", os.path.join(args.out, "last.pt"),
               "--data-dir", args.data_dir, "--cm-dump", args.cm_dump,
               "--window", args.cm_window, "--max-bytes", str(args.cm_bytes),
               "--gate-steps", str(args.cm_gate_steps),
               "--out", os.path.join(args.out, "cm_gate.json")]
        print("running gated CM residual:", " ".join(cmd))
        subprocess.run(cmd, check=False)


if __name__ == "__main__":
    main()
