#!/usr/bin/env python3
"""Run the Mamba/MoE ablation matrix.

Each entry is a separate train.py run; results go to <out>/ablation.json.
Use `--quick` for phone-sized shape smokes and the default profile for a
GPU/Kaggle run.
"""
import argparse
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))

CONFIGS = {
    "none": {"moe_position": "none"},
    "bottom": {"moe_position": "bottom"},
    "ffn": {"moe_position": "ffn"},
    "both": {"moe_position": "both"},
    "ffn_pair": {"moe_position": "ffn", "pair_matrix": True},
    "ffn_pair_noshared": {"moe_position": "ffn", "pair_matrix": True, "no_shared": True},
    "ffn_pair_d128": {"moe_position": "ffn", "pair_matrix": True, "d_model": 128},
    "ffn_pair_d256": {"moe_position": "ffn", "pair_matrix": True, "d_model": 256},
    "ffn_pair_d512": {"moe_position": "ffn", "pair_matrix": True, "d_model": 512},
    "prior_gate": {
        "moe_position": "ffn",
        "pair_matrix": True,
        "prior": "unigram",
        "context_gate": True,
    },
    "retrieval_mm": {"moe_position": "ffn", "pair_matrix": True, "retrieval": "multimatch"},
    "retrieval_suffix": {"moe_position": "ffn", "pair_matrix": True, "retrieval": "suffix"},
    "retrieval_knn": {"moe_position": "ffn", "pair_matrix": True, "retrieval": "knn"},
    "hier_word": {"moe_position": "ffn", "pair_matrix": True, "data": "word"},
}


def build_cmd(name, cfg, args):
    cmd = [
        sys.executable,
        os.path.join(HERE, "train.py"),
        "--data-dir", cfg.get("data_dir", args.data_dir),
        "--out", os.path.join(args.out, name),
        "--result-json", os.path.join(args.out, f"{name}.json"),
        "--steps", str(args.steps),
        "--batch-size", str(args.batch),
        "--seq-len", str(args.seq_len),
        "--d-model", str(cfg.get("d_model", args.d_model)),
        "--layers", str(args.layers),
        "--experts", str(cfg.get("experts", args.experts)),
        "--rank", str(args.rank),
        "--top-k", str(args.top_k),
        "--shared-hidden", str(args.shared_hidden),
        "--eval-interval", str(args.eval_interval),
        "--eval-batches", str(args.eval_batches),
        "--log-interval", str(max(1, args.steps // 3)),
        "--threads", str(args.threads),
        "--device", args.device,
        "--moe-position", cfg.get("moe_position", "bottom"),
        "--retrieval-tokens", str(args.retrieval_tokens),
        "--retrieval-items", str(args.retrieval_items),
    ]
    if cfg.get("pair_matrix"):
        cmd.append("--pair-matrix")
    if cfg.get("no_shared"):
        cmd.append("--no-shared")
    if cfg.get("prior"):
        cmd += ["--prior", cfg["prior"], "--prior-tokens", str(args.prior_tokens)]
    if cfg.get("context_gate"):
        cmd.append("--context-gate")
    if cfg.get("retrieval"):
        cmd += ["--retrieval", cfg["retrieval"], "--retrieval-weight", str(args.retrieval_weight)]
        if cfg["retrieval"] == "knn":
            cmd += ["--retrieval-build-step", "0"]
    return cmd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=os.path.expanduser("~/nanlm-data/tok32k_mixed"))
    ap.add_argument("--word-data-dir", default=os.path.expanduser("~/nanlm-data/tok32k_wt2b"))
    ap.add_argument("--out", default=os.path.expanduser("~/nanlm-data/ablations"))
    ap.add_argument("--configs", nargs="+", default=list(CONFIGS))
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--seq-len", type=int, default=64)
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--experts", type=int, default=64)
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--top-k", type=int, default=2)
    ap.add_argument("--shared-hidden", type=int, default=256)
    ap.add_argument("--eval-interval", type=int, default=10)
    ap.add_argument("--eval-batches", type=int, default=8)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--retrieval-tokens", type=int, default=100_000)
    ap.add_argument("--retrieval-items", type=int, default=2_000)
    ap.add_argument("--retrieval-weight", type=float, default=0.3)
    ap.add_argument("--prior-tokens", type=int, default=100_000)
    args = ap.parse_args()

    if args.quick:
        args.steps = min(args.steps, 1)
        args.batch = 1
        args.seq_len = min(args.seq_len, 8)
        args.d_model = 32
        args.layers = 1
        args.experts = 4
        args.rank = 4
        args.shared_hidden = 32
        args.eval_interval = 1
        args.eval_batches = 1
        args.retrieval_tokens = 10_000
        args.retrieval_items = 128
        args.prior_tokens = 10_000

    os.makedirs(args.out, exist_ok=True)
    results = {}
    for name in args.configs:
        if name not in CONFIGS:
            print(f"unknown config {name}", file=sys.stderr)
            continue
        cfg = dict(CONFIGS[name])
        if cfg.get("data") == "word":
            if not os.path.exists(os.path.join(args.word_data_dir, "train.bin")):
                print(f"skip {name}: no word tokenizer at {args.word_data_dir}")
                continue
            cfg["data_dir"] = args.word_data_dir
            cfg.pop("data")
        cmd = build_cmd(name, cfg, args)
        log = os.path.join(args.out, f"{name}.log")
        print(f"\n=== {name} ===\n{' '.join(cmd)}")
        t0 = time.time()
        with open(log, "w") as f:
            p = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
        wall = time.time() - t0
        result_path = os.path.join(args.out, f"{name}.json")
        if p.returncode == 0 and os.path.exists(result_path):
            r = json.load(open(result_path))
            results[name] = {
                "bpc": r["eval"]["bpc"],
                "nats_per_token": r["eval"]["nats_per_token"],
                "ppl": r["eval"].get("ppl"),
                "params": r["params"],
                "wall_s": wall,
                "config": cfg,
            }
            print(f"{name}: bpc={r['eval']['bpc']:.4f} params={r['params']:,} wall={wall:.1f}s")
        else:
            results[name] = {"failed": True, "returncode": p.returncode, "log": log}
            print(f"{name}: FAILED (see {log})")

    json.dump(results, open(os.path.join(args.out, "ablation.json"), "w"), indent=1)
    print(f"\nwrote {args.out}/ablation.json")
    for name, r in sorted(results.items(), key=lambda kv: kv[1].get("bpc", 99)):
        if "bpc" in r:
            print(f"{name:24s} bpc={r['bpc']:.4f} params={r['params']:,}")


if __name__ == "__main__":
    main()
