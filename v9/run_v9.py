#!/usr/bin/env python3
"""NaN v9 runner.

dev  : train(shakespeare+wt2_train) -> eval wt2_valid windows
final: train(shakespeare+wt2_train+wt2_valid) -> eval wt2_test windows

Reports nats, bpc, byte/word/sentence top1/top5 (+SE) for frozen and online
modes.  The C core can also save/load a checkpoint (all learned tables, match
model, mixers and APMs) with -save/-load so CI can publish trained weights.
"""
import argparse
import json
import math
import os
import re
import struct
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
WORD = re.compile(rb"[A-Za-z0-9<>']+")


def windows(path):
    f = open(path, "rb").read()
    nw, skip = struct.unpack("<II", f[:8])
    o = 8
    ws = []
    for _ in range(nw):
        (l,) = struct.unpack("<I", f[o:o + 4])
        o += 4
        ws.append(f[o:o + l])
        o += l
    return ws, skip


def score(winfile, resfile):
    ws, skip = windows(winfile)
    R = np.frombuffer(
        open(resfile, "rb").read(),
        dtype=[("nll", "<f4"), ("t1", "u1"), ("t5", "u1"), ("p", "u1", 2)],
    )
    nll, t1, t5 = R["nll"].astype(float), R["t1"].astype(float), R["t5"].astype(float)
    words, sents, k = [], [], 0
    for w in ws:
        n = len(w) - skip
        a1 = np.zeros(len(w), bool)
        a5 = np.zeros(len(w), bool)
        a1[skip:] = t1[k:k + n] > 0
        a5[skip:] = t5[k:k + n] > 0
        k += n
        for m in WORD.finditer(w):
            a, b = m.span()
            if a >= skip:
                words.append((a1[a:b].all(), a5[a:b].all()))
        s = skip
        for m in re.finditer(rb"[.!?] ", w):
            e = m.end()
            if e - s >= 8 and s >= skip:
                sents.append(
                    (a1[s:e].all(), a5[s:e].all(), a1[s:e].mean(), a5[s:e].mean())
                )
            s = e
    se = lambda x: float(x.std(ddof=1) / math.sqrt(len(x)))
    W = np.array(words, float)
    S = np.array(sents, float)
    return {
        "n_bytes": len(nll),
        "nats": float(nll.mean()),
        "nats_se": se(nll),
        "bpc": float(nll.mean() / math.log(2)),
        "byte_top1": float(t1.mean()),
        "byte_top1_se": se(t1),
        "byte_top5": float(t5.mean()),
        "byte_top5_se": se(t5),
        "n_words": len(W),
        "word_top1": float(W[:, 0].mean()),
        "word_top5": float(W[:, 1].mean()),
        "n_sents": len(S),
        "sent_top1_strict": float(S[:, 0].mean()),
        "sent_top5_strict": float(S[:, 1].mean()),
        "sent_mean_byte_acc_top1": float(S[:, 2].mean()),
        "sent_mean_byte_acc_top5": float(S[:, 3].mean()),
    }


def main(
    mode="dev",
    corpora=None,
    outdir=None,
    tshift=-1,
    save=None,
    load=None,
    maxb=None,
    tag=None,
    evalonly=False,
    nancm=None,
    extra=(),
    windows="v9",
    evmatch=False,
    evcarry=False,
):
    if mode not in ("dev", "final"):
        raise SystemExit(f"unknown mode {mode!r} (use dev or final)")
    C = os.path.abspath(corpora or os.path.join(HERE, "..", "corpora"))
    outdir = os.path.abspath(outdir or HERE)
    os.makedirs(outdir, exist_ok=True)
    tag = tag or f"v9_{mode}"
    nancm = os.path.abspath(nancm or os.path.join(HERE, "nancm"))
    t0 = time.time()

    if mode == "dev":
        train = [f"{C}/tinyshakespeare.txt", f"{C}/wt2_train.txt"]
        if windows == "4k":
            evals = {"valid64_4k": f"{HERE}/win_valid64_4k.bin"}
        else:
            evals = {"valid40": f"{HERE}/win_valid40.bin"}
    else:
        train = [
            f"{C}/tinyshakespeare.txt",
            f"{C}/wt2_train.txt",
            f"{C}/wt2_valid.txt",
        ]
        if windows == "4k":
            evals = {"test64_4k": f"{HERE}/win_test64_4k.bin"}
        else:
            evals = {
                "test10": f"{HERE}/win_test10.bin",
                "test40": f"{HERE}/win_test40.bin",
            }

    for p in ([] if evalonly else train):
        if not os.path.exists(p):
            raise SystemExit(f"missing corpus: {p}")
    if not os.path.exists(nancm):
        raise SystemExit(f"missing nancm binary: {nancm} (run make)")

    out = {
        "version": tag,
        "mode": mode,
        "train_files": [] if evalonly else [os.path.basename(t) for t in train],
        "bpe_merges": sum(1 for _ in open(f"{HERE}/bpe_merges.txt")),
        "config": {
            "tshift": tshift,
            "evalonly": evalonly,
            "load": load,
            "save": save,
            "maxb": maxb,
            "windows": windows,
            "evmatch": evmatch,
            "evcarry": evcarry,
        },
    }
    for name, wf in evals.items():
        cmd = [
            nancm,
            "-bpe", f"{HERE}/bpe_merges.txt",
            "-lr1", "0.0015",
            "-tshift", str(tshift),
            "-eval", wf,
            "-evalat", "1000000,3000000,6000000",
            "-o", f"{outdir}/{tag}_{name}",
        ]
        if load:
            cmd += ["-load", os.path.abspath(load)]
        if save:
            cmd += ["-save", os.path.abspath(save)]
        if maxb:
            cmd += ["-maxb", str(maxb)]
        if evmatch:
            cmd += ["-evmatch"]
        if evcarry:
            cmd += ["-evcarry"]
        cmd += list(extra)
        if not evalonly:
            cmd += train
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode:
            sys.stderr.write(r.stdout)
            sys.stderr.write(r.stderr)
            raise SystemExit(f"nancm failed for {name} with exit code {r.returncode}")
        curve = [l for l in r.stdout.splitlines() if l.startswith("EVAL")]
        out[name] = {
            "frozen": score(wf, f"{outdir}/{tag}_{name}_frozen.bin"),
            "online": score(wf, f"{outdir}/{tag}_{name}_online.bin"),
            "curve_lines": curve,
        }
        if mode == "dev":
            break  # one eval set only in dev

    out["wall_s"] = time.time() - t0
    metrics_path = os.path.join(outdir, f"metrics_{tag}.json")
    with open(metrics_path, "w") as f:
        json.dump(out, f, indent=1)

    for name in [k for k in out if k.startswith(("valid", "test"))]:
        for m in ("frozen", "online"):
            e = out[name][m]
            print(
                f"{name:8s} {m:7s} nats={e['nats']:.4f}±{e['nats_se']:.3f} "
                f"bpc={e['bpc']:.4f} B top1={e['byte_top1']*100:.2f}% "
                f"top5={e['byte_top5']*100:.2f}% | "
                f"W({e['n_words']}) top1={e['word_top1']*100:.1f}% "
                f"top5={e['word_top5']*100:.1f}% | "
                f"S({e['n_sents']}) acc1={e['sent_mean_byte_acc_top1']:.3f} "
                f"acc5={e['sent_mean_byte_acc_top5']:.3f} "
                f"strict={e['sent_top1_strict']:.2f}/{e['sent_top5_strict']:.2f}"
            )
    print(f"metrics: {metrics_path}")
    return out


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "score":
        if len(sys.argv) != 4:
            raise SystemExit("usage: run_v9.py score <windows.bin> <results.bin>")
        print(json.dumps(score(sys.argv[2], sys.argv[3]), indent=1))
    else:
        ap = argparse.ArgumentParser(description=__doc__)
        ap.add_argument("mode", nargs="?", default="dev", choices=("dev", "final"))
        ap.add_argument("--corpora", default=None)
        ap.add_argument("--out", default=None, help="output directory")
        ap.add_argument("--tshift", type=int, default=-1)
        ap.add_argument("--save", default=None, help="write trained checkpoint")
        ap.add_argument("--load", default=None, help="start from checkpoint")
        ap.add_argument("--maxb", type=int, default=None)
        ap.add_argument("--tag", default=None)
        ap.add_argument("--evalonly", action="store_true", help="skip training files")
        ap.add_argument("--nancm", default=None)
        ap.add_argument("--windows", choices=("v9", "4k"), default="v9")
        ap.add_argument("--evmatch", action="store_true", help="match model may use training history")
        ap.add_argument("--evcarry", action="store_true", help="carry byte state across online windows")
        args, extra = ap.parse_known_args()
        main(
            mode=args.mode,
            corpora=args.corpora,
            outdir=args.out,
            tshift=args.tshift,
            save=args.save,
            load=args.load,
            maxb=args.maxb,
            tag=args.tag,
            evalonly=args.evalonly,
            nancm=args.nancm,
            extra=extra,
            windows=args.windows,
            evmatch=args.evmatch,
            evcarry=args.evcarry,
        )
