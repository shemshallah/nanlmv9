# nanlmv9

[![v9 dev release](https://github.com/shemshallah/nanlmv9/actions/workflows/dev-release.yml/badge.svg)](https://github.com/shemshallah/nanlmv9/actions/workflows/dev-release.yml)

NaN **v9** is a byte-level context-mixing language model (PAQ/lpaq family)
whose contexts include BPE subword features.  The C core predicts one byte at
a time, so bits-per-byte (bpc) is unambiguous.  This repository holds the v9
source, the frozen evaluation windows, a development-release runner, and the
trained checkpoints published by that runner.

## Published v9 dev result

`v9/metrics_v9_dev.json` is the reference run: train on Tiny Shakespeare +
WikiText-2 train, evaluate 40 held-out WikiText-2 valid windows
(16,000 scored bytes), `-tshift -1`.

| mode | bpc | byte top-1 | byte top-5 | word top-1 |
|---|---:|---:|---:|---:|
| frozen | 1.5962 | 67.63% | 88.41% | 14.32% |
| online | 1.5556 | 68.06% | 88.56% | 15.52% |

These numbers are the target for development releases.  The CI default
`-tshift 2` uses smaller hash tables so the published checkpoint stays a
reasonable size; its expected dev result is:

| `-tshift` | valid40 frozen bpc | checkpoint |
|---|---:|---:|
| `-1` (reference) | 1.5962 | largest (~1.6 GiB RAM) |
| `2` (CI default) | 1.6493 | ~273 MiB |

Dispatch the workflow with `tshift=-1` to reproduce the exact reference
configuration (expect a much larger checkpoint and slower upload).

## Match-history evaluation (new)

The original eval reset `mm_floor = histlen` at every window, so the match
model could never reach the training corpus.  `-evmatch` allows training-history
matches (frozen eval restores the match tables per window to prevent
cross-window leakage), and `-evcarry` carries byte context across online
windows.

Measured on the CI checkpoint (`-tshift 2`) with 64 × 4,096-byte windows
(229,376 scored bytes):

| eval | old | `-evmatch -evcarry` | gain |
|---|---:|---:|---:|
| wt2 valid, frozen | 1.5967 | **1.5697** | 0.0270 |
| wt2 valid, online | 1.5049 | **1.4948** | 0.0101 |
| wt2 test, frozen | — | **1.5589** | — |
| wt2 test, online | — | **1.4907** | — |

For reference on the same valid split: gzip-9 = 2.66 bpc, zstd-19 = 2.20 bpc,
xz-9e = 2.18 bpc.

```sh
# 4k windows + match-history/online-carry evaluation
python3 v9/run_v9.py dev --evalonly \
  --load out/nanlmv9-weights.ckpt --windows 4k \
  --evmatch --evcarry --out out/eval4k --tag eval4k
```

## Hybrid experiments

`experiments/mamba_moe/` contains the 32k word+BPE tokenizer, a 256-expert
low-rank MoE + selective-SSM prototype, and the NaNROUTER data pipeline.  It
documents the 256-register / 32³+32³ pair-space design, the SSM recurrence, and
the measured CPU limits (the full neural run needs a GPU).

## Quick start

```sh
# 1. public corpora (checksum-verified)
bash scripts/fetch_corpora.sh v9/corpora

# 2. build the C core
make -C v9

# 3. verify C and Python BPE segmentation are identical
python3 scripts/verify_bpe.py --nancm v9/nancm \
  --merges v9/bpe_merges.txt --sample v9/corpora/wt2_test.txt

# 4. dev run: train + evaluate valid40, save weights
python3 v9/run_v9.py dev \
  --corpora v9/corpora --out out --tshift 2 \
  --save out/nanlmv9-weights.ckpt
```

Termux works with `pkg install clang python numpy` (or the existing Termux
toolchain); the Makefile honours `CC`/`CFLAGS`.

### Evaluate saved weights only

```sh
python3 v9/run_v9.py dev --evalonly \
  --load out/nanlmv9-weights.ckpt \
  --out out/loaded --tag loaded
```

Or directly with the C core:

```sh
v9/nancm -bpe v9/bpe_merges.txt \
  -load nanlmv9-weights.ckpt \
  -eval v9/win_valid40.bin -noonline -o out/loaded
```

The checkpoint contains the complete learned state: all context tables, the
match model, mixers, and APMs.  History is stored too, so a loaded checkpoint
can continue training (`-load ... -save ... train.txt`) or run online eval.
The matching `bpe_merges.txt` is required for checkpoints trained with BPE
contexts.

## Development releases

The `v9 dev release` workflow (`.github/workflows/dev-release.yml`):

1. fetches the public corpora and verifies SHA-256;
2. builds `nancm` and verifies C/Python BPE parity;
3. trains on Tiny Shakespeare + WikiText-2 train, evaluates `valid40`, and
   saves the checkpoint;
4. publishes a **prerelease** tagged `dev-v9-<run>-<sha>` with:
   - `nanlmv9-weights-<tag>.ckpt` — trained weights
   - `nancm-linux-x86_64` — compiled core
   - `bpe_merges.txt`, `win_valid40.bin`
   - `metrics_dev.json`, `manifest.json`, `train-eval.log`
   - `nanlmv9-source-<tag>.tar.gz` and `SHA256SUMS`
5. prunes older `dev-v9-*` releases, keeping the newest N (default 3).

Stable/non-`dev-v9-*` releases are never pruned.  A manual
`prune dev releases` workflow is available for dry runs or explicit cleanup.

## Layout

```
v9/nancm.c          C context-mixing core (+ -save/-load checkpoints)
v9/bpe.py           reference BPE trainer/encoder
v9/bpe_merges.txt   frozen 2,000-merge BPE table
v9/run_v9.py        train/eval harness and scorer
v9/win_*.bin        frozen evaluation windows
experiments/mamba_moe/  32k tokenizer + 256-expert MoE/SSM prototype
scripts/            corpora fetch, BPE parity check, release pruning
.github/workflows/  dev release + manual prune
```

## Reproducibility notes

- Corpora are not committed; `scripts/fetch_corpora.sh` pins their SHA-256.
- `-tshift -1` = reference quality (largest tables, ~1.6 GiB RAM).
- `-tshift 1` = larger tables, moderate checkpoint.
- `-tshift 2` = CI default, compact checkpoint.
- `-tshift 3` = fastest/smallest, lower quality.
- Everything is deterministic for a fixed input order and build; floating
  point reductions can differ slightly across compilers.

Licensed under Apache-2.0 (see `LICENSE`).
