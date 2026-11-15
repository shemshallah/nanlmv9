#!/usr/bin/env python3
"""Verify that the C core and bpe.py produce identical BPE segmentations.

The C -segdump path splits words longer than 48 bytes into 48-byte chunks
(the context window cap), so this checker mirrors that rule exactly.
"""
import argparse
import os
import re
import struct
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
WORD = re.compile(rb"[A-Za-z0-9\x80-\xff]+")
PRE = re.compile(rb"[A-Za-z0-9\x80-\xff]+|[^A-Za-z0-9\x80-\xff]")


def c_like_encode(bpe, data):
    ids = []
    for tok in PRE.findall(data):
        if WORD.fullmatch(tok):
            for off in range(0, len(tok), 48):
                ids.extend(bpe.word(tok[off:off + 48]))
        else:
            ids.append(tok[0])
    return ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nancm", default=os.path.join(HERE, "..", "v9", "nancm"))
    ap.add_argument("--merges", default=os.path.join(HERE, "..", "v9", "bpe_merges.txt"))
    ap.add_argument("--sample", default=os.path.join(HERE, "..", "v9", "corpora", "wt2_test.txt"))
    ap.add_argument("--bytes", type=int, default=65536)
    args = ap.parse_args()

    sys.path.insert(0, os.path.join(HERE, "..", "v9"))
    from bpe import BPE

    data = open(args.sample, "rb").read()[:args.bytes]
    bpe = BPE.load(args.merges)
    py = c_like_encode(bpe, data)

    with tempfile.TemporaryDirectory() as td:
        sample = os.path.join(td, "sample.bin")
        out = os.path.join(td, "c.bin")
        open(sample, "wb").write(data)
        r = subprocess.run(
            [args.nancm, "-segdump", args.merges, sample, out],
            capture_output=True,
            text=True,
        )
        if r.returncode:
            sys.stderr.write(r.stdout + r.stderr)
            raise SystemExit("nancm -segdump failed")
        raw = open(out, "rb").read()
    c = list(struct.unpack("<%dI" % (len(raw) // 4), raw))

    if py != c:
        n = min(len(py), len(c))
        i = next((k for k in range(n) if py[k] != c[k]), n)
        raise SystemExit(
            f"BPE mismatch at token {i}: python={py[i:i+4]} c={c[i:i+4]}"
        )
    print(f"BPE parity OK: {len(data)} bytes, {len(py)} tokens, {len(bpe.merges)} merges")


if __name__ == "__main__":
    main()
