#!/usr/bin/env python3
"""32k hybrid word+BPE tokenizer with exact byte fallback.

IDs:
  0..255            raw bytes
  256..256+M-1      BPE merge tokens (M = merges in v9/bpe_merges.txt)
  ...               top whole words from the vocabulary corpus

Frequent words become one token; rare words use the v9 BPE table; anything
else falls back to raw bytes.  The tokenization is a bijection on bytes, so
bits-per-byte is exact.
"""
import argparse
import collections
import json
import os
import re
import sys
from array import array

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "v9"))
from bpe import BPE  # noqa: E402

TOKEN_RE = re.compile(r" ?\w+|\s+|[^\w\s]", re.UNICODE)
WORD_RE = re.compile(r"\w+", re.UNICODE)


def iter_chunks(path, max_bytes=None):
    remaining = max_bytes
    with open(path, "rb") as f:
        while True:
            chunk = f.read(1 << 20)
            if not chunk:
                break
            if remaining is not None:
                if len(chunk) > remaining:
                    chunk = chunk[:remaining]
                remaining -= len(chunk)
            yield chunk.decode("utf-8", "surrogateescape")
            if remaining is not None and remaining <= 0:
                break


def build_vocab(paths, bpe, vocab_size, sample_bytes):
    word_counts = collections.Counter()
    for path in paths:
        for text in iter_chunks(path, sample_bytes):
            for w in TOKEN_RE.findall(text):
                if WORD_RE.fullmatch(w.lstrip(" ")):
                    word_counts[w.encode("utf-8", "surrogateescape")] += 1

    vocab = [bytes([i]) for i in range(256)]
    seen = set(vocab)
    for r in range(len(bpe.merges)):
        b = bpe.vocab[256 + r]
        if b not in seen:
            seen.add(b)
            vocab.append(b)
    word_to_id = {}
    for w, _ in word_counts.most_common():
        if len(vocab) >= vocab_size:
            break
        if len(w) <= 1 or w in seen:
            continue
        seen.add(w)
        word_to_id[w] = len(vocab)
        vocab.append(w)
    return vocab, word_to_id


def encode(text, bpe, word_to_id):
    ids = array("H")
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


def encode_file(src, dst, bpe, word_to_id, max_bytes=None):
    out = array("H")
    total = 0
    with open(dst, "wb") as f:
        for text in iter_chunks(src, max_bytes):
            out.extend(encode(text, bpe, word_to_id))
            total += len(text.encode("utf-8", "surrogateescape"))
            if len(out) >= 1 << 20:
                out.tofile(f)
                out = array("H")
        if out:
            out.tofile(f)
    return total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab-texts", nargs="+", required=True)
    ap.add_argument("--train-texts", nargs="+", required=True)
    ap.add_argument("--valid-texts", nargs="+", default=[])
    ap.add_argument("--test-texts", nargs="+", default=[])
    ap.add_argument("--out", required=True)
    ap.add_argument("--vocab-size", type=int, default=32000)
    ap.add_argument("--vocab-sample-bytes", type=int, default=50_000_000)
    ap.add_argument("--max-train-bytes", type=int, default=None)
    ap.add_argument("--merges", default=os.path.join(HERE, "..", "..", "v9", "bpe_merges.txt"))
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    bpe = BPE.load(args.merges)
    vocab, word_to_id = build_vocab(args.vocab_texts, bpe, args.vocab_size, args.vocab_sample_bytes)
    meta = {
        "vocab_size": len(vocab),
        "bpe_merges": len(bpe.merges),
        "vocab_hex": [b.hex() for b in vocab],
        "bytes_per_token": [len(b) for b in vocab],
        "vocab_texts": args.vocab_texts,
        "train_texts": args.train_texts,
        "valid_texts": args.valid_texts,
        "test_texts": args.test_texts,
    }

    for name, paths, maxb in (
        ("train", args.train_texts, args.max_train_bytes),
        ("valid", args.valid_texts, None),
        ("test", args.test_texts, None),
    ):
        if not paths:
            continue
        ids_path = os.path.join(args.out, f"{name}.bin")
        total_bytes = 0
        parts = []
        for i, p in enumerate(paths):
            part = ids_path + f".part{i}"
            parts.append(part)
            total_bytes += encode_file(p, part, bpe, word_to_id, maxb)
        if len(parts) > 1:
            with open(ids_path, "wb") as out:
                for part in parts:
                    with open(part, "rb") as f:
                        while True:
                            chunk = f.read(1 << 20)
                            if not chunk:
                                break
                            out.write(chunk)
                    os.remove(part)
        else:
            os.replace(parts[0], ids_path)
        ntok = os.path.getsize(ids_path) // 2
        meta[name] = {"path": ids_path, "tokens": ntok, "bytes": total_bytes}
        print(f"{name}: {ntok:,} tokens, {total_bytes:,} bytes, {total_bytes/max(1,ntok):.3f} bytes/token")

    with open(os.path.join(args.out, "meta.json"), "w") as f:
        json.dump(meta, f)
    print(f"vocab: {len(vocab):,} tokens ({len(bpe.merges):,} BPE merges) -> {args.out}/meta.json")


if __name__ == "__main__":
    main()
