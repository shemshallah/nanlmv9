#!/usr/bin/env python3
"""Exact byte-fallback word/punctuation tokenizer for the hybrid experiments.

Vocabulary layout:
  ids 0..255      raw byte values (always available as fallback)
  ids 256..N-1    the most frequent word/punctuation/whitespace chunks

The tokenizer is a bijection on bytes: unknown chunks are emitted as their
individual byte ids.  That makes bits-per-byte exact and avoids <unk>.
"""
import argparse
import collections
import json
import os
import re
import sys
from array import array

TOKEN_RE = re.compile(r"\w+|[^\w\s]|\s+", re.UNICODE)


def iter_tokens(path, max_bytes=None):
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


def count_tokens(paths, sample_bytes_per_file=None):
    counter = collections.Counter()
    for path in paths:
        n = 0
        for text in iter_tokens(path, sample_bytes_per_file):
            counter.update(TOKEN_RE.findall(text))
            n += len(text)
    return counter


def build_vocab(paths, vocab_size=32000, sample_bytes_per_file=None):
    counter = count_tokens(paths, sample_bytes_per_file)
    vocab = [bytes([i]) for i in range(256)]
    seen = set(vocab)
    for tok, _ in counter.most_common():
        if len(vocab) >= vocab_size:
            break
        b = tok.encode("utf-8", "surrogateescape")
        if b in seen:
            continue
        seen.add(b)
        vocab.append(b)
    return vocab


def encode(text, token_to_id):
    ids = array("H")
    for tok in TOKEN_RE.findall(text):
        i = token_to_id.get(tok.encode("utf-8", "surrogateescape"))
        if i is not None:
            ids.append(i)
        else:
            ids.extend(tok.encode("utf-8", "surrogateescape"))
    return ids


def encode_file(src, dst, token_to_id, max_bytes=None):
    out = array("H")
    total = 0
    with open(dst, "wb") as f:
        for text in iter_tokens(src, max_bytes):
            out.extend(encode(text, token_to_id))
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
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    vocab = build_vocab(args.vocab_texts, args.vocab_size, args.vocab_sample_bytes)
    token_to_id = {b: i for i, b in enumerate(vocab)}
    meta = {
        "vocab_size": len(vocab),
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
            total_bytes += encode_file(p, part, token_to_id, maxb)
        if len(parts) > 1:
            # concatenate parts without loading them all in memory
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
    print(f"vocab: {len(vocab):,} tokens -> {args.out}/meta.json")


if __name__ == "__main__":
    main()
