#!/usr/bin/env python3
"""Correct BPE for NaN v9.
Pretokenization (causal-safe): a *word* is a maximal run of [A-Za-z0-9] or bytes>=0x80; every other byte is its own
token (id = byte value). Merges apply only inside words. Ids: 0..255 bytes, 256+r = merge rank r.
encode() APPLIES merges (lowest rank first), decode() inverts, tables are saved (bpe_merges.txt) and are the SAME file
the C core loads, so Python and C segmentations can be diffed."""
import re, sys, collections, json
WORD = re.compile(rb"[A-Za-z0-9\x80-\xff]+")
PRE = re.compile(rb"[A-Za-z0-9\x80-\xff]+|[^A-Za-z0-9\x80-\xff]")

def train(paths, n_merges=2000, max_bytes=12_000_000, min_freq=2):
    wc = collections.Counter()
    for p in paths:
        data = open(p, 'rb').read()[:max_bytes]
        wc.update(WORD.findall(data))
    words = [(tuple(w), f) for w, f in wc.items() if f >= min_freq]
    seqs = [list(w) for w, _ in words]; freqs = [f for _, f in words]
    pairs = collections.Counter(); where = collections.defaultdict(set)
    for i, s in enumerate(seqs):
        for a, b in zip(s, s[1:]):
            pairs[(a, b)] += freqs[i]; where[(a, b)].add(i)
    merges = []
    for r in range(n_merges):
        if not pairs: break
        best = max(pairs.items(), key=lambda kv: (kv[1], -kv[0][0], -kv[0][1]))[0]
        if pairs[best] < 2: break
        nid = 256 + r; merges.append(best)
        for i in list(where[best]):
            s = seqs[i]; f = freqs[i]
            # remove old pair counts for this word
            for a, b in zip(s, s[1:]):
                pairs[(a, b)] -= f
                if pairs[(a, b)] <= 0: pairs.pop((a, b), None)
            out = []; j = 0
            while j < len(s):
                if j < len(s) - 1 and (s[j], s[j + 1]) == best: out.append(nid); j += 2
                else: out.append(s[j]); j += 1
            seqs[i] = out
            for a, b in zip(out, out[1:]):
                pairs[(a, b)] += f; where[(a, b)].add(i)
        where.pop(best, None); pairs.pop(best, None)
    return merges

class BPE:
    def __init__(self, merges):
        self.merges = [tuple(m) for m in merges]
        self.rank = {m: r for r, m in enumerate(self.merges)}
        self.cache = {}
        self.vocab = {i: bytes([i]) for i in range(256)}
        for r, (a, b) in enumerate(self.merges): self.vocab[256 + r] = self.vocab[a] + self.vocab[b]
    def word(self, w: bytes):
        c = self.cache.get(w)
        if c is not None: return c
        s = list(w)
        while len(s) > 1:
            best = None; br = 1 << 30
            for a, b in zip(s, s[1:]):
                r = self.rank.get((a, b), 1 << 30)
                if r < br: br, best = r, (a, b)
            if best is None: break
            out = []; j = 0
            while j < len(s):
                if j < len(s) - 1 and (s[j], s[j + 1]) == best: out.append(256 + br); j += 2
                else: out.append(s[j]); j += 1
            s = out
        self.cache[w] = s
        return s
    def encode(self, data: bytes):
        out = []
        for t in PRE.findall(data):
            if WORD.fullmatch(t): out.extend(self.word(t))
            else: out.append(t[0])
        return out
    def decode(self, ids):
        return b"".join(self.vocab[i] for i in ids)
    def save(self, path):
        open(path, 'w').write("\n".join(f"{a} {b}" for a, b in self.merges) + "\n")
    @staticmethod
    def load(path):
        return BPE([tuple(map(int, l.split())) for l in open(path) if l.strip()])

if __name__ == "__main__":
    C = "../corpora"
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
    m = train([f"{C}/wt2_train.txt"], n)          # trained on TRAIN only (never test)
    b = BPE(m); b.save("bpe_merges.txt")
    test = open(f"{C}/wt2_test.txt", 'rb').read()
    ids = b.encode(test)
    assert b.decode(ids) == test, "ROUNDTRIP FAILED"
    print(f"merges={len(m)} roundtrip=OK test_bytes={len(test)} tokens={len(ids)} bytes/token={len(test)/len(ids):.3f}")
    print("first merges:", [b.vocab[256 + i] for i in range(20)])
    print("last merges:", [b.vocab[256 + i] for i in range(len(m) - 10, len(m))])
