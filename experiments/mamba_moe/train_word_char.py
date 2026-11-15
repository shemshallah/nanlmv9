#!/usr/bin/env python3
"""Train the custom-Mamba word branch + conditional character branch.

Word tokens are the 32k tokenizer; the character branch consumes raw bytes and
is conditioned on the word branch via a learned word-to-char bridge.
"""
import argparse, json, math, os, sys
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "v9"))
from bpe import BPE
from cm_residual_smoke import load_window_bytes, build_word_map, encode_text
from experimental import WordCharMamba


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=os.path.expanduser("~/nanlm-data/tok32k_mixed"))
    ap.add_argument("--window", default=os.path.join(HERE, "..", "..", "v9", "win_valid8_4k.bin"))
    ap.add_argument("--out", default=os.path.expanduser("~/nanlm-data/runs/word_char"))
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--tokens", type=int, default=12)
    ap.add_argument("--d-model", type=int, default=64)
    ap.add_argument("--char-model", type=int, default=32)
    ap.add_argument("--experts", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--max-seconds", type=float, default=None)
    ap.add_argument("--result-json", default=None)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    meta = json.load(open(os.path.join(args.data_dir, "meta.json")))
    vocab = [bytes.fromhex(h) for h in meta["vocab_hex"]]
    bpe = BPE.load(os.path.join(HERE, "..", "..", "v9", "bpe_merges.txt"))
    _, word_to_id = build_word_map(meta, bpe)
    raw = load_window_bytes(args.window)
    ids = encode_text(raw.decode("utf-8", "surrogateescape"), bpe, word_to_id)
    recon = b"".join(vocab[i] for i in ids)
    byte_ids = list(recon)
    lengths_all = [len(vocab[i]) for i in ids]
    print(f"tokens={len(ids):,} bytes={len(recon):,}")

    model = WordCharMamba(meta["vocab_size"], d_model=args.d_model,
                          char_model=args.char_model, layers=1, n_experts=args.experts)
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    print("params", f"{model.params():,}")

    t0 = __import__("time").time()
    last = (0.0, 0.0)
    for step in range(args.steps):
        if args.max_seconds and (__import__("time").time() - t0) >= args.max_seconds:
            print(f"time budget reached at step {step}")
            break
        # random token start; take a fixed token count
        start = int(torch.randint(0, max(1, len(ids) - args.tokens - 1), (1,)))
        tids = ids[start:start + args.tokens]
        lens = lengths_all[start:start + args.tokens]
        byte_start = sum(lengths_all[:start])
        nbytes = sum(lens)
        bids = byte_ids[byte_start:byte_start + nbytes]
        if len(bids) < 2:
            continue
        tok_x = torch.tensor(tids[:-1], dtype=torch.long, device=device)[None]
        tok_y = torch.tensor(tids[1:], dtype=torch.long, device=device)[None]
        tok_lens = torch.tensor(lens[:-1], dtype=torch.long, device=device)
        byte_x = torch.tensor(bids[:-1], dtype=torch.long, device=device)[None]
        byte_y = torch.tensor(bids[1:], dtype=torch.long, device=device)[None]
        # The char branch sees one extra byte at the end of the token stream;
        # truncate the bridge context to the byte inputs.
        char_extra_lens = tok_lens.clone()
        if int(char_extra_lens.sum()) > byte_x.shape[1]:
            char_extra_lens[-1] -= int(char_extra_lens.sum()) - byte_x.shape[1]
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(args.amp and device.type == "cuda")):
            wl, cl, aux = model(tok_x, byte_x, char_extra_lens)
        wl, cl = wl.float(), cl.float()
        word_loss = F.cross_entropy(wl.reshape(-1, meta["vocab_size"]), tok_y.reshape(-1))
        char_loss = F.cross_entropy(cl.reshape(-1, 256), byte_y.reshape(-1))
        loss = word_loss + char_loss + 0.01 * (aux if torch.is_tensor(aux) else 0.0)
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        last = (float(word_loss), float(char_loss) / math.log(2))
        if step % 2 == 0:
            print(f"step {step:3d} word_ce={last[0]:.4f} char_bpc={last[1]:.4f}")
    if args.result_json:
        json.dump({"word_ce": last[0], "char_bpc": last[1], "params": model.params(),
                   "config": vars(args)}, open(args.result_json, "w"), indent=1)
    torch.save({"model": model.state_dict(), "config": vars(args),
                "meta": {"vocab_size": meta["vocab_size"]}},
               os.path.join(args.out, "last.pt"))
    print("saved", os.path.join(args.out, "last.pt"))


if __name__ == "__main__":
    main()
