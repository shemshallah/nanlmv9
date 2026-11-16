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
from byte_layer import CMBytePrior
from hybrid import ContextGate


def eval_heldout(model, gate, cm_prior, word_prior, word_gate, ids, lens, offs, recon, vocab_size, tokens, chunks, device):
    model.eval()
    seq = min(tokens, len(ids) - 1)
    chunks = min(chunks, max(1, (len(ids) - seq - 1) // seq))
    wsum = wcnt = csum = ccnt = 0.0
    wcm = 0.0
    with torch.no_grad():
        for k in range(chunks):
            st = k * seq
            tids = ids[st:st+seq+1]; ls = lens[st:st+seq]; bst = offs[st]; nb = sum(ls)
            bids = list(recon[bst:bst+nb])
            tx=torch.tensor(tids[:-1],dtype=torch.long,device=device)[None]; ty=torch.tensor(tids[1:],dtype=torch.long,device=device)[None]
            bx=torch.tensor(bids[:-1],dtype=torch.long,device=device)[None]; by=torch.tensor(bids[1:],dtype=torch.long,device=device)[None]
            tl=torch.tensor(ls[:-1],dtype=torch.long,device=device); extra=tl.clone()
            if int(extra.sum())>bx.shape[1]: extra[-1]-=int(extra.sum())-bx.shape[1]
            wl,cl,_=model(tx,bx,extra,previous_only=True)
            wl,cl=wl.float().reshape(-1,vocab_size),cl.float().reshape(-1,256)
            if word_prior is not None:
                wp=word_prior[None].expand(wl.shape[0],-1); ex=torch.linspace(0,1,wl.shape[0],device=device)[:,None]
                wm,ww=word_gate([wl,wp],extra=ex); wsum+=float(F.cross_entropy(wm,ty.reshape(-1),reduction='sum')); wcm_word=float(ww[:,1].mean())
            else:
                wsum+=float(F.cross_entropy(wl,ty.reshape(-1),reduction='sum')); wcm_word=0.0
            wcnt+=ty.numel()
            if cm_prior is not None:
                rows=[cm_prior.logits(bst+p) for p in range(1,len(bids))]
                cm=torch.stack(rows).to(device); ex=torch.linspace(0,1,cm.shape[0],device=device)[:,None]
                mixed,cw=gate([cl,cm],extra=ex); wcm+=float(cw[:,1].sum()); csum+=float(F.cross_entropy(mixed,by.reshape(-1),reduction='sum'))
            else:
                csum+=float(F.cross_entropy(cl,by.reshape(-1),reduction='sum'))
            ccnt+=by.numel()
    model.train()
    return {"eval_chunks":chunks,"eval_tokens":int(wcnt),"eval_bytes":int(ccnt),
            "eval_word_ce":wsum/max(1,wcnt),"eval_word_bpb":(wsum/max(1,wcnt))/math.log(2)/(len(recon)/len(ids)),
            "eval_char_bpc":csum/math.log(2)/max(1,ccnt),"eval_mean_w_cm":wcm/max(1,ccnt) if cm_prior else 0.0}

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
    ap.add_argument("--cm-dump", default=None, help="real v9 CM byte prior for gated char branch")
    ap.add_argument("--scan", choices=("sequential", "chunked"), default="sequential")
    ap.add_argument("--scan-chunk", type=int, default=16)
    ap.add_argument("--save-interval", type=int, default=500)
    ap.add_argument("--resume", default=None)
    ap.add_argument("--eval-window", default=None)
    ap.add_argument("--eval-cm-dump", default=None)
    ap.add_argument("--eval-chunks", type=int, default=20)
    ap.add_argument("--word-prior", default=None)
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
    offsets = []
    pos = 0
    for ln in lengths_all:
        offsets.append(pos)
        pos += ln
    print(f"tokens={len(ids):,} bytes={len(recon):,}")

    model = WordCharMamba(meta["vocab_size"], d_model=args.d_model,
                          char_model=args.char_model, layers=1, n_experts=args.experts,
                          scan_mode=args.scan, scan_chunk=args.scan_chunk)
    model.to(device)
    cm_prior = CMBytePrior.load(args.cm_dump) if args.cm_dump else None
    gate = ContextGate(n_sources=2, hidden=64, extra_features=1).to(device) if cm_prior else None
    word_prior = torch.from_numpy(__import__("numpy").load(args.word_prior)).to(device) if args.word_prior else None
    word_gate = ContextGate(n_sources=2, hidden=64, extra_features=1).to(device) if word_prior is not None else None
    params = list(model.parameters()) + (list(gate.parameters()) if gate else []) + (list(word_gate.parameters()) if word_gate else [])
    opt = torch.optim.AdamW(params, lr=args.lr)
    start_step = 0
    if args.resume:
        ck = torch.load(args.resume, map_location=device)
        model.load_state_dict(ck["model"])
        if gate is not None and ck.get("gate") is not None:
            gate.load_state_dict(ck["gate"])
        if "opt" in ck:
            opt.load_state_dict(ck["opt"])
        start_step = ck.get("step", 0)
        print(f"resumed at step {start_step}")
    print("params", f"{model.params():,}")

    t0 = __import__("time").time()
    last = (0.0, 0.0)
    last_w_cm = 0.0
    last_w_word = 0.0
    for step in range(start_step, args.steps):
        if args.max_seconds and (__import__("time").time() - t0) >= args.max_seconds:
            print(f"time budget reached at step {step}")
            break
        # random token start; take a fixed token count
        start = int(torch.randint(0, max(1, len(ids) - args.tokens - 1), (1,)))
        tids = ids[start:start + args.tokens]
        lens = lengths_all[start:start + args.tokens]
        byte_start = offsets[start]
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
        wl_flat = wl.reshape(-1, meta["vocab_size"])
        if word_prior is not None:
            wp = word_prior[None].expand(wl_flat.shape[0], -1)
            ex = torch.linspace(0, 1, wl_flat.shape[0], device=device)[:, None]
            wmix, ww = word_gate([wl_flat, wp], extra=ex)
            word_loss = F.cross_entropy(wmix, tok_y.reshape(-1))
            last_w_word = float(ww[:, 1].mean())
        else:
            word_loss = F.cross_entropy(wl_flat, tok_y.reshape(-1))
            last_w_word = 0.0
        if cm_prior is not None:
            cm_rows = [cm_prior.logits(byte_start + p) for p in range(1, len(bids))]
            cm_logits = torch.stack(cm_rows).to(device)
            ex = torch.linspace(0, 1, cm_logits.shape[0], device=device)[:, None]
            char_mixed, cw = gate([cl.reshape(-1, 256), cm_logits], extra=ex)
            char_loss = F.cross_entropy(char_mixed, byte_y.reshape(-1))
            last_w_cm = float(cw[:, 1].mean())
        else:
            char_loss = F.cross_entropy(cl.reshape(-1, 256), byte_y.reshape(-1))
            last_w_cm = 0.0
        loss = word_loss + char_loss + 0.01 * (aux if torch.is_tensor(aux) else 0.0)
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step()
        last = (float(word_loss), float(char_loss) / math.log(2))
        if args.save_interval and (step + 1) % args.save_interval == 0:
            torch.save({"model": model.state_dict(), "gate": gate.state_dict() if gate else None,
                        "opt": opt.state_dict(), "step": step + 1, "config": vars(args),
                        "meta": {"vocab_size": meta["vocab_size"]}},
                       os.path.join(args.out, "last.pt"))
        if step % 2 == 0:
            print(f"step {step:3d} word_ce={last[0]:.4f} char_bpc={last[1]:.4f}")
    eval_metrics = {}
    if args.eval_window:
        eraw = load_window_bytes(args.eval_window)
        eids = encode_text(eraw.decode("utf-8", "surrogateescape"), bpe, word_to_id)
        erecon = b"".join(vocab[i] for i in eids)
        elens = [len(vocab[i]) for i in eids]; eoffs=[]; pp=0
        for l in elens: eoffs.append(pp); pp+=l
        eprior = CMBytePrior.load(args.eval_cm_dump) if args.eval_cm_dump else None
        eval_metrics = eval_heldout(model, gate, eprior, word_prior, word_gate, eids, elens, eoffs, erecon,
                                    meta["vocab_size"], args.tokens, args.eval_chunks, device)
        print("heldout", json.dumps(eval_metrics))
    if args.result_json:
        json.dump({"word_ce": last[0], "char_bpc": last[1], "params": model.params(),
                   "mean_w_cm": last_w_cm, "mean_w_word_prior": last_w_word,
                   "heldout": eval_metrics, "config": vars(args)},
                  open(args.result_json, "w"), indent=1)
    torch.save({"model": model.state_dict(), "gate": gate.state_dict() if gate else None,
                "opt": opt.state_dict(), "step": step + 1, "config": vars(args),
                "meta": {"vocab_size": meta["vocab_size"]}},
               os.path.join(args.out, "last.pt"))
    print("saved", os.path.join(args.out, "last.pt"))


if __name__ == "__main__":
    main()
