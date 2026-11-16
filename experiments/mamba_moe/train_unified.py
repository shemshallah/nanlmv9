#!/usr/bin/env python3
"""Canonical unified trainer scaffold (not auto-run).

Model: bytes -> byte SSM + boundary head -> word SSM/MoE -> char SSM.
Likelihood: word-token CE through the 32k tokenizer is the normalized bpb.
Auxiliary: boundary BCE + gated char/CM/retrieval byte loss.
"""
import argparse, json, math, os, sys, time
import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "v9"))
from bpe import BPE
from byte_layer import ByteTrie, CMBytePrior
from cm_residual_smoke import load_window_bytes, build_word_map, encode_text
from unified import UnifiedByteLM, boundary_loss, normalized_word_bpb, fusion_features, boundary_metrics
from retrieval import MultiMatchIndex


def chunk(ids, lens, recon, offs, start, ntok):
    tids = ids[start:start + ntok]
    ls = lens[start:start + ntok]
    bstart = offs[start]
    bids = list(recon[bstart:bstart + sum(ls)])
    return tids, ls, bids, bstart


def byte_contexts(bids, order):
    n = max(0, len(bids) - 1)
    ctx = torch.zeros(n, order, dtype=torch.long)
    for p in range(1, len(bids)):
        prev = bids[max(0, p - order):p]
        ctx[p - 1, order - len(prev):] = torch.tensor(prev, dtype=torch.long)
    return ctx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--window", required=True)
    ap.add_argument("--out", default=os.path.expanduser("~/nanlm-data/runs/unified"))
    ap.add_argument("--tokens", type=int, default=64)
    ap.add_argument("--d-model", type=int, default=256)
    ap.add_argument("--char-model", type=int, default=64)
    ap.add_argument("--experts", type=int, default=64)
    ap.add_argument("--scan", choices=("sequential", "chunked"), default="chunked")
    ap.add_argument("--scan-chunk", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--steps", type=int, default=100000)
    ap.add_argument("--max-seconds", type=float, default=None)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--cm-dump", default=None)
    ap.add_argument("--eval-window", default=None)
    ap.add_argument("--eval-cm-dump", default=None)
    ap.add_argument("--eval-chunks", type=int, default=20)
    ap.add_argument("--retrieval", choices=("none", "multimatch"), default="none")
    ap.add_argument("--retrieval-order", type=int, default=4)
    ap.add_argument("--save-interval", type=int, default=500)
    ap.add_argument("--result-json", default=None)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device))
    meta = json.load(open(os.path.join(args.data_dir, "meta.json")))
    vocab = [bytes.fromhex(h) for h in meta["vocab_hex"]]
    bpe = BPE.load(os.path.join(HERE, "..", "..", "v9", "bpe_merges.txt"))
    _, w2i = build_word_map(meta, bpe)
    raw = load_window_bytes(args.window)
    ids = encode_text(raw.decode("utf-8", "surrogateescape"), bpe, w2i)
    recon = b"".join(vocab[i] for i in ids)
    lens = [len(vocab[i]) for i in ids]; offs = []; p = 0
    for l in lens: offs.append(p); p += l
    print(f"tokens={len(ids):,} bytes={len(recon):,}")
    cm = CMBytePrior.load(args.cm_dump) if args.cm_dump else None
    retriever = None
    if args.retrieval == "multimatch":
        retriever = MultiMatchIndex(order=args.retrieval_order, vocab_size=256)
        retriever.build(np.frombuffer(recon, dtype=np.uint8), max_tokens=min(len(recon), 2_000_000))
    model = UnifiedByteLM(meta["vocab_size"], d_model=args.d_model, char_model=args.char_model,
                          experts=args.experts, scan_mode=args.scan, scan_chunk=args.scan_chunk).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    print("params", f"{sum(p.numel() for p in model.parameters()):,}")

    def run_eval():
        if not args.eval_window:
            return {}
        eraw = load_window_bytes(args.eval_window)
        eids = encode_text(eraw.decode("utf-8", "surrogateescape"), bpe, w2i)
        erecon = b"".join(vocab[i] for i in eids)
        elens = [len(vocab[i]) for i in eids]; eoffs=[]; pp=0
        for l in elens: eoffs.append(pp); pp+=l
        ep = CMBytePrior.load(args.eval_cm_dump) if args.eval_cm_dump else None
        n = min(args.tokens, len(eids)-1); chunks = min(args.eval_chunks, max(1,(len(eids)-n-1)//n))
        wsum=wcnt=0.0; csum=ccnt=0.0; wcm=0.0
        with torch.no_grad():
            for k in range(chunks):
                tids,ls,bids,bs=chunk(eids,elens,erecon,eoffs,k*n,n)
                bx=torch.tensor(bids[:-1],dtype=torch.long,device=device)[None]
                by=torch.tensor(bids[1:],dtype=torch.long,device=device)[None]
                tx=torch.tensor(tids,dtype=torch.long,device=device)[None]
                tl=torch.tensor(ls,dtype=torch.long,device=device)[None]
                out=model(bx,tx,tl)
                wl=out["word_logits"][:,:-1].float().reshape(-1,meta["vocab_size"])
                ty=torch.tensor(tids[1:],dtype=torch.long,device=device)
                bpt=torch.tensor(ls[1:],dtype=torch.float32,device=device).sum()
                wsum+=float(F.cross_entropy(wl,ty,reduction="sum")); wcnt+=bpt.item()
                cl=out["char_logits"].float().reshape(-1,256)
                if ep is not None:
                    rows=[ep.logits(bs+1+p) for p in range(len(bids)-1)]
                    cm_l=torch.stack(rows).to(device)
                    extra=torch.tensor(ep.probs[bs+1:bs+len(bids),0],dtype=torch.float32,device=device)[:,None]
                    ret_l=retriever.logits(byte_contexts(bids,args.retrieval_order)).to(device) if retriever is not None else None
                    mixed,w,_=model.fuse_bytes(out["char_logits"], cm_l, ret_l, extra=extra, training=False, cap=0.9)
                    cl=mixed.reshape(-1,256); wcm+=float(w[:,1].mean())
                csum+=float(F.cross_entropy(cl,by.reshape(-1),reduction="sum")); ccnt+=by.numel()
        bm = boundary_metrics(out["boundary_logits"], tl)
        return {"word_bpb":wsum/math.log(2)/max(1,wcnt),"char_bpc":csum/math.log(2)/max(1,ccnt),"cm_weight":wcm/max(1,chunks),"bytes":int(ccnt),**bm}

    t0=time.time(); n=args.tokens
    for step in range(args.steps):
        if args.max_seconds and time.time()-t0>=args.max_seconds: break
        st=int(torch.randint(0,max(1,len(ids)-n-1),(1,)))
        tids,ls,bids,bs=chunk(ids,lens,recon,offs,st,n)
        bx=torch.tensor(bids[:-1],dtype=torch.long,device=device)[None]
        by=torch.tensor(bids[1:],dtype=torch.long,device=device)[None]
        tx=torch.tensor(tids,dtype=torch.long,device=device)[None]
        tl=torch.tensor(ls,dtype=torch.long,device=device)[None]
        with torch.autocast(device_type="cuda",dtype=torch.bfloat16,enabled=(args.amp and device.type=="cuda")):
            out=model(bx,tx,tl)
        wl=out["word_logits"][:,:-1].float().reshape(-1,meta["vocab_size"])
        ty=torch.tensor(tids[1:],dtype=torch.long,device=device)
        bpt=torch.tensor(ls[1:],dtype=torch.float32,device=device).sum()
        word_ce=F.cross_entropy(wl,ty)
        bloss=boundary_loss(out["boundary_logits"],tl)
        cl=out["char_logits"].float().reshape(-1,256)
        cm_l = None; ret_l = None; extra = None
        if cm is not None and bs+len(bids)<=cm.n:
            rows=[cm.logits(bs+1+p) for p in range(len(bids)-1)]
            cm_l=torch.stack(rows).to(device)
            extra=torch.tensor(cm.probs[bs+1:bs+len(bids),0],dtype=torch.float32,device=device)[:,None]
        if retriever is not None:
            ret_l=retriever.logits(byte_contexts(bids,args.retrieval_order)).to(device)
        if cm_l is not None or ret_l is not None:
            mixed,w,_=model.fuse_bytes(out["char_logits"],cm_l,ret_l,extra=extra,training=True,cap=0.9)
            cl=mixed.reshape(-1,256)
        char_loss=F.cross_entropy(cl,by.reshape(-1))
        loss=word_ce+0.1*bloss+0.1*char_loss
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
        if step%10==0: print(f"step {step:5d} word_ce={float(word_ce):.4f} char_bpc={float(char_loss)/math.log(2):.4f} bnd={float(bloss):.4f}")
        if args.save_interval and (step+1)%args.save_interval==0:
            torch.save({"model":model.state_dict(),"step":step+1,"config":vars(args),"meta":{"vocab_size":meta["vocab_size"]}},os.path.join(args.out,"last.pt"))
    ev=run_eval(); print("heldout",json.dumps(ev))
    if args.result_json: json.dump({"train_last":None,"heldout":ev,"config":vars(args)},open(args.result_json,"w"),indent=1)
    os.makedirs(args.out,exist_ok=True)
    torch.save({"model":model.state_dict(),"config":vars(args),"meta":{"vocab_size":meta["vocab_size"]}},os.path.join(args.out,"last.pt"))

if __name__=="__main__": main()
