#!/usr/bin/env python3
"""Top-1/top-5 metrics for the unified model: byte, token/subword, word, sentence."""
import argparse, json, math, os, re, sys
import numpy as np
import torch, torch.nn.functional as F
HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,os.path.join(HERE,'..','..','v9'))
from bpe import BPE
from cm_residual_smoke import load_window_bytes, build_word_map, encode_text
from unified import UnifiedByteLM, boundary_metrics
from byte_layer import CMBytePrior
from retrieval import MultiMatchIndex
WORD=re.compile(rb"[A-Za-z0-9<>']+")


def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--ckpt',required=True); ap.add_argument('--data-dir',default=os.path.expanduser('~/nanlm-data/tok32k_wnmix')); ap.add_argument('--window',required=True); ap.add_argument('--tokens',type=int,default=64); ap.add_argument('--chunks',type=int,default=20); ap.add_argument('--device',default='auto'); ap.add_argument('--cm-dump',default=None); ap.add_argument('--retrieval',choices=('none','multimatch'),default='none'); ap.add_argument('--retrieval-order',type=int,default=4); args=ap.parse_args()
 device=torch.device('cuda' if args.device=='auto' and torch.cuda.is_available() else ('cpu' if args.device=='auto' else args.device))
 ck=torch.load(args.ckpt,map_location=device); c=ck['config']
 model=UnifiedByteLM(ck['meta']['vocab_size'],d_model=c['d_model'],char_model=c['char_model'],experts=c['experts'],scan_mode=c.get('scan','chunked'),scan_chunk=c.get('scan_chunk',4)).to(device)
 model.load_state_dict(ck['model']); model.eval()
 meta=json.load(open(os.path.join(args.data_dir,'meta.json'))); vocab=[bytes.fromhex(h) for h in meta['vocab_hex']]
 bpe=BPE.load(os.path.join(HERE,'..','..','v9','bpe_merges.txt')); _,w2i=build_word_map(meta,bpe)
 raw=load_window_bytes(args.window); ids=encode_text(raw.decode('utf-8','surrogateescape'),bpe,w2i); recon=b''.join(vocab[i] for i in ids)
 lens=[len(vocab[i]) for i in ids]; offs=[]; p=0
 for l in lens: offs.append(p); p+=l
 n=min(args.tokens,len(ids)-1); chunks=min(args.chunks,max(1,(len(ids)-n-1)//n))
 cm=CMBytePrior.load(args.cm_dump) if args.cm_dump else None
 retriever=None
 if args.retrieval=='multimatch':
  retriever=MultiMatchIndex(order=args.retrieval_order,vocab_size=256); retriever.build(np.frombuffer(recon,dtype=np.uint8),max_tokens=min(len(recon),2000000))
 maxbyte=offs[chunks*n+n]+sum(lens[chunks*n:chunks*n+n])
 bt1=np.zeros(max(1,maxbyte),bool); bt5=np.zeros(max(1,maxbyte),bool); have=np.zeros(max(1,maxbyte),bool)
 tok1=tok5=tn=0; wsum=0.0; wbytes=0; tp=fp=fn=0; wchar=wcm=wret=0.0
 with torch.no_grad():
  for k in range(chunks):
   st=k*n; tids=ids[st:st+n+1]; ls=lens[st:st+n]; bs=offs[st]; nb=sum(ls); bids=list(recon[bs:bs+nb])
   bx=torch.tensor(bids[:-1],dtype=torch.long,device=device)[None]; tx=torch.tensor(tids[:-1],dtype=torch.long,device=device)[None]; tl=torch.tensor(ls,dtype=torch.long,device=device)[None]
   out=model(bx,tx,tl)
   wl=out['word_logits'][0].float(); ty=torch.tensor(tids[1:],dtype=torch.long,device=device)
   t5=torch.topk(wl,5,dim=-1).indices; tok1+=int((t5[:,0]==ty).sum()); tok5+=int((t5==ty[:,None]).any(-1).sum()); tn+=ty.numel()
   wsum+=float(F.cross_entropy(wl,ty,reduction='sum')); wbytes+=int(sum(lens[st+1:st+n+1]))
   cl=out['char_logits'][0].float(); cy=torch.tensor(bids[1:],dtype=torch.long,device=device)
   cm_l=ret_l=None; extra=None
   if cm is not None:
    rows=[cm.logits(bs+1+p) for p in range(len(bids)-1)]; cm_l=torch.stack(rows).to(device)
    extra=torch.tensor(cm.probs[bs+1:bs+len(bids),0],dtype=torch.float32,device=device)[:,None]
   if retriever is not None:
    ctx=torch.zeros(len(bids)-1,args.retrieval_order,dtype=torch.long)
    for p in range(1,len(bids)):
     prev=bids[max(0,p-args.retrieval_order):p]; ctx[p-1,args.retrieval_order-len(prev):]=torch.tensor(prev)
    ret_l=retriever.logits(ctx).to(device)
   if cm_l is not None or ret_l is not None:
    cl,w,_=model.fuse_bytes(cl,cm_l,ret_l,extra=extra,training=False,cap=0.9)
    cl=cl.reshape(-1,256); wchar+=float(w[:,0].mean()); wcm+=float(w[:,1].mean());
    if w.shape[1]>2: wret+=float(w[:,2].mean())
   c5=torch.topk(cl,5,dim=-1).indices; a1=(c5[:,0]==cy).cpu().numpy(); a5=(c5==cy[:,None]).any(-1).cpu().numpy()
   pos=bs+1; m=min(len(a1),maxbyte-pos); bt1[pos:pos+m]=a1[:m]; bt5[pos:pos+m]=a5[:m]; have[pos:pos+m]=True
   bm=boundary_metrics(out['boundary_logits'],tl); # aggregate via counts approximated from f1? recompute mask
   from unified import predict_boundaries
   mask=predict_boundaries(out['boundary_logits']).cpu(); gold=torch.zeros_like(mask); ends=(torch.cumsum(tl.cpu(),1)-1).clamp(0,mask.shape[1]-1); gold.scatter_(1,ends,True)
   tp+=int((mask&gold).sum()); fp+=int((mask&~gold).sum()); fn+=int((~mask&gold).sum())
 prec=tp/max(1,tp+fp); rec=tp/max(1,tp+fn); f1=2*prec*rec/max(1e-9,prec+rec)
 # word and sentence groups over the scored target bytes
 words=[]; sents=[]; s=1
 for m in WORD.finditer(recon[:maxbyte]):
  a,b=m.span()
  if a>=1 and b<=maxbyte and have[a:b].all(): words.append((bt1[a:b].all(),bt5[a:b].all(),bt1[a:b].mean(),bt5[a:b].mean()))
 for m in re.finditer(rb'[.!?] ', recon[:maxbyte]):
  e=m.end()
  if e-s>=8 and s>=1 and have[s:e].all(): sents.append((bt1[s:e].all(),bt5[s:e].all(),bt1[s:e].mean(),bt5[s:e].mean()))
  s=e
 W=np.array(words,float) if words else np.zeros((0,4)); S=np.array(sents,float) if sents else np.zeros((0,4)); scored=have[1:maxbyte]
 res={'word_bpb':wsum/math.log(2)/max(1,wbytes),'byte_top1':float(bt1[1:maxbyte].mean()),'byte_top5':float(bt5[1:maxbyte].mean()),'token_top1':tok1/max(1,tn),'token_top5':tok5/max(1,tn),'n_tokens':tn,'n_words':len(W),'word_top1':float(W[:,0].mean()) if len(W) else 0,'word_top5':float(W[:,1].mean()) if len(W) else 0,'word_mean_byte_acc_top1':float(W[:,2].mean()) if len(W) else 0,'word_mean_byte_acc_top5':float(W[:,3].mean()) if len(W) else 0,'char_weight':wchar/max(1,chunks),'cm_weight':wcm/max(1,chunks),'retrieval_weight':wret/max(1,chunks),'n_sents':len(S),'sent_top1_strict':float(S[:,0].mean()) if len(S) else 0,'sent_top5_strict':float(S[:,1].mean()) if len(S) else 0,'sent_mean_byte_acc_top1':float(S[:,2].mean()) if len(S) else 0,'sent_mean_byte_acc_top5':float(S[:,3].mean()) if len(S) else 0,'boundary_precision':prec,'boundary_recall':rec,'boundary_f1':f1,'scored_bytes':int(scored.sum())}
 print(json.dumps(res,indent=1))
if __name__=='__main__': main()
