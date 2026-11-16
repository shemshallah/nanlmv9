#!/usr/bin/env python3
"""Held-out word+char evaluation (no CM gate unless a matching prior is supplied)."""
import argparse, json, math, os, sys
import torch, torch.nn.functional as F
HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,os.path.join(HERE,'..','..','v9'))
from bpe import BPE
from cm_residual_smoke import load_window_bytes, build_word_map, encode_text
from experimental import WordCharMamba

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--ckpt',required=True); ap.add_argument('--data-dir',default=os.path.expanduser('~/nanlm-data/tok32k_mixed')); ap.add_argument('--window',required=True); ap.add_argument('--tokens',type=int,default=64); ap.add_argument('--chunks',type=int,default=200); ap.add_argument('--device',default='auto'); args=ap.parse_args()
 device=torch.device('cuda' if args.device=='auto' and torch.cuda.is_available() else ('cpu' if args.device=='auto' else args.device))
 ck=torch.load(args.ckpt,map_location=device); c=ck['config']
 model=WordCharMamba(ck['meta']['vocab_size'],d_model=c['d_model'],char_model=c['char_model'],layers=1,n_experts=c['experts'],scan_mode=c.get('scan','sequential'),scan_chunk=c.get('scan_chunk',16)).to(device)
 model.load_state_dict(ck['model']); model.eval()
 meta=json.load(open(os.path.join(args.data_dir,'meta.json'))); vocab=[bytes.fromhex(h) for h in meta['vocab_hex']]
 bpe=BPE.load(os.path.join(HERE,'..','..','v9','bpe_merges.txt')); _,w2i=build_word_map(meta,bpe)
 raw=load_window_bytes(args.window); ids=encode_text(raw.decode('utf-8','surrogateescape'),bpe,w2i); recon=b''.join(vocab[i] for i in ids)
 lens=[len(vocab[i]) for i in ids]; offs=[]; p=0
 for l in lens: offs.append(p); p+=l
 seq=min(args.tokens,len(ids)-1); chunks=min(args.chunks,max(1,(len(ids)-seq-1)//seq))
 wsum=wn=0.0; csum=cn=0.0
 with torch.no_grad():
  for k in range(chunks):
   st=k*seq; tids=ids[st:st+seq+1]; ls=lens[st:st+seq]; bst=offs[st]; nb=sum(ls); bids=list(recon[bst:bst+nb])
   tx=torch.tensor(tids[:-1],dtype=torch.long,device=device)[None]; ty=torch.tensor(tids[1:],dtype=torch.long,device=device)[None]
   bx=torch.tensor(bids[:-1],dtype=torch.long,device=device)[None]; by=torch.tensor(bids[1:],dtype=torch.long,device=device)[None]
   tl=torch.tensor(ls[:-1],dtype=torch.long,device=device)
   extra=tl.clone()
   if int(extra.sum())>bx.shape[1]: extra[-1]-=int(extra.sum())-bx.shape[1]
   wl,cl,_=model(tx,bx,extra)
   wl,cl=wl.float().reshape(-1,ck['meta']['vocab_size']),cl.float().reshape(-1,256)
   wl_loss=F.cross_entropy(wl,ty.reshape(-1),reduction='sum'); cl_loss=F.cross_entropy(cl,by.reshape(-1),reduction='sum')
   wsum+=float(wl_loss); wn+=ty.numel(); csum+=float(cl_loss); cn+=by.numel()
 wce=wsum/max(1,wn); cbpc=csum/math.log(2)/max(1,cn)
 print(json.dumps({'word_ce':wce,'word_bpb':wce/math.log(2)/(len(recon)/len(ids)),'char_bpc':cbpc,'tokens':wn,'bytes':cn,'chunks':chunks},indent=1))
if __name__=='__main__': main()
