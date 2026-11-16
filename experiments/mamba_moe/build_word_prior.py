#!/usr/bin/env python3
"""Build a word/concept unigram prior from WordNet glosses + wordlists."""
import argparse, json, math, os, sys, re
import numpy as np
HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,os.path.join(HERE,'..','..','v9'))
from bpe import BPE
WORD=re.compile(r"[A-Za-z][A-Za-z'-]*")
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--data-dir',required=True); ap.add_argument('--out',required=True); ap.add_argument('texts',nargs='+'); args=ap.parse_args()
 meta=json.load(open(os.path.join(args.data_dir,'meta.json'))); vocab=[bytes.fromhex(h) for h in meta['vocab_hex']]
 w2i={b:i for i,b in enumerate(vocab) if len(b)>1 and i>=256+meta['bpe_merges']}
 cnt=np.zeros(len(vocab),dtype=np.float64)
 for p in args.texts:
  raw=open(p,'rb').read().decode('utf-8','ignore')
  for w in WORD.findall(raw):
   i=w2i.get(w.encode())
   if i is not None: cnt[i]+=1
 cnt+=0.1; cnt/=cnt.sum(); logp=np.log(cnt).astype(np.float32)
 np.save(args.out,logp); print('saved',args.out,'vocab',len(vocab),'nonzero',int((cnt>0.1/len(vocab)).sum()))
if __name__=='__main__': main()
