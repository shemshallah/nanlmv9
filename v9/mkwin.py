import sys, struct
src, out, nwin = sys.argv[1], sys.argv[2], int(sys.argv[3]); win=int(sys.argv[4]) if len(sys.argv)>4 else 420; skip=int(sys.argv[5]) if len(sys.argv)>5 else 20
d = open(src,'rb').read(); span=(len(d)-win)//nwin
with open(out,'wb') as f:
    f.write(struct.pack('<II', nwin, skip))
    for k in range(nwin):
        w=d[k*span:k*span+win]; f.write(struct.pack('<I',len(w))); f.write(w)
