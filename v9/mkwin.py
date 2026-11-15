import sys, struct
src, out, nwin = sys.argv[1], sys.argv[2], int(sys.argv[3]); win=420; skip=20
d = open(src,'rb').read(); span=(len(d)-win)//nwin
with open(out,'wb') as f:
    f.write(struct.pack('<II', nwin, skip))
    for k in range(nwin):
        w=d[k*span:k*span+win]; f.write(struct.pack('<I',len(w))); f.write(w)
