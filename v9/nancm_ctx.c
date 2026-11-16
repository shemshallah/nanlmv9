// nancm.c -- NaN v9 byte-level context-mixing core (bitwise, PAQ/lpaq family) with SUBWORD contexts.
// Predicts BYTES (so bpc is unambiguous). BPE enters as *causal* context features: the current word prefix is
// segmented with the same merge table as bpe.py (verified by --segdump), giving morphology-aware contexts.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
typedef uint8_t u8; typedef uint16_t u16; typedef uint32_t u32; typedef uint64_t u64;

#define NCTX 26
#define NMM 2
#define NIN (NCTX*2 + 2 + NMM*2 + 1)
#define NMIX 5
#define MAXH (48u<<20)

typedef struct { u16 p; u8 n; u8 chk; } Slot;
static int TBITS[NCTX] = {18,24,24,24,24,24,24, 22,24,24, 22,24, 24,24, 22,22,22,22,20,22,22,20, 20,22,20,22};
static const int LIMIT[NCTX] = {250,250,250,160,120,100,80, 250,120,80, 250,120, 200,160, 250,160,160,160,120,250,200,160, 160,250,120,200};
static Slot *T[NCTX]; static Slot o0t[256];
static int rt[256]; static float conf[256];
static float stab[4096];
static u8 *hist; static u32 histlen = 0, mm_floor = 0;

static inline float squashf(float x){ if(x>20)x=20; if(x<-20)x=-20; return 1.0f/(1.0f+expf(-x)); }
static inline float clampf(float x,float a,float b){ return x<a?a:(x>b?b:x); }
static inline u32 mix(u32 a,u32 b){ u32 x=a*0x9E3779B1u ^ (b+0x7F4A7C15u)*0x85EBCA6Bu; x^=x>>16; x*=0x7FEB352Du; x^=x>>15; x*=0x846CA68Bu; x^=x>>16; return x; }
static inline int isword(int c){ return (c>='a'&&c<='z')||(c>='A'&&c<='Z')||(c>='0'&&c<='9')||c>=0x80; }
static inline int fold(int c){ return (c>='A'&&c<='Z')?c+32:c; }

// ---------- BPE (same table as bpe.py) ----------
static u16 *pr = NULL; static int nmerge = 0; static int use_sub = 1;
static int ev_match_hist = 0, ev_carry = 0;
static int bpe_load(const char *path){
  pr = (u16*)calloc(4096u*4096u, 2); FILE *f=fopen(path,"r"); if(!f) return -1; int a,b,r=0;
  while(fscanf(f,"%d %d",&a,&b)==2){ if(a<4096&&b<4096) pr[a*4096+b]=(u16)(r+1); r++; } fclose(f); nmerge=r; return r; }
static int seg(const u8 *w,int L,int *t){ int n=L>48?48:L; for(int i=0;i<n;i++) t[i]=w[i];
  while(n>1){ int best=-1; int br=65535; for(int i=0;i<n-1;i++){ int r=pr[t[i]*4096+t[i+1]]; if(r&&r<br){br=r;best=i;} }
    if(best<0)break; t[best]=256+(br-1); for(int i=best+1;i<n-1;i++) t[i]=t[i+1]; n--; } return n; }

// ---------- byte state ----------
typedef struct { u32 c4,c8; u32 wh,pw1,pw2; u32 fwh; int wlen,linepos,wcount; u8 w[64]; int mptr[NMM], mlen[NMM]; int mmatched[NMM]; } BS;
static BS bs;
static u32 *mht[NMM]; static const int MINLEN[NMM]={5,12}; static const int MHB=22;
static float msm[NMM][64];
static u8 indirect[65536]; static u32 lastpos[256];
static u32 h[NCTX]; static int cls_c1, wl_sel;

static void mm_byte_update(BS *s,int c){
  for(int j=0;j<NMM;j++){
    if(s->mlen[j]>0 && (int)hist[s->mptr[j]]==c){ s->mlen[j]++; s->mptr[j]++; } else s->mlen[j]=0;
    u32 pos=histlen; // hist already has c appended
    if(pos-mm_floor>=(u32)MINLEN[j]){
      u32 hv=0; for(int k=1;k<=MINLEN[j];k++) hv=hv*0x2F0B4A27u+hist[pos-k]+1; hv=(hv^(hv>>15))*0x9E3779B1u>>(32-MHB);
      if(s->mlen[j]==0){ u32 cand=mht[j][hv]; if(cand>mm_floor && cand<pos){ int l=0; while(l<255 && cand-1-l>=mm_floor && pos-1-l>cand-1-l && hist[cand-1-l]==hist[pos-1-l]) l++;
          if(l>=MINLEN[j]){ s->mlen[j]=l; s->mptr[j]=cand; } } }
      mht[j][hv]=pos;
    }
  }
}
static void bs_push(BS *s,int c){
  int c1=s->c4&255,c2=(s->c4>>8)&255;
  hist[histlen++]=(u8)c;
  indirect[(c2<<8)|c1]=(u8)c;
  lastpos[c]=histlen;
  s->c8=(s->c8<<8)|(s->c4>>24); s->c4=(s->c4<<8)|c;
  if(isword(c)){ s->wh=mix(s->wh+1,c); s->fwh=mix(s->fwh+1,fold(c)); if(s->wlen<63) s->w[s->wlen]=(u8)c; s->wlen++; }
  else { if(s->wlen>0){ s->pw2=s->pw1; s->pw1=s->wh; } s->wh=0; s->fwh=0; s->wlen=0; }
  if(c==' '&&s->wlen>0) s->wcount++;
  if(c=='.'||c=='!'||c=='?'||c=='\n') s->wcount=0;
  s->linepos = (c=='\n') ? 0 : s->linepos+1;
  mm_byte_update(s,c);
}
static void compute_ctx(const BS *s){
  u32 c1=s->c4&255,c2=(s->c4>>8)&255,c3=(s->c4>>16)&255,c4=s->c4>>24;
  h[0]=c1+1;
  h[1]=mix(s->c4&0xFFFF,2);
  h[2]=mix(s->c4&0xFFFFFF,3);
  h[3]=mix(s->c4,4);
  h[4]=mix(s->c4,mix(s->c8&0xFF,5));
  h[5]=mix(s->c4,mix(s->c8&0xFFFF,6));
  h[6]=mix(s->c4,mix(s->c8,8));
  int inw=s->wlen>0;
  h[7]= inw? mix(s->wh,7) : mix(c1,0x77);
  h[8]= mix(inw? s->wh : mix(c1,9), s->pw1+8);
  h[9]= mix(mix(inw? s->wh : mix(c1,10), s->pw1), s->pw2+9);
  u32 sa=0xFFFF,sb=0xFFFF; int ntok=0;
  if(use_sub && inw && pr){ int t[48]; ntok=seg(s->w,s->wlen,t); sa=t[ntok-1]; if(ntok>1) sb=t[ntok-2]; }
  h[10]= mix(mix(sa,sb),ntok>3?3:ntok);
  h[11]= mix(mix(mix(sa,sb),s->pw1),11+(inw?0:c1));
  h[12]= mix(mix(c2,c3),12);
  h[13]= mix(mix(fold(c1)|(fold(c2)<<8),fold(c3)|(fold(c4)<<8)),13);
  h[14]= mix(s->fwh,14);
  { u32 suf=0; int st=s->wlen>3?s->wlen-3:0; for(int k=st;k<s->wlen&&k<64;k++) suf=suf*257+s->w[k];
    h[15]= mix(suf | ((u32)(s->wlen>15?15:s->wlen)<<24),15); }
  h[16]= mix(c1 | (c3<<8),16);
  h[17]= mix(c2 | (c4<<8),17);
  h[18]= mix(((u32)(s->linepos>255?255:s->linepos)<<8)|(u32)c1,18);
  h[19]= mix(s->pw1 ^ ((u32)(s->wlen>15?15:s->wlen)<<24),19);
  h[20]= mix(indirect[(c2<<8)|c1],20);
  { u32 dist=histlen-lastpos[c1]; if(dist>4095)dist=4095; h[21]= mix(dist | ((u32)c1<<12),21); }
  h[22]= mix((u32)(s->wcount>31?31:s->wcount) | ((u32)c1<<8),22);
  h[23]= mix(s->pw1 ^ ((u32)(s->wlen>0?s->w[0]:0)<<24) ^ ((u32)(s->wlen>15?15:s->wlen)<<16),23);
  { u32 bc=0; for(int k=0;k<3;k++){ int cc=(k==0?c1:(k==1?c2:c3)); int cl=(cc>='a'&&cc<='z')?0:(cc>='A'&&cc<='Z')?1:(cc>='0'&&cc<='9')?2:(cc==' ')?3:(cc=='\n')?4:(cc>=0x80)?5:6; bc|=((u32)cl)<<(k*4); } h[24]= mix(bc,24); }
  h[25]= mix(c1 | (c2<<8) | (c3<<16),25);
  cls_c1 = (c1>='a'&&c1<='z')?0:(c1>='A'&&c1<='Z')?1:(c1>='0'&&c1<='9')?2:(c1==' ')?3:(c1=='\n')?4:(c1>=0x80)?5:6;
  wl_sel = (s->wlen>7?7:s->wlen)*8+cls_c1; // 64
}

// ---------- tables ----------
static inline Slot* find(int i,u32 hv,int c0,int alloc){
  u32 x=(hv^((u32)c0*0x2545F491u))*0x9E3779B1u; x^=x>>15; x*=0x85EBCA6Bu; x^=x>>13;
  u32 s=x>>(32-TBITS[i]); u8 chk=(u8)(x&255)|1; Slot *a=&T[i][s], *b=&T[i][s^1];
  if(a->n && a->chk==chk) return a; if(b->n && b->chk==chk) return b;
  if(!alloc) return NULL;
  Slot *v=(a->n<=b->n)?a:b; v->p=32768; v->n=0; v->chk=chk; return v;
}
static inline void supd(Slot *s,int bit,int lim){ int n=s->n; int tg=bit?65535:0; int d=(int)(((int64_t)(tg-(int)s->p)*rt[n])>>16); int p=(int)s->p+d; if(p<24)p=24; if(p>65511)p=65511; s->p=(u16)p; if(n<lim) s->n++; }
static inline float slogit(Slot *s){ return stab[s->p>>4]; }

// ---------- mixers / APM ----------
static const int SELN[NMIX]={256,64,256,128,64};
static float *W1[NMIX]; static float *W2; static float lr1=0.003f, lr2=0.0015f;
typedef struct { float *t; int lo; float w; int ctx; } APM;
static APM apm1, apm2;
static void apm_init(APM *a,int n){ a->t=(float*)malloc(sizeof(float)*n*33); for(int i=0;i<n;i++) for(int j=0;j<33;j++) a->t[i*33+j]=squashf((j-16)*0.5f); }
static inline float apm_pp(APM *a,int ctx,float st){ float pos=(clampf(st,-7.99f,7.99f)+8.f)*2.f; int lo=(int)pos; float w=pos-lo; a->lo=lo; a->w=w; a->ctx=ctx; return a->t[ctx*33+lo]*(1-w)+a->t[ctx*33+lo+1]*w; }
static inline void apm_up(APM *a,int bit,float rate){ float *p=&a->t[a->ctx*33+a->lo]; p[0]+=(bit-p[0])*rate*(1-a->w); p[1]+=(bit-p[1])*rate*a->w; }

typedef struct { Slot *sl[NCTX]; Slot *s0; float x[NIN]; int sel[NMIX]; float pm[NMIX], d[NMIX]; float pmix,a1,a2,pf; int mact[NMM],mebit[NMM],midx[NMM]; int c0,bp; } PS;
static PS ps;
static int lenb(int l){ return l<=0?0:(l>=28?7:1+(l>>2)); }

static float predict(int c0,int bp,int alloc){
  int k=0,nhit=0; ps.c0=c0; ps.bp=bp;
  for(int i=0;i<NCTX;i++){ Slot *s=find(i,h[i],c0,alloc); ps.sl[i]=s;
    if(s&&s->n){ ps.x[k++]=slogit(s); ps.x[k++]=((float)s->p-32768.f)*(1.f/32768.f)*conf[s->n]; if(i>=2)nhit++; } else { ps.x[k++]=0; ps.x[k++]=0; } }
  { Slot *s=&o0t[c0]; ps.s0=s; if(s->n){ ps.x[k++]=slogit(s); ps.x[k++]=((float)s->p-32768.f)*(1.f/32768.f)*conf[s->n]; } else { ps.x[k++]=0; ps.x[k++]=0; } }
  int mlb=0;
  for(int j=0;j<NMM;j++){ ps.mact[j]=0; int l=bs.mlen[j];
    if(l>0){ int pb=hist[bs.mptr[j]]; if(((pb|256)>>(8-bp))==c0){ int eb=(pb>>(7-bp))&1; int li=l>63?63:l; float st=clampf(logf(msm[j][li]/(1.f-msm[j][li])),-8,8);
        ps.mact[j]=1; ps.mebit[j]=eb; ps.midx[j]=li; ps.x[k++]=eb?st:-st; ps.x[k++]=(eb?1.f:-1.f)*(l>=16?1.f:0.5f); if(j==0) mlb=lenb(l); continue; } }
    ps.x[k++]=0; ps.x[k++]=0; }
  ps.x[k++]=0.4f;
  ps.sel[0]=c0; ps.sel[1]=mlb*8+bp; ps.sel[2]=(int)(bs.c4&255); ps.sel[3]=(nhit>15?15:nhit)*8+bp; ps.sel[4]=(wl_sel&63);
  float o2=0; for(int m=0;m<NMIX;m++){ float *w=&W1[m][ps.sel[m]*NIN]; float dot=0; for(int i=0;i<NIN;i++) dot+=w[i]*ps.x[i];
    dot=clampf(dot,-10,10); ps.d[m]=dot; ps.pm[m]=squashf(dot); }
  float *w2=&W2[c0*NMIX]; for(int m=0;m<NMIX;m++) o2+=w2[m]*ps.d[m];
  o2=clampf(o2,-10,10); ps.pmix=squashf(o2);
  ps.a1=apm_pp(&apm1,c0,o2); ps.a2=apm_pp(&apm2,c0*256+(bs.c4&255),o2);
  float pf=(ps.pmix+ps.a1+2*ps.a2)*0.25f; ps.pf=clampf(pf,1.f/16384,1.f-1.f/16384); return ps.pf;
}
static void update(int bit){
  for(int i=0;i<NCTX;i++) if(ps.sl[i]) supd(ps.sl[i],bit,LIMIT[i]);
  supd(ps.s0,bit,255);
  for(int j=0;j<NMM;j++) if(ps.mact[j]){ float *p=&msm[j][ps.midx[j]]; *p+=((bit==ps.mebit[j])-*p)*0.02f; *p=clampf(*p,0.02f,0.9995f); }
  for(int m=0;m<NMIX;m++){ float e=(bit-ps.pm[m])*lr1; float *w=&W1[m][ps.sel[m]*NIN]; for(int i=0;i<NIN;i++) w[i]+=e*ps.x[i]; }
  { float e=(bit-ps.pmix)*lr2; float *w2=&W2[ps.c0*NMIX]; for(int m=0;m<NMIX;m++) w2[m]+=e*ps.d[m]; }
  apm_up(&apm1,bit,0.02f); apm_up(&apm2,bit,0.02f);
}

// ---------- init ----------
static void init_model(int tb_shift){
  for(int i=0;i<NCTX;i++){ int b=TBITS[i]-tb_shift; if(b<16)b=16; TBITS[i]=b; T[i]=(Slot*)calloc((size_t)1<<b,sizeof(Slot)); if(!T[i]){fprintf(stderr,"oom\n");exit(1);} }
  for(int c=0;c<256;c++){ o0t[c].p=32768; o0t[c].n=0; }
  for(int n=0;n<256;n++){ rt[n]=(int)(65536.0/(n+1.5)); conf[n]=n>=4?1.f:n/4.f; }
  for(int i=0;i<4096;i++){ float p=(i+0.5f)/4096.f; stab[i]=clampf(logf(p/(1-p)),-8.3f,8.3f); }
  for(int m=0;m<NMIX;m++){ W1[m]=(float*)malloc(sizeof(float)*SELN[m]*NIN); for(int i=0;i<SELN[m]*NIN;i++) W1[m][i]=0.12f; }
  W2=(float*)malloc(sizeof(float)*256*NMIX); for(int i=0;i<256*NMIX;i++) W2[i]=1.0f/NMIX;
  apm_init(&apm1,256); apm_init(&apm2,65536);
  hist=(u8*)malloc(MAXH);
  for(int j=0;j<NMM;j++){ mht[j]=(u32*)calloc((size_t)1<<MHB,4); for(int l=0;l<64;l++) msm[j][l]=1.f-1.f/(l+2.f); }
  memset(indirect,0,sizeof(indirect)); memset(lastpos,0,sizeof(lastpos));
  memset(&bs,0,sizeof(bs)); compute_ctx(&bs);
}

// ---------- train / eval ----------
static int train_bytes(const u8 *d,size_t n,double *bits_acc){
  for(size_t i=0;i<n;i++){ int c=d[i]; int c0=1; compute_ctx(&bs);
    for(int bp=0;bp<8;bp++){ int bit=(c>>(7-bp))&1; float p=predict(c0,bp,1); *bits_acc+=-log2f(bit?p:1-p); update(bit); c0=c0*2+bit; }
    bs_push(&bs,c); }
  return 0;
}
typedef struct { float nll; u8 t1,t5,pad[2]; } Rec;
static void dist256(float *pr256){ // full byte distribution from node predictions (no allocation / no update)
  float p1[256]; for(int c0=1;c0<256;c0++){ int bp=0; int t=c0; while(t>1){t>>=1;bp++;} p1[c0]=predict(c0,bp,0); }
  for(int v=0;v<256;v++){ float pr=1.f; int c0=1; for(int b=7;b>=0;b--){ int bit=(v>>b)&1; pr*= bit?p1[c0]:(1.f-p1[c0]); c0=c0*2+bit; } pr256[v]=pr; }
}
static int dump_prior(const char *wpath,const char *opath,int topk){
  if(topk<1) topk=1; if(topk>256) topk=256;
  FILE *f=fopen(wpath,"rb"); if(!f){fprintf(stderr,"no windows\n");return -1;}
  u32 nw,skip; fread(&nw,4,1,f); fread(&skip,4,1,f);
  long total=0;
  for(u32 w=0;w<nw;w++){ u32 len; fread(&len,4,1,f); fseek(f,len,SEEK_CUR); if(len>skip) total+=len-skip; }
  fseek(f,8,SEEK_SET);
  FILE *fo=fopen(opath,"wb"); if(!fo){fclose(f);return -1;}
  const char magic[8]={'N','A','N','C','M','9','P','1'}; fwrite(magic,1,8,fo);
  fwrite(&total,4,1,fo); fwrite(&topk,4,1,fo);
  BS save=bs; u32 hsave=histlen; u32 fsave=mm_floor;
  u32 *mht_save[NMM]={0}; int restore_mht=ev_match_hist;
  if(restore_mht) for(int j=0;j<NMM;j++){ size_t bytes=(size_t)1<<MHB; mht_save[j]=(u32*)malloc(bytes*4); memcpy(mht_save[j],mht[j],bytes*4); }
  float pr256[256]; u8 ids[256]; float ps[256]; u8 used[256];
  for(u32 w=0;w<nw;w++){ u32 len; fread(&len,4,1,f); u8 *buf=(u8*)malloc(len); fread(buf,1,len,f);
    histlen=hsave; mm_floor=ev_match_hist?0:histlen; memset(&bs,0,sizeof(bs));
    for(u32 i=0;i<len;i++){ int c=buf[i]; compute_ctx(&bs);
      if(i>=skip){ dist256(pr256); memset(used,0,256);
        for(int j=0;j<topk;j++){ int bi=-1; float bp=-1; for(int v=0;v<256;v++) if(!used[v]&&pr256[v]>bp){bp=pr256[v];bi=v;} used[bi]=1; ids[j]=(u8)bi; ps[j]=bp; }
        fwrite(&c,1,1,fo); fwrite(ids,1,topk,fo); fwrite(ps,4,topk,fo); }
      bs_push(&bs,c); }
    if(restore_mht) for(int j=0;j<NMM;j++) memcpy(mht[j],mht_save[j],((size_t)1<<MHB)*4);
    free(buf); }
  if(restore_mht) for(int j=0;j<NMM;j++) free(mht_save[j]);
  bs=save; histlen=hsave; mm_floor=fsave;
  fclose(f); fclose(fo);
  fprintf(stderr,"CM prior dumped: %s (%ld bytes, top-%d)\n",opath,total,topk);
  return 0;
}

static void eval_file(const char *wpath,int online,const char *opath,double *o_nats,double *o_t1,double *o_t5,long *o_n){
  FILE *f=fopen(wpath,"rb"); if(!f){fprintf(stderr,"no windows\n");exit(1);} u32 nw,skip; fread(&nw,4,1,f); fread(&skip,4,1,f);
  FILE *fo=fopen(opath,"wb"); BS save=bs; u32 hsave=histlen; u32 fsave=mm_floor; double nats=0,t1=0,t5=0; long n=0; float pr256[256];
  u32 *mht_save[NMM]={0}; int restore_mht = ev_match_hist && !online;
  if(restore_mht) for(int j=0;j<NMM;j++){ size_t bytes=(size_t)1<<MHB; mht_save[j]=(u32*)malloc(bytes*4); if(!mht_save[j]){fprintf(stderr,"oom mht\n");exit(1);} memcpy(mht_save[j],mht[j],bytes*4); }
  for(u32 w=0;w<nw;w++){ u32 len; fread(&len,4,1,f); u8 *buf=(u8*)malloc(len); fread(buf,1,len,f);
    if(!online){ histlen=hsave; }
    if(ev_match_hist) mm_floor=0; else mm_floor=histlen;
    if(!online || !ev_carry) memset(&bs,0,sizeof(bs));
    for(u32 i=0;i<len;i++){ int c=buf[i]; compute_ctx(&bs);
      if(i>=skip){ dist256(pr256); float pt=pr256[c]; int rank=0; int top1=1; for(int v=0;v<256;v++) if(pr256[v]>pt) rank++; top1=(rank==0);
        Rec r; r.nll=0; r.t1=top1; r.t5=(rank<5); r.pad[0]=r.pad[1]=0;
        if(online){ int c0=1; double nl=0; for(int bp=0;bp<8;bp++){ int bit=(c>>(7-bp))&1; float p=predict(c0,bp,1); nl+=-logf(bit?p:1-p); update(bit); c0=c0*2+bit; } r.nll=(float)nl; }
        else r.nll=-logf(fmaxf(pt,1e-9f));
        nats+=r.nll; t1+=r.t1; t5+=r.t5; n++; fwrite(&r,sizeof(r),1,fo);
      } else if(online){ int c0=1; for(int bp=0;bp<8;bp++){ int bit=(c>>(7-bp))&1; predict(c0,bp,1); update(bit); c0=c0*2+bit; } }
      bs_push(&bs,c); }
    if(restore_mht) for(int j=0;j<NMM;j++) memcpy(mht[j],mht_save[j],((size_t)1<<MHB)*4);
    free(buf); }
  fclose(f); fclose(fo);
  if(restore_mht) for(int j=0;j<NMM;j++) free(mht_save[j]);
  if(!online){ bs=save; histlen=hsave; } mm_floor=fsave;
  *o_nats=nats/n; *o_t1=t1/n; *o_t5=t5/n; *o_n=n;
}

// ---------- checkpoint (weights) ----------
#define CK_MAGIC "NANCM9CX"
#define CK_VERSION 2u
typedef struct { u32 idx; Slot s; } CKEnt;
typedef struct {
  char magic[8];
  u32 version,nctx,nmix,nin,nmm,mhb;
  u32 tbs,use_sub,nmerge;
  u32 histlen,mm_floor;
  u32 tbits[NCTX];
  float lr1,lr2;
  BS bs;
  float msm[NMM][64];
} CKHead;

static int ck_write(FILE *f,const void *p,size_t n){
  return fwrite(p,1,n,f)==n?0:-1;
}
static int save_checkpoint(const char *path){
  FILE *f=fopen(path,"wb"); if(!f){ fprintf(stderr,"cannot write %s\n",path); return -1; }
  CKHead H; memset(&H,0,sizeof(H)); memcpy(H.magic,CK_MAGIC,8);
  H.version=CK_VERSION; H.nctx=NCTX; H.nmix=NMIX; H.nin=NIN; H.nmm=NMM; H.mhb=MHB;
  H.tbs=0; H.use_sub=use_sub; H.nmerge=nmerge;
  H.histlen=histlen; H.mm_floor=mm_floor; H.lr1=lr1; H.lr2=lr2;
  for(int i=0;i<NCTX;i++) H.tbits[i]=(u32)TBITS[i];
  H.bs=bs; memcpy(H.msm,msm,sizeof(msm));
  if(ck_write(f,&H,sizeof(H))){ fclose(f); return -1; }
  CKEnt *buf=(CKEnt*)malloc(sizeof(CKEnt)*8192); if(!buf){ fclose(f); return -1; }
  for(int i=0;i<NCTX;i++){
    u64 sz=(u64)1<<TBITS[i], cnt=0, bn=0;
    for(u64 j=0;j<sz;j++) if(T[i][j].n) cnt++;
    if(ck_write(f,&cnt,8)){ free(buf); fclose(f); return -1; }
    for(u64 j=0;j<sz;j++){ Slot s=T[i][j]; if(!s.n) continue;
      buf[bn].idx=(u32)j; buf[bn].s=s; bn++;
      if(bn==8192){ if(ck_write(f,buf,sizeof(CKEnt)*bn)){ free(buf); fclose(f); return -1; } bn=0; } }
    if(bn && ck_write(f,buf,sizeof(CKEnt)*bn)){ free(buf); fclose(f); return -1; }
  }
  u64 cnt=0,bn=0;
  for(int j=0;j<256;j++) if(o0t[j].n) cnt++;
  if(ck_write(f,&cnt,8)){ free(buf); fclose(f); return -1; }
  for(int j=0;j<256;j++) if(o0t[j].n){ buf[bn].idx=(u32)j; buf[bn].s=o0t[j]; bn++;
    if(bn==8192){ if(ck_write(f,buf,sizeof(CKEnt)*bn)){ free(buf); fclose(f); return -1; } bn=0; } }
  if(bn && ck_write(f,buf,sizeof(CKEnt)*bn)){ free(buf); fclose(f); return -1; }
  free(buf);
  if(histlen && ck_write(f,hist,histlen)){ fclose(f); return -1; }
  for(int j=0;j<NMM;j++) if(fwrite(mht[j],4,(size_t)1<<MHB,f)!=(size_t)1<<MHB){ fclose(f); return -1; }
  for(int m=0;m<NMIX;m++) if(fwrite(W1[m],sizeof(float),(size_t)SELN[m]*NIN,f)!=(size_t)SELN[m]*NIN){ fclose(f); return -1; }
  if(fwrite(W2,sizeof(float),(size_t)256*NMIX,f)!=(size_t)256*NMIX){ fclose(f); return -1; }
  if(fwrite(apm1.t,sizeof(float),(size_t)256*33,f)!=(size_t)256*33){ fclose(f); return -1; }
  if(fwrite(apm2.t,sizeof(float),(size_t)65536*33,f)!=(size_t)65536*33){ fclose(f); return -1; }
  long sz=ftell(f); fclose(f);
  fprintf(stderr,"checkpoint saved: %s (%.1f MiB)\n",path,sz/1048576.0);
  return 0;
}
static int load_checkpoint(const char *path){
  FILE *f=fopen(path,"rb"); if(!f){ fprintf(stderr,"cannot read %s\n",path); return -1; }
  CKHead H; if(fread(&H,sizeof(H),1,f)!=1){ fclose(f); return -1; }
  if(memcmp(H.magic,CK_MAGIC,8)||H.version!=CK_VERSION||H.nctx!=NCTX||H.nmix!=NMIX||H.nin!=NIN||
     H.nmm!=NMM||H.mhb!=(u32)MHB){ fprintf(stderr,"checkpoint format mismatch\n"); fclose(f); return -1; }
  if(H.nmerge>0 && (!pr || (int)H.nmerge!=nmerge)){ fprintf(stderr,"checkpoint needs matching -bpe merges (%u)\n",H.nmerge); fclose(f); return -1; }
  for(int i=0;i<NCTX;i++){ if(H.tbits[i]<16||H.tbits[i]>28){ fclose(f); return -1; } TBITS[i]=(int)H.tbits[i]; }
  init_model(0);
  use_sub=(int)H.use_sub;
  histlen=H.histlen; mm_floor=H.mm_floor; lr1=H.lr1; lr2=H.lr2; bs=H.bs;
  memcpy(msm,H.msm,sizeof(msm));
  if(histlen>MAXH || mm_floor>histlen){ fclose(f); return -1; }
  CKEnt *buf=(CKEnt*)malloc(sizeof(CKEnt)*8192); if(!buf){ fclose(f); return -1; }
  for(int i=0;i<NCTX;i++){
    u64 cnt; if(fread(&cnt,8,1,f)!=1){ free(buf); fclose(f); return -1; }
    u64 sz=(u64)1<<TBITS[i];
    while(cnt){ u64 n=cnt>8192?8192:cnt;
      if(fread(buf,sizeof(CKEnt),n,f)!=n){ free(buf); fclose(f); return -1; }
      for(u64 k=0;k<n;k++){ if(buf[k].idx>=sz){ free(buf); fclose(f); return -1; } T[i][buf[k].idx]=buf[k].s; }
      cnt-=n; }
  }
  { u64 cnt; if(fread(&cnt,8,1,f)!=1){ free(buf); fclose(f); return -1; }
    while(cnt){ u64 n=cnt>8192?8192:cnt;
      if(fread(buf,sizeof(CKEnt),n,f)!=n){ free(buf); fclose(f); return -1; }
      for(u64 k=0;k<n;k++){ if(buf[k].idx>=256){ free(buf); fclose(f); return -1; } o0t[buf[k].idx]=buf[k].s; }
      cnt-=n; } }
  free(buf);
  if(histlen && fread(hist,1,histlen,f)!=histlen){ fclose(f); return -1; }
  for(int j=0;j<NMM;j++) if(fread(mht[j],4,(size_t)1<<MHB,f)!=(size_t)1<<MHB){ fclose(f); return -1; }
  for(int m=0;m<NMIX;m++) if(fread(W1[m],sizeof(float),(size_t)SELN[m]*NIN,f)!=(size_t)SELN[m]*NIN){ fclose(f); return -1; }
  if(fread(W2,sizeof(float),(size_t)256*NMIX,f)!=(size_t)256*NMIX){ fclose(f); return -1; }
  if(fread(apm1.t,sizeof(float),(size_t)256*33,f)!=(size_t)256*33){ fclose(f); return -1; }
  if(fread(apm2.t,sizeof(float),(size_t)65536*33,f)!=(size_t)65536*33){ fclose(f); return -1; }
  fclose(f);
  compute_ctx(&bs);
  fprintf(stderr,"checkpoint loaded: %s (hist=%u floor=%u)\n",path,histlen,mm_floor);
  return 0;
}

int main(int argc,char **argv){
  const char *evalp=NULL,*bpep=NULL,*outp="res",*savep=NULL,*loadp=NULL,*dumpp=NULL,*dumpinp=NULL; int dumptop=8,dumponly=0; int tbs=0; long evalat[32]; int nea=0; int online_final=1; char *files[16]; int nf=0; long maxb=1L<<60;
  for(int i=1;i<argc;i++){
    if(!strcmp(argv[i],"-eval")) evalp=argv[++i];
    else if(!strcmp(argv[i],"-bpe")) bpep=argv[++i];
    else if(!strcmp(argv[i],"-o")) outp=argv[++i];
    else if(!strcmp(argv[i],"-tshift")) tbs=atoi(argv[++i]);
    else if(!strcmp(argv[i],"-nosub")) use_sub=0;
    else if(!strcmp(argv[i],"-lr1")) lr1=atof(argv[++i]);
    else if(!strcmp(argv[i],"-lr2")) lr2=atof(argv[++i]);
    else if(!strcmp(argv[i],"-maxb")) maxb=atol(argv[++i]);
    else if(!strcmp(argv[i],"-noonline")) online_final=0;
    else if(!strcmp(argv[i],"-dump")) dumpp=argv[++i];
    else if(!strcmp(argv[i],"-dumpin")) dumpinp=argv[++i];
    else if(!strcmp(argv[i],"-dumptop")) dumptop=atoi(argv[++i]);
    else if(!strcmp(argv[i],"-dumponly")) dumponly=1;
    else if(!strcmp(argv[i],"-save")) savep=argv[++i];
    else if(!strcmp(argv[i],"-load")) loadp=argv[++i];
    else if(!strcmp(argv[i],"-evmatch")) ev_match_hist=1;
    else if(!strcmp(argv[i],"-evcarry")) ev_carry=1;
    else if(!strcmp(argv[i],"-evalat")){ char *s=strtok(argv[++i],","); while(s){ evalat[nea++]=atol(s); s=strtok(NULL,","); } }
    else if(!strcmp(argv[i],"-segdump")){ // segdump <merges> <file> <out>
      bpe_load(argv[i+1]); FILE *f=fopen(argv[i+2],"rb"); fseek(f,0,SEEK_END); long n=ftell(f); fseek(f,0,SEEK_SET); u8 *d=(u8*)malloc(n); fread(d,1,n,f); fclose(f);
      FILE *fo=fopen(argv[i+3],"wb"); long i0=0; while(i0<n){ if(isword(d[i0])){ long j=i0; while(j<n&&isword(d[j])) j++; int t[48]; long L=j-i0; long off=0;
          // long words: segment in 48-byte chunks (C context cap); python test restricts to words <=48
          while(off<L){ int cl=(L-off>48)?48:(int)(L-off); int nt=seg(d+i0+off,cl,t); for(int k=0;k<nt;k++){ u32 v=t[k]; fwrite(&v,4,1,fo);} off+=cl; } i0=j; }
        else { u32 v=d[i0]; fwrite(&v,4,1,fo); i0++; } }
      fclose(fo); return 0; }
    else files[nf++]=argv[i];
  }
  if(bpep && bpe_load(bpep)<0){ fprintf(stderr,"bpe load failed\n"); return 1; }
  if(bpep) fprintf(stderr,"bpe merges loaded: %d subword=%d\n",nmerge,use_sub);
  if(loadp){ if(load_checkpoint(loadp)) return 1; }
  else init_model(tbs);
  long total=0; double bits_win=0; long win=0; int eai=0; double nats,t1,t5; long ne;
  for(int fi=0;fi<nf;fi++){ FILE *f=fopen(files[fi],"rb"); if(!f){fprintf(stderr,"cannot open %s\n",files[fi]);return 1;} fseek(f,0,SEEK_END); long n=ftell(f); fseek(f,0,SEEK_SET);
    if(n>maxb-total) n=maxb-total; u8 *d=(u8*)malloc(n); fread(d,1,n,f); fclose(f);
    long done=0; const long CH=1<<18;
    while(done<n){ long m=n-done<CH?n-done:CH; train_bytes(d+done,m,&bits_win); done+=m; total+=m; win+=m;
      if(win>=(1L<<20)){ fprintf(stderr,"[train %s] %ld bytes  running bpc(last 1M)=%.3f\n",files[fi],total,bits_win/win); win=0; bits_win=0; }
      while(eai<nea && total>=evalat[eai]){ if(evalp){ char o[256]; snprintf(o,256,"%s_frozen_%ld.bin",outp,evalat[eai]); eval_file(evalp,0,o,&nats,&t1,&t5,&ne);
          printf("EVAL frozen @%ld bytes: nats=%.4f bpc=%.4f top1=%.4f top5=%.4f n=%ld\n",total,nats,nats/log(2),t1,t5,ne); fflush(stdout);} eai++; } }
    free(d); }
  if(savep && save_checkpoint(savep)) return 1;
  if(dumpp){
    const char *inp=dumpinp?dumpinp:evalp;
    if(!inp){ fprintf(stderr,"-dump needs -dumpin or -eval\n"); return 1; }
    if(dump_prior(inp,dumpp,dumptop)) return 1;
    if(dumponly) return 0;
  }
  if(evalp){ char o[256]; snprintf(o,256,"%s_frozen.bin",outp); eval_file(evalp,0,o,&nats,&t1,&t5,&ne);
    printf("FINAL frozen @%ld bytes: nats=%.4f bpc=%.4f top1=%.4f top5=%.4f n=%ld\n",total,nats,nats/log(2),t1,t5,ne);
    if(online_final){ snprintf(o,256,"%s_online.bin",outp); eval_file(evalp,1,o,&nats,&t1,&t5,&ne);
      printf("FINAL online @%ld bytes: nats=%.4f bpc=%.4f top1=%.4f top5=%.4f n=%ld\n",total,nats,nats/log(2),t1,t5,ne); } }
  return 0;
}
