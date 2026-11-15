CC ?= cc
CFLAGS ?= -O3 -Wall -Wextra
LDLIBS ?= -lm

.PHONY: all clean check

all: nancm

nancm: nancm.c
	$(CC) $(CFLAGS) -o $@ $< $(LDLIBS)

check: nancm
	python3 ../scripts/verify_bpe.py --nancm ./nancm --merges ./bpe_merges.txt

clean:
	rm -f nancm
