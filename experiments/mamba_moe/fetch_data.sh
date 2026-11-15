#!/usr/bin/env bash
# Fetch the NaNROUTER corpora used by the 32k hybrid tokenizer experiments.
# Override the destination with: bash fetch_data.sh /path/to/nanlm-data
set -euo pipefail
ROOT="${1:-$HOME/nanlm-data}"
WL="$ROOT/wordlists"
CORP="$ROOT/corpora"
mkdir -p "$WL" "$CORP/wt103" "$CORP/domains"

echo "== wordlists"
cd "$WL"
for tag in \
  wordlist-google-10000-english \
  wordlist-wordfreq \
  wordlist-david47k-top-english-wordlists \
  wordlist-dwyl-english-words \
  wordlist-en-wl-wordlist \
  wordlist-wordset-dictionary
do
  gh release download "$tag" -R shemshallah/NaNROUTER --clobber
done
for f in *.tar.zst; do
  d="${f%.tar.zst}"
  mkdir -p "$d"
  zstd -dc "$f" | tar -xf - -C "$d"
done

echo "== WikiText-103 raw"
cd "$CORP/wt103"
gh release download wikitext103 -R shemshallah/NaNROUTER \
  -p 'wiki.train.raw' -p 'wiki.valid.raw' -p 'wiki.test.raw' -p 'SHA256SUMS' --clobber
sha256sum -c SHA256SUMS

echo "== domain corpora"
cd "$CORP/domains"
for tag in corpora-humanities corpora-chemistry corpora-standards; do
  gh release download "$tag" -R shemshallah/NaNROUTER --clobber
done

echo "== mixed 150 MB training slice"
cd "$CORP"
{
  head -c 100000000 wt103/wiki.train.raw
  printf '\n'
  head -c 20000000 domains/humanities.txt
  printf '\n'
  head -c 10000000 domains/chemistry.txt
  printf '\n'
  head -c 20000000 domains/standards.txt
  printf '\n'
} > mixed_train.txt
wc -c mixed_train.txt
