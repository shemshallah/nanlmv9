#!/usr/bin/env bash
# Fetch the public v9 corpora into <outdir> (default: v9/corpora).
set -euo pipefail
out="${1:-v9/corpora}"
mkdir -p "$out"
base="https://raw.githubusercontent.com/pytorch/examples/main/word_language_model/data/wikitext-2"

fetch() {
  local url="$1" dest="$2" want="$3"
  if [[ -s "$dest" ]] && echo "$want  $dest" | sha256sum -c --status; then
    echo "cached: $dest"
    return
  fi
  echo "fetch: $url"
  curl -fsSL --retry 3 --retry-delay 2 --max-time 120 -o "$dest.tmp" "$url"
  echo "$want  $dest.tmp" | sha256sum -c --status
  mv "$dest.tmp" "$dest"
}

fetch "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt" \
  "$out/tinyshakespeare.txt" \
  86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed
fetch "$base/train.txt" "$out/wt2_train.txt" \
  9e9fa1ad55b1c2c95b08e37dd8e653f638fac2c6de904b79e813611eefbc985f
fetch "$base/valid.txt" "$out/wt2_valid.txt" \
  f0737ed31fc1329026e95cb8b98e19c2a182c39c240ab909dc31abf2f8af58e8
fetch "$base/test.txt" "$out/wt2_test.txt" \
  d790b833ef8cf03a90db7bf1271b7520b83c45ce07ba3c1a9699df81e239eca0

ls -lh "$out"
