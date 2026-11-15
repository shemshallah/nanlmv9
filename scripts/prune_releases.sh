#!/usr/bin/env bash
# Prune old dev releases from this repository. Only tags beginning with
# dev-v9- are considered; stable releases are never touched.
#
# usage: prune_releases.sh [keep=3] [dry_run=false]
set -euo pipefail
keep="${1:-3}"
dry="${2:-false}"
repo="${GH_REPO:-$(gh repo view --json nameWithOwner -q .nameWithOwner)}"

if [[ ! "$keep" =~ ^[0-9]+$ ]] || (( keep < 1 )); then
  echo "keep must be a positive integer" >&2
  exit 2
fi

mapfile -t tags < <(
  gh release list --repo "$repo" --limit 100 \
    --json tagName,createdAt \
    --jq '.[] | select(.tagName | startswith("dev-v9-")) | [.createdAt, .tagName] | @tsv' \
  | sort -r | cut -f2
)

echo "repo=$repo keep=$keep found=${#tags[@]} dev releases"
if (( ${#tags[@]} <= keep )); then
  echo "nothing to prune"
  exit 0
fi

for ((i = keep; i < ${#tags[@]}; i++)); do
  tag="${tags[$i]}"
  if [[ "$dry" == "true" ]]; then
    echo "would delete release + tag: $tag"
  else
    echo "deleting release + tag: $tag"
    gh release delete "$tag" --repo "$repo" --yes --cleanup-tag
  fi
done

if [[ "$dry" != "true" ]]; then
  gh release list --repo "$repo" --limit 100
fi
