#!/usr/bin/env bash
set -euo pipefail

snapshot_source="${1:-public_record_snapshot.json}"
snapshot_branch="${PUBLIC_SNAPSHOT_BRANCH:-public-snapshot}"

if [[ ! -f "$snapshot_source" ]]; then
  echo "Snapshot file not found: $snapshot_source" >&2
  exit 1
fi

snapshot_source="$(cd "$(dirname "$snapshot_source")" && pwd)/$(basename "$snapshot_source")"
snapshot_workspace="$(mktemp -d)"
snapshot_checkout="$snapshot_workspace/checkout"

cleanup() {
  git worktree remove --force "$snapshot_checkout" >/dev/null 2>&1 || true
  rm -rf "$snapshot_workspace"
}
trap cleanup EXIT

git config user.name "github-actions[bot]"
git config user.email "41898282+github-actions[bot]@users.noreply.github.com"

if git ls-remote --exit-code --heads origin "$snapshot_branch" >/dev/null 2>&1; then
  git fetch origin "$snapshot_branch"
  git worktree add --detach "$snapshot_checkout" "origin/$snapshot_branch"
else
  git worktree add --detach "$snapshot_checkout" HEAD
  (
    cd "$snapshot_checkout"
    git switch --orphan "$snapshot_branch"
    git rm -rf .
  )
fi

cp "$snapshot_source" "$snapshot_checkout/public_record_snapshot.json"
(
  cd "$snapshot_checkout"
  git add -f public_record_snapshot.json
  if git diff --cached --quiet; then
    echo "Verified public snapshot is unchanged."
    exit 0
  fi
  git commit -m "Refresh verified public record snapshot"
  git push origin "HEAD:$snapshot_branch"
)
