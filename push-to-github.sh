#!/usr/bin/env bash
# Push this repository to GitHub.
# Usage:  GHUSER=your-github-username ./push-to-github.sh [repo-name]
set -euo pipefail

REPO_NAME="${1:-nlp-text-measure-thesis}"
: "${GHUSER:?please set GHUSER, e.g. export GHUSER=your-name}"

git branch -M main
git remote remove origin 2>/dev/null || true
git remote add origin "https://github.com/${GHUSER}/${REPO_NAME}.git"

# Create the remote repository first (web UI) or use: gh repo create "$REPO_NAME" --public --source=. --push
echo "Pushing to https://github.com/${GHUSER}/${REPO_NAME}.git ..."
git -c http.sslBackend=openssl push -u origin main

echo
echo "Done. Open: https://github.com/${GHUSER}/${REPO_NAME}"
