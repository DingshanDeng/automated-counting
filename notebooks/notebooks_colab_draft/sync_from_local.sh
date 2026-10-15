#!/usr/bin/env bash
# Copy the local package, pyproject.toml and splits.csv next to the Colab notebooks, so this
# folder can be uploaded to Google Drive as is. Rerun after changing src/ or the splits.
# The copies are gitignored (src/, splits/, pyproject.toml match the root .gitignore).
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
repo="$(cd "$here/../.." && pwd)"

rm -rf "$here/src" "$here/splits"
mkdir -p "$here/src" "$here/splits"
rsync -a --exclude "__pycache__" --exclude "*.egg-info" "$repo/src/cellcount" "$here/src/"
cp "$repo/pyproject.toml" "$here/"
cp "$repo/splits/splits.csv" "$here/splits/"
echo "synced into $here:"
(cd "$here" && find src splits pyproject.toml -type f | sort)
