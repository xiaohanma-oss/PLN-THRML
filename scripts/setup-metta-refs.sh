#!/usr/bin/env bash
# setup-metta-refs.sh — Copy MeTTa language references from a local
# hyperon-experimental clone into docs/references/metta-lang/.
#
# Usage:
#   ./scripts/setup-metta-refs.sh [path-to-hyperon-experimental]
#
# If no path is given, defaults to ~/hyperon-experimental.
# If hyperon-experimental is not cloned locally, clone it first:
#   git clone https://github.com/trueagi-io/hyperon-experimental ~/hyperon-experimental

set -euo pipefail

HYPERON="${1:-$HOME/hyperon-experimental}"

if [ ! -d "$HYPERON" ]; then
    echo "Error: hyperon-experimental not found at $HYPERON"
    echo "Clone it first:  git clone https://github.com/trueagi-io/hyperon-experimental $HYPERON"
    exit 1
fi

DEST="$(cd "$(dirname "$0")/.." && pwd)/docs/references/metta-lang"

mkdir -p "$DEST"/{spec,stdlib,examples}

# Language spec
cp "$HYPERON/docs/metta.md"         "$DEST/spec/"
cp "$HYPERON/docs/minimal-metta.md" "$DEST/spec/"

# Standard library
cp "$HYPERON/lib/src/metta/runner/stdlib/stdlib.metta" "$DEST/stdlib/"

# Curated examples
for f in a1_symbols b2_backchain c1_grounded_basic c2_spaces c3_pln_stv \
         d1_gadt d2_higherfunc e1_kb_write e2_states; do
    cp "$HYPERON/python/tests/scripts/${f}.metta" "$DEST/examples/"
done

echo "MeTTa references set up at $DEST"
echo "Files copied:"
find "$DEST" -type f | sort | sed 's|^|  |'
