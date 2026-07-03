#!/usr/bin/env bash
# gen_corpus.sh <out_dir> <n_files>
#
# Generate a synthetic Rust repository with deterministic, predictable
# cross-file structure so the graph extractor has real CALLS / IMPORTS /
# TYPE_REF edges to find:
#
#   * Each module modNNN.rs defines `struct WidgetNNN` with `new()`,
#     a method, and a free function `build_NNN`.
#   * Module N's free function `use`s and calls module (N-1)'s struct
#     (`use crate::modMMM::WidgetMMM;` + `WidgetMMM::new()`), producing
#     a real cross-file IMPORTS + CALLS + TYPE_REF chain.
#   * lib.rs declares every module.
#
# Output is fully deterministic for a given (out_dir, n_files): the same
# inputs produce byte-identical files. That is what the DETERMINISM
# battle relies on.

set -euo pipefail

OUT="$1"
N="$2"

mkdir -p "$OUT/src"

# Cargo.toml so the tree looks like a real crate (the indexer keys off
# git, not cargo, but this keeps the corpus realistic).
cat > "$OUT/Cargo.toml" <<EOF
[package]
name = "battle_corpus"
version = "0.0.0"
edition = "2021"

[lib]
path = "src/lib.rs"
EOF

LIB="$OUT/src/lib.rs"
: > "$LIB"
echo "// Auto-generated battle corpus. $N modules." >> "$LIB"

i=0
while [[ "$i" -lt "$N" ]]; do
    name=$(printf 'mod%04d' "$i")
    echo "pub mod $name;" >> "$LIB"

    f="$OUT/src/$name.rs"
    {
        echo "// Module $i of the battle corpus."
        # Cross-file edge: depend on the previous module's struct.
        if [[ "$i" -gt 0 ]]; then
            prev=$(printf 'mod%04d' "$((i - 1))")
            pidx="$((i - 1))"
            echo "use crate::${prev}::Widget${pidx};"
        fi
        echo ""
        echo "/// Widget${i} is the canonical struct for module $i."
        echo "pub struct Widget${i} {"
        echo "    pub id: u64,"
        echo "    pub label: String,"
        echo "}"
        echo ""
        echo "impl Widget${i} {"
        echo "    pub fn new(id: u64) -> Self {"
        echo "        Self { id, label: String::from(\"w${i}\") }"
        echo "    }"
        echo ""
        echo "    pub fn rank(&self) -> u64 {"
        echo "        self.id.wrapping_mul(${i} + 1)"
        echo "    }"
        echo "}"
        echo ""
        # Free function that calls into the previous module: real
        # cross-file CALLS + TYPE_REF.
        if [[ "$i" -gt 0 ]]; then
            pidx="$((i - 1))"
            echo "pub fn build_${i}() -> Widget${pidx} {"
            echo "    let w = Widget${pidx}::new(${i} as u64);"
            echo "    w"
            echo "}"
        else
            echo "pub fn build_${i}() -> Widget${i} {"
            echo "    Widget${i}::new(${i} as u64)"
            echo "}"
        fi
    } > "$f"

    i="$((i + 1))"
done
