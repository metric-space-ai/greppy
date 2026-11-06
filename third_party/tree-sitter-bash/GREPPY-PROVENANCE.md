# Pinned Bash grammar

This is the Rust/C portion of upstream `tree-sitter-bash` 0.25.1, MIT licensed.
The original license is retained in LICENSE.

- Upstream commit/tag: a06c2e4415e9bc0346c6b86d401879ffb44058f7 / v0.25.1.
- Source archive: https://static.crates.io/crates/tree-sitter-bash/tree-sitter-bash-0.25.1.crate
- Archive SHA-256: 9e5ec769279cc91b561d3df0d8a5deb26b0ad40d183127f409494d6d8fc53062.
- Generator: tree-sitter 0.25.10, language ABI15, the same version as the released parser.
- Generator acquisition and compressed binary SHA-256 are pinned in
  ../../tools/regenerate_bash_grammar.py. The generator is build tooling only;
  normal Greppy builds compile the checked-in C source and require no Node or
  Tree-sitter executable at build time or runtime.

The grammar change gives `=` the same comparison precedence as `==`/`!=` in
ordinary conditional expressions and includes it in their recursive pattern
branch. The separate arithmetic assignment tables are unchanged. Bash accepts
all three comparison operators in conditional expressions. The upstream
special-case `=` rule appears only directly under test_command, so it cannot
serve as the left side of a logical conjunction/disjunction. Greppy's CLI edit
validation and extraction both load this same generated language; neither
rewrites the user's bytes or ignores missing/error nodes.

Only the highlighting data filename changes from highlights.scm to
highlights.query, with the Rust include_str! adjusted. Its bytes are unchanged.
This avoids misclassifying Tree-sitter query data as Scheme while importing the
source. This is not a change to the highlighting query or a Scheme validator fix.

Reproduction (on the existing Linux x86_64 CI host):

    python3 tools/regenerate_bash_grammar.py --output-dir <new-empty-output-directory>

The helper verifies the pinned generator archive, records the actual generator,
Node version, ABI, grammar hash and all generated output hashes, and requires
byte-identical committed outputs. CI retains the generation receipt and exact
outputs. The checked-in parser.c, grammar.json, node-types.json and parser.h were
produced together by CI run42194399650, artifact13490663879. GENERATION.json records
the actual tools and input/output byte hashes; subsequent CI regenerates and
checks the committed files rather than accepting hand-edited generated C.

The CLI regression exercises actual write/dry-run/refusal and unchanged bytes,
with Bash -n as an independent syntax oracle. Its mutation temporarily selects
the checksum-pinned released registry grammar and requires the original exit13 /
missing ]] failure, then restores the local grammar and verifies green again.
No automatic syntax-error allowance or unrelated language relaxation is used.
