# Bounded literal source previews

Human-readable `search-pattern --code` output bounds each oversized source
line independently of the result count. `--limit` still means number of hits;
`--all` does not disable the per-line preview bound.

Lines of at most 4096 UTF-8 bytes are printed unchanged. Larger lines show a
window of at most 3072 bytes, with explicit byte counts omitted before/after
and the original total. The window includes leading context around a display
match when available, without splitting UTF-8 characters. Fixed-string
positions are exact. A Rust-regex display anchor is best effort for grep's ERE
dialect; an unsupported display regex falls back to the start of the line and
never changes grep's hit set, counts, or exit code.

Every preview includes `greppy read-file` with an absolute, shell-quoted path
and an exact one-line range. This reads the current source file: it is not an
immutable search receipt, and a later edit may change it. Neither matching nor
previewing edits the original file. JSON output retains its existing envelope
and explicit output-budget contract.

Tests cover a middle match in a 200-KB JSON line, regex/fixed/all forms,
executable recovery for a path containing a quote, exact source preservation,
ordinary output, no-match exit status, and UTF-8 boundary/omission arithmetic.
