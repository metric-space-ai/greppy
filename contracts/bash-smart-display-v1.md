# Bash-smart display and raw-output recovery

Command capture, command exit status and explicit raw expansion are distinct
from the default display. Raw stdout/stderr remain byte-preserving.

- The verdict and selected diagnostic/regex lines appear first. Short output
  (at most 80 total lines) also retains its raw skeleton; a short diagnostic
  may consequently occur again there. This is not an expansion-store failure.
- Longer output is folded. If an active index writer prevents opening the
  expansion store, folding still works and names the existing raw spool
  instead of inventing an expansion ID.
- An individual displayed line longer than 4096 bytes is a marked preview,
  retaining at most 2048 leading and 1024 trailing bytes plus an omission
  notice. Valid UTF-8 is not split. This applies to diagnostics, regex lifts,
  folded head/tail and short output: a data-URL stack cannot bypass the rule
  merely because it occupies one line.
- For each stream with an oversized line, output names its actual raw-log
  path and `greppy read-file` recovery. Original capture and explicit expansion
  are not rewritten to contain the preview or its marker.
- Errors and nonzero child exits remain errors. Previewing an enormous stack
  must not replace the actual diagnostic, invent a successful result or imply
  that omitted bytes were never produced.

The line formatter has standalone tests for giant data URLs, UTF-8 boundaries,
ordinary/binary byte preservation and writer failures. The CLI integration
regression additionally verifies both streams, the retained nonzero exit,
visible exception message, bounded display and byte-exact raw-log recovery.
