# Draft: the bash-smart lines for AGENTS.md (owner sign-off at smoke pass)

Proposed placement: its own two-line section between EDIT and the footer,
since it is neither search, navigation, read nor edit:

RUN:
  bash-smart -- CMD …        runs CMD untouched — same bytes, env and exit
                             code; long output arrives as its head and tail
                             with one line naming the hidden range, and
                             `greppy expand ID` prints exactly those lines

Guard-test additions (prompt_contract.rs) once the line is in:
- the RUN section lists exactly one command;
- the line promises nothing the binary does not hold (no classifier claims
  in 0.3.0 — no "the lines that matter", which is the 0.4.0 head);
- `--` appears in the signature (argv passthrough is part of the contract).

Explicitly NOT promised in 0.3.0: lifted signal lines (novelty/classifier),
per-line timestamps. The prompt sells only what ships: skeleton, collapse,
true gap line, exact expand.
