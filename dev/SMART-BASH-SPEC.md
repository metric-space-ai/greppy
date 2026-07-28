# bash-smart — output specification and training annex

Owner-approved design, 2026-07-28/29. The shared laws of `dev/NAV-OUTPUT-SPEC.md`
apply. Output is English.

## What it is

A wrapper that runs a shell command untouched and delivers its OUTPUT in the
expand economy: short output verbatim, long output as skeleton + lifted signal
+ the whole thing behind a priced id. The command, its environment and its exit
code are never altered — the name carries the trade, like `read-smart`.

```
  bash-smart -- CMD …        runs the command; long output arrives as its head
                             and tail, the lines that matter lifted from the
                             middle, the rest behind an expand id
```

## The four layers

**1. Skeleton — mechanical, always.** Under ~80 output lines: everything,
verbatim, untouched. Above: first 20 lines, a gap line, last 30 (the trained
place for verdicts), exit code passed through. On exit ≠ 0 the tail widens.
stderr is preferred verbatim (a file-descriptor fact, not a judgment); stdout
pages. The full raw output is stored behind the id — today's harnesses truncate
blindly and the cut part is gone forever; here overflow becomes retrievable.

**2. Repetition collapse — arithmetic, before any model.** Identical and
template-identical consecutive lines collapse to one plus a count
(`… 311 weitere \`Compiling …\`-Zeilen`). This typically shrinks a 5,000-line
wall to a few hundred unique lines the model ever sees.

**3. Line classification — the embedded model, windowed.** The collapsed
middle runs through the resident model in ~64-line windows with ~8-line
overlap; a small head reads the hidden state at each line-end token — one
prediction per line, ONE forward pass per window. Classes:

```
error / warning / question / artifact / progress / text
+ a continuation bit ("this line continues the previous one")
```

The continuation bit lifts BLOCKS: a rust error is five lines, a traceback
twenty — the engineer quotes the block, so the head must too. Display policy
is derived, not judged: error, question and artifact blocks are lifted verbatim
with line numbers; warnings are counted and sampled; progress stays collapsed;
text stays in the pack unless layer 4 surfaces it.

**Six classes, not seven — measured, 2026-07-29.** The original scheme split
the remainder into `result` (the answer) and `info` (context). A 4-round
double-labeling QA (M3 vs Kimi, ~470 lines per round) put the five specific
classes near 100% agreement and made result/info oscillate 99 → 98 → 46 → 48
with every wording change, in exact anti-phase with info: the boundary is a
judgment call, not a definition gap. Deriving it from behaviour instead
(lines reappearing in the agent's next turn) failed on the same data — 1 of 76
error lines, 160 of 186 walls empty — because agents paraphrase rather than
quote. So the tool never asks which prose line is "the answer": what the head
cannot label reliably, the mechanics carry. Full evidence:
`greppy-data-pipeline/docs/CLASS-SCHEME-DECISION.md`.

**4. Surprisal net — for what no class catches.** In the same pass, per-line
perplexity under the model; the most surprising unclassified lines are lifted
too. The classifier covers the known shapes, surprise covers the exotic tool
with random printf formats — importance relative to the wall itself, no tool
database required.

**Byte gate, always:** a lifted line is displayed only if it exists
byte-identically at its claimed line in the stored output. The model can
select, never invent. Failure modes are all graceful: a missed line costs
comfort (skeleton + expand remain complete), a wrong pick costs noise, an
invention is dropped before display. Daemon cold → skeleton alone.

## Streaming (cargo & friends write continuously)

The consumer is an agent, not a human at a terminal: it receives tool output
only when the command ends, so the delivery unit is the COMPLETED output and
the four layers run exactly once, at exit. There is no live display to feed —
but continuous writers still dictate the capture mechanics:

- **Drain and spool.** Both streams are drained continuously and spooled
  incrementally to the pack store — never buffered in RAM. An undrained pipe
  blocks the child at 64 KB and fakes a hang. The store has a size cap with
  head-and-tail retention and an explicit gap marker; silent truncation is
  exactly the harness behaviour this tool exists to end.
- **No TTY, by design.** Under pipes most tools switch to line output
  themselves. Residual `\r` rewrites count as rewrites of ONE line for
  skeleton and collapse (else a progress bar is "a thousand lines"); the
  stored bytes stay untouched — the byte gate refers to the store.
- **Kill/timeout is a first-class outcome.** A killed command (cargo watch, a
  hung test) delivers the full skeleton of the partial output plus the expand
  id, the signal stated as such, an unterminated last line marked. Everything
  accumulated until the kill is in the store — nothing is lost anymore.
- **Per-line timestamps, layer 1.** Streaming yields the time of every line
  for free; recorded as pack metadata. The line before the largest gap is
  lifted on timeout walls — "where it hung" is the first thing the agent
  needs.
- **Interleaving is approximate.** stdout and stderr are captured separately
  (stderr verbatim is a file-descriptor fact); relative order between the two
  streams is therefore approximate and stated as such, not faked.

Out of the box: model and head ship in the binary like everything else. No
telemetry, no runtime learning, no state — identical behaviour everywhere.

## Training annex

**Data stack:**

```
main tap     HF agent-trace datasets (345 at review time) — tool outputs in
             exactly the deployment distribution, PLUS the agent's next turn
supplement   GitHub Actions logs of public repos, conclusion:failure filtered
             for error-class balance
eval anchor  the breakage farm: clone top repos, injure them deliberately
             (delete a file, revert a commit, twist a version pin), build —
             authentic failure walls with KNOWN cause
```

No license screening (owner decision: technical tool output and LLM chat
content carry no copyright in the EU). Secrets scrubbing applies ONLY where
training is generative: a classifier head outputs six numbers and cannot
reproduce anything, so head training needs no scrub — but a next-token
fine-tune on walls feeds the SAME model that writes hint sentences, and only
there could a memorized key ever resurface. If surprisal works without a
fine-tune (measure first), the scrub is moot entirely.

**Pipeline and fleet roles:**

```
M3      sight the datasets: schema family, junk flag (bulk judgment)
Sol     one extractor per harness schema (Claude Code, Codex, Hermes, Pi) —
        completion-critical parser work, runs as code afterwards
script  mass extraction on gpu3: wall + following agent turn — zero tokens
script  behaviour labels for free: wall lines that reappear in the agent's
        next turn were acted upon
M3      6-class + continuation line labels, wall-by-wall, with the agent's
        continuation as context; "unclear" is a permitted answer
Kimi    5% double-label sample; disagreement adjudicated or dropped; an
        agreement rate under ~95% means a class definition is mushy and gets
        sharpened BEFORE the mass run (200-wall pilot first)
944k    head training on the existing distillation infrastructure; training
pipeline samples are windows with per-line labels — train what runs, no
        distribution gap between training and inference
```

**Evaluation:** held-out walls from tools absent from the training corpus
(the generalization number), plus the breakage farm as ground truth — the
lifted block must name the injected cause. The surprisal net needs no labels
at all; its training is plain next-token loss on walls, and possibly the
existing model suffices unmodified — measure before training.

**Sequencing (owner decision, revised):** 0.3.0 never waits for bash-smart —
in either direction. The training-free v1 (skeleton, collapse, expand,
embedding-novelty lift, byte gate) is built as the fifth work order behind
EDIT; whatever is fully accepted by the time the REST of 0.3.0 is production
ready, ships with it — the rest ships 0.4.0. The trained classifier head is
0.4.0 regardless: it hangs on the data pipeline. The prompt line enters
AGENTS.md only when the shipped binary holds it.
e2b therefore benches 0.3.0 without bash-smart; the ~40 shell-fallback burst
turns of e2a stay visible in e2b and become the before-number for the 0.4.0
measurement. The data work (M3 sighting, Sol extractors) may start any time —
it touches no 0.3.0 code.
