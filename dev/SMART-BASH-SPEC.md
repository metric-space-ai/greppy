# bash-smart — output specification and training annex (0.3.x)

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
error / warning / result / artifact / question / progress / info
+ a continuation bit ("this line continues the previous one")
```

The continuation bit lifts BLOCKS: a rust error is five lines, a traceback
twenty — the engineer quotes the block, so the head must too. Display policy
is derived, not judged: error, result, question and artifact blocks are lifted
verbatim with line numbers; warnings are counted and sampled; progress stays
collapsed; info stays in the pack.

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
training is generative: a classifier head outputs seven numbers and cannot
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
M3      7-class + continuation line labels, wall-by-wall, with the agent's
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

**Sequencing:** fourth work order, behind the EDIT family. Wired into the e2b
bench harness so the effect is measured where it was motivated: the ~40
shell-fallback burst turns in e2a.
