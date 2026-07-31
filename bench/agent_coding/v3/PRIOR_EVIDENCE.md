# What the previous coding-bench runs actually showed

Read before quoting any earlier number. Measured from
`/mnt/nvme1/bench030/gate-v4-run/results.json` on gpu3
(`run_id: gate-v4-n30-gpu3-20260721`, 30 tasks x 3 arms, the 0.2.1-era
harness), summed over all tasks:

| arm | solved | input tokens | uncached | cache read | tool calls | source opens |
|---|---|---:|---:|---:|---:|---:|
| explorer | 30/30 | 2,602,651 | 323,220 | 2,279,431 | 403 | 253 |
| greppy | 30/30 | 2,897,228 | 330,240 | 2,566,988 | 446 | 259 |
| greppy-edit | 30/30 | 5,179,803 | 403,368 | 4,776,435 | 621 | 391 |

Ratios against the explorer baseline: greppy **1.11x** input and **1.11x**
tool calls; greppy-edit **1.99x** input and **1.54x** tool calls.

Three conclusions, none comfortable:

1. **There is no prior positive coding-bench evidence for greppy on cost.**
   On this run greppy cost MORE than the uncoached baseline. Any release claim
   of lower cost cannot cite this harness.
2. **The 0.46x figure does not come from here.** It must be located in the
   navigation/discovery measurements before it is quoted anywhere; until then
   it is an unsourced number. Nothing in this repository records it — the only
   occurrences are in documents written on 2026-07-31 while looking for it.
3. **The run was structurally unfair in both directions,** which is why it
   cannot be rehabilitated by reinterpretation: `greppy-edit` had only `bash`
   while `explorer` had `bash,read,edit,write`, so a large part of its 1.99x
   is the missing native tools, not greppy; edit detection matched the literal
   word `edit` and therefore observed zero 0.3.0 edits; and the agent could
   read the solution out of the cloned upstream history.

A fourth observation matters for v3's design: **all three arms solved 30/30.**
At a 100% ceiling the task bank cannot differentiate capability at all, and
cost is the only remaining signal. That is precisely the argument for v3's
"medium-or-larger unfamiliar repositories, natural issues" corpus — not a
bigger version of the same easy bank.

Cache accounting is the other lesson in these numbers: 88% of the input
tokens are cache reads. Any cost comparison that does not price cache read
and cache write explicitly is measuring cache warmth, not tool efficiency.
The v3 plan's "Bruttokosten" line needs that pricing rule stated.
