# A prompt-stated call budget is not a budget

Measured 2026-08-02. This documents an approach that was built, tested and
**rejected**, so it is not tried again as if it were open.

## Why a budget was wanted

The 0.2.1 paper's headline number is a *frontier* comparison:

> to reach the lexical baseline's best correctness, Greppy costs 62% less on
> MiniMax-M3, 39% less on GLM-5.2, 80% less on Qwen3.6-27B, and 37% less on
> Kimi-K3

The mean of those four is 54.5% less — **0.455×**, the "0.46×" the release
criterion refers to. It is computed over call budgets 1/2/4/8/natural, and the
MiniMax-M3 figure comes from the **budget-1** point: Greppy answers 79.0% of
tasks correctly at one tool call for $1.20 per 1,000 tasks, against the lexical
agent's best natural-stop point of 70% for $3.13.

`bench/agent_efficiency` only ever measured natural stop. There, greppy spends
*more* because it answers more thoroughly: **+6.1 pp correctness at 0.93–1.00×
cost**. That is not the same quantity as 0.46× and must never be compared to it.

## What was built

`run_bench.py --budget N` appends an identical cap sentence to every arm's
system prompt at the single place all arms pass through, so the added bytes are
arm-identical by construction. `CALL_BUDGET` enters `prompt_contract()`, so a
budgeted run carries a different hash than the natural-stop run.

That part works and is kept. `pi` has no max-tool-call flag, so the prompt was
the only lever available.

## Why it does not work

Twelve real sessions at `--budget 1` (4 tasks × 3 arms, MiniMax-M3):

```
arm         tool calls per task      compliance   mean
explorer    2, 6, 1, 4               1/4          3.25
grep        6, 3, 1, 1               2/4          2.75
greppy      1, 2, 1, 3               2/4          1.75
```

The model ignores the cap in most sessions, **and it overruns by different
amounts per arm**. At "budget 1" a large part of greppy's apparent advantage
would be better cap *compliance*, not lower cost at equal budget. That produces
a number that looks like the paper's and measures something else — and it hits
precisely the budget-1 point the 62% figure comes from.

## Why partial enforcement is worse than none

The harness injects a run-local `bin` directory, so `grep`, `greppy`, `cat` and
friends could be wrapped in a counter that refuses after N calls. But pi's
`read` tool is **internal** and cannot be wrapped. The model would substitute
reads for shell calls, which biases the result toward whichever arm reads more —
a new, harness-created bias replacing the one being removed.

## Where that leaves the comparison

- The only valid 0.3.0 measurement is natural stop: **+6.1 pp correctness at
  0.93–1.00× cost** against the lexical baseline (`explorer`), 115 tasks,
  MiniMax-M3, single repetition.
- **0.46× cannot be reproduced with this driver.** Doing so needs an agent
  driver with a hard, complete tool-call limit — infrastructure work, not
  greppy work.
- Until then, no claim of the form "0.3.0 costs X× the baseline" may be made at
  a fixed budget, and the natural-stop numbers may not be presented as if they
  were the paper's frontier.

## Also worth keeping

A non-interactive shell on gpu3 silently produces a run in which every arm has
`return_code 1`, zero tokens and zero tool calls — which a compliance check
reads as perfect compliance. `pi` needs Node 22. See `bench/RUNBOOK.md`. Always
check return codes and token counts before computing any ratio.
