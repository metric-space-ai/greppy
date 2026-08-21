#!/usr/bin/env python3
"""Mac spot tests: realistic, spectrum-wide discovery prompts, both arms.

Serial (ONE MiniMax session at a time — gpu3 owns the 6-session budget).
Uses run_bench.run_pi verbatim so the arms match the benchmark exactly.
"""
import json
import os
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
os.chdir(HERE)
# API key: env, else ~/.minimax.key. The binary always carries
# EmbeddingGemma, so context self-builds vectors without model configuration.
if not os.environ.get("MINIMAX_API_KEY"):
    keyfile = pathlib.Path.home() / ".minimax.key"
    if keyfile.exists():
        os.environ["MINIMAX_API_KEY"] = keyfile.read_text().strip()

import run_bench  # noqa: E402

OUT = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else pathlib.Path("/tmp/spot")
OUT.mkdir(parents=True, exist_ok=True)

# Realistic prompts a user would actually type — spectrum-wide:
# semantic (2), flow/how (2), natural graph (2).
PROBES = [
    ("flask_sem", "flask",
     "where does flask decide the order in which configuration sources override each other?"),
    ("zod_sem", "zod",
     "where does zod implement coercion for numbers?"),
    ("gson_flow", "gson",
     "how does gson decide whether to use a custom TypeAdapter or plain reflection for a class?"),
    ("django_flow", "django",
     "when django boots, where does it load and validate the settings module?"),
    ("serde_graph", "serde",
     "I'm about to rename wrap_in_const — anything that would break?"),
    ("tokio_graph", "tokio",
     "does anything outside of tests actually call interval_at?"),
]

results = []
for name, repo, q in PROBES:
    root = str(run_bench.REALCORPUS / repo)
    row = {"name": name, "repo": repo, "q": q}
    for arm, sysprompt in (("grep", run_bench.GREP_SYS),
                           ("greppy", run_bench.gp_sys(root))):
        print(f"== {name} [{arm}] ...", flush=True)
        r = run_bench.run_pi(
            sysprompt, q, cwd=root, timeout=300,
            raw_path=OUT / f"{name}.{arm}.jsonl",
        )
        row[arm] = {k: r.get(k) for k in
                    ("total", "variable_input", "output", "tool_calls",
                     "turns", "wall_s", "error")}
        row[arm]["answer"] = (r.get("answer") or "")[:1500]
        print(f"   tokens={r.get('total')} calls={r.get('tool_calls')} "
              f"wall={r.get('wall_s')}s err={r.get('error')}", flush=True)
    results.append(row)
    (OUT / "spot_results.json").write_text(json.dumps(results, indent=1))

print("\n=== SUMMARY (grep vs greppy) ===")
for row in results:
    g, p = row.get("grep", {}), row.get("greppy", {})
    if g.get("total") and p.get("total"):
        print(f"{row['name']:12s} total {g['total']:>7}/{p['total']:>7} "
              f"= {g['total']/p['total']:.2f}x  calls {g['tool_calls']}/{p['tool_calls']}  "
              f"wall {g['wall_s']}/{p['wall_s']}s")
print(f"\nraw transcripts + answers: {OUT}")
