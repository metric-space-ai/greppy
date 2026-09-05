# Basic browser study fixture

This fixture provides four basic synthetic cases and an intermediate inventory workflow. A run contains exactly one participant task, so measurements can compare like with like:

* `text`: replace the existing note with **Ready for review** and save it.
* `checkbox`: enable the checkbox and change the now-enabled quantity to **3**.
* `address`: choose **Germany**, **Berlin**, and postcode **10115**, then wait for visible validation.
* `dialog`: open **Complete basic task** and click the Save button inside that dialog; another Save button is visible outside it.
* `table`: filter to **EU** and **at least 3 available**, sort unit price **low to high**, then reserve **3** of the cheapest matching item in its confirmation dialog. The reserved row leaves the active filter; the reservation remains visible after reload.

Create a run with deterministic facts and start the local server. State is outside the repository; set `GREPPY_BASIC_FIXTURE_RUN_DIR` or pass `--run-dir` explicitly.

```sh
python3 server.py create-run --case text --seed pair-1 --run-dir /Volumes/tmp/dev-artifacts/greppy/web-study/basic-runs
python3 server.py serve --port 8765 --run-dir /Volumes/tmp/dev-artifacts/greppy/web-study/basic-runs
```

The command prints the 12-character run ID. Open `http://127.0.0.1:8765/?run_id=RUN_ID`. With `--port 0`, `serve` prints the assigned URL after binding and flushes it immediately. Verify from the host:

```sh
python3 server.py verify-run RUN_ID --run-dir /Volumes/tmp/dev-artifacts/greppy/web-study/basic-runs
```

The oracle checks the exact expected state for the selected case and exits nonzero for incomplete or false states. Every accepted mutation increments a revision and appends an event to the JSON state. Runs reject path traversal and invalid action types; the server listens only on `127.0.0.1`. There are no external services, Node dependencies, solution links, or task-reading side effects.

The table case is opt-in (`prepare_series.py --cases table`); the existing four-case default is unchanged. Prices and row order vary deterministically by seed, with a unique cheapest eligible item. The oracle independently checks filter/sort settings, exactly one correct reservation, total price and stock effects. Missing filters, a valid reservation of the wrong item, or duplicate reservations cannot pass. This is a development fixture, not an Office replacement or held-out acceptance case. Freeze the added `table_case.py` and `static/table.js` dependencies with each new series; previously frozen cases and recorded failures stay unchanged.

New plans include a versioned, arm-independent `task_goal` with the exact synthetic business objective. Before spawning each participant, save the complete intended prompt with `python3 dispatch.py SERIES POSITION --task-name NAME --message-file PROMPT.txt`, then pass its exact `message` to the Luna/medium participant. The prompt must contain the planned goal and trial URL; tool instructions belong in the full prompt, not the shared business objective. Record any revised condition as a new series.

The create-only `prepared-dispatches/NN.json` pins the goal, full message and plan hashes. It explicitly says `prepared_not_sent`: retain the actual spawn receipt separately and never treat this file as proof of delivery or of unchanged transport. Do not backfill historical goals from oracles or encrypted prompts. These records supply prospective task context; they do not label an observation as relevant, prove action causality or authorize model training. They contain only synthetic study content; credentials and private reasoning must not be included.

For new C conditions, the full participant prompt must preserve host command telemetry: use `text(await tools.exec_command({cmd: COMMAND}))`, not output-only forwarding; if it returns a session handle, poll that handle to its terminal exit code before proceeding. Quote shell arguments normally. `web status` is not a replacement for a host-process completion result. Declare this as a transport correction against Series 04 and test it with the unchanged browser candidate before attributing any difference to E1. The probe and exact historical call references are in `docs/web-response-cost-findings-20260905.md`.

To isolate onboarding costs, preregister a new plan with `prepare_series.py --onboarding explicit_transport_v1` and freeze every prompt before the first participant with `python3 onboarding.py SERIES --name-prefix s06`. Both arms receive a concrete supported opening call, including a retained CUA tab binding with `visible:false` and complete Greppy shell-result forwarding. The template contains no task-specific selectors or interaction sequence. It refuses an unregistered condition or existing dispatch directory. Use the unchanged previous candidate for this ablation; new engine code is a separate condition. Prepared prompts still require separately recorded actual spawn receipts.

For the currently installed Browser plugin, preregister `--onboarding browser_plugin_transport_v2`.
This uses `mcp__node_repl__js` and the installed Browser skill instead of the historical
`mcp__cua_repl` interface. Greppy's transport instructions stay unchanged. Freeze
all prompts and hash the installed skill/client before dispatch; include their
normal setup and documentation costs in the actual provider totals. This changes
the standard-tool condition and must not be represented as the same historical
baseline or as an isolated Greppy fix ablation. Old v1 plans and prompts stay intact.

After the S07 isolation finding, newly prepared contexts use a distinct runtime
owner for every participant (`per_trial_owner_v1`). Empty working directories
alone do not prevent `session list` and recovery from reaching another trial in
the same owner. Existing aliases/plans are never rewritten. New runtime owners
change startup conditions; report them separately from shared warm-runtime runs.
The summary refuses a token-pass verdict when session evidence is missing or
reused, even if all numeric token comparisons and binary hashes would pass.
This conservative ID check is not a full browser-profile isolation proof.

Use `--onboarding browser_plugin_synthetic_v3` for the next synthetic development
series. Both arms receive the same explicit fact that reservations are local
test records, with no order, payment or contract. The business goal and allowed
UI APIs remain the same. V1/V2 prompts and their recorded outcomes remain intact;
this is a declared new harness condition, not a retrospective repair of S07.
