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
