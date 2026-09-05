# Basic browser study fixture

This fixture provides four independent, synthetic cases. A run contains exactly one participant task, so measurements can compare like with like:

* `text`: replace the existing note with **Ready for review** and save it.
* `checkbox`: enable the checkbox and change the now-enabled quantity to **3**.
* `address`: choose **Germany**, **Berlin**, and postcode **10115**, then wait for visible validation.
* `dialog`: open **Complete basic task** and click the Save button inside that dialog; another Save button is visible outside it.

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
