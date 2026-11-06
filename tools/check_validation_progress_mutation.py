#!/usr/bin/env python3
"""Exercise actual candidate validation and journal liveness; mutate production sources."""
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
FILTER = "validation_progress"
WALK = "embedding::tests::validation_progress_counts_completed_candidates_before_exact_spans"
JOURNAL = "background_progress_tests::validation_progress_publishes_candidates_without_heartbeat_work"
AGE = "indexing::progress_status_tests::validation_progress_work_age_is_independent_of_liveness"
ETA = "query_progress::tests::validation_progress_candidates_stay_distinct_from_spans_without_eta_or_poll_spam"


def run(package, tests, red=None):
    args = ["cargo", "test", "--locked", "--jobs", "2", "-p", package, "--lib"]
    if package == "greppy":
        args += ["--features", "ci-test-assets"]
    env = os.environ.copy()
    env["RUST_MIN_STACK"] = "8388608"
    result = subprocess.run(
        [*args, FILTER, "--", "--test-threads=1"], cwd=ROOT, env=env,
        text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=900,
    )
    if red:
        test, cause = red
        if not (result.returncode == 101 and test + " ... FAILED" in result.stdout
                and "test result: FAILED" in result.stdout and cause in result.stdout):
            raise AssertionError("Mutation did not fail for the intended contract:\n" + result.stdout)
    elif not (result.returncode == 0 and all(test + " ... ok" in result.stdout for test in tests)):
        raise AssertionError("Contract did not execute successfully:\n" + result.stdout)


def mutation(path, before, after, package, test, cause, label):
    original = path.read_text()
    if original.count(before) != 1:
        raise AssertionError("Mutation anchor must be unique: " + label)
    try:
        path.write_text(original.replace(before, after, 1))
        run(package, [], red=(test, cause))
        print("mutation: " + label + " violates the real contract (exit101)", flush=True)
    finally:
        path.write_text(original)


run("greppy-indexer", [WALK])
run("greppy", [JOURNAL, AGE, ETA])
print("mutation: baseline candidate walk, journal, work age and coalesced reporter contracts green", flush=True)
mutation(
    ROOT / "crates/indexer/src/embedding.rs",
    "            candidate_completed(&node);\n        }\n    }\n    Ok(work)",
    "            // Mutation: leave normal candidate completion silent.\n        }\n    }\n    Ok(work)",
    "greppy-indexer", WALK,
    "every scoped candidate must finish validation before the exact span total is published",
    "suppressing actual completed-candidate events",
)
mutation(
    ROOT / "crates/indexer/src/embedding.rs",
    "            let source = match cached_file_source(&mut file_cache, root, &node.file_path) {\n"
    "                Ok(Some(source)) => source,\n                Ok(None) | Err(_) => {",
    "            candidate_completed(&node); // Mutation: report work before validating it.\n"
    "            let source = match cached_file_source(&mut file_cache, root, &node.file_path) {\n"
    "                Ok(Some(source)) => source,\n                Ok(None) | Err(_) => {",
    "greppy-indexer", WALK, "candidate completion must follow actual token validation",
    "publishing a candidate before its real token validation",
)
mutation(
    ROOT / "crates/cli/src/lib.rs",
    "    publish_locked_background_job(path, &current)\n}",
    "    object.insert(\"last_work_progress_at_unix_secs\".into(), serde_json::json!(unix_now_secs_cli()));\n"
    "    publish_locked_background_job(path, &current)\n}",
    "greppy", JOURNAL, "liveness must not fabricate completed validation work",
    "counting a liveness heartbeat as completed work",
)
mutation(
    ROOT / "crates/cli/src/indexing.rs",
    'timestamp("last_work_progress_at_unix_secs")', "heartbeat",
    "greppy", AGE, "fresh heartbeat must not hide old validation work",
    "using heartbeat age as validation-work age",
)
mutation(
    ROOT / "crates/cli/src/query_progress.rs",
    'progress_changed && job.state != "counting_embeddings"', "progress_changed",
    "greppy", ETA, "candidate throughput must not guess an embedding ETA",
    "extrapolating embedding ETA from candidate throughput",
)
run("greppy-indexer", [WALK])
run("greppy", [JOURNAL, AGE, ETA])
print("mutation: all production sources restored; real contracts green", flush=True)
