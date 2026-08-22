import json
import os
import pathlib
import re
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

from bench.agent_efficiency import parallel_acceptance_run, real_corpus


class RealCorpusContractTests(unittest.TestCase):
    def test_every_repository_uses_a_full_commit_sha(self) -> None:
        for name, spec in real_corpus.REPOS.items():
            with self.subTest(repository=name):
                self.assertRegex(spec["commit"], re.compile(r"^[0-9a-f]{40}$"))

    def test_every_corpus_component_honors_the_shared_external_root(self) -> None:
        repository = pathlib.Path(__file__).resolve().parents[2]
        probe = """
import json
from bench import summary_quality
from bench.agent_efficiency import (
    context_cost,
    gen_corpus,
    gen_real_tasks,
    real_corpus,
    recall_audit,
    run_bench,
    verify_tasks,
)
print(json.dumps({
    "synthetic": str(gen_corpus.CORPUS),
    "real": str(real_corpus.ROOT),
    "runner_synthetic": str(run_bench.CORPUS),
    "runner_real": str(run_bench.REALCORPUS),
    "task_mirror_source": str(gen_real_tasks.REAL_REPO_ROOT),
    "context_cost": str(context_cost.CORPUS),
    "task_verifier": str(verify_tasks.CORPUS),
    "task_verifier_bin": str(verify_tasks.BIN),
    "recall_audit": str(recall_audit.REALCORPUS),
    "summary_quality": str(summary_quality.REALCORPUS),
    "tracked_candidates": str(gen_real_tasks.CANDIDATES),
}))
"""
        with tempfile.TemporaryDirectory() as directory:
            external = pathlib.Path(directory).resolve()
            candidate = external / "candidate-greppy"
            env = dict(
                os.environ,
                GREPPY_BENCH_CORPUS_HOME=str(external),
                GREPPY_BENCH_BIN=str(candidate),
            )
            completed = subprocess.run(
                [sys.executable, "-c", probe],
                cwd=repository,
                env=env,
                check=True,
                capture_output=True,
                text=True,
            )
            paths = json.loads(completed.stdout)

        self.assertEqual(paths["synthetic"], str(external / "corpus"))
        self.assertEqual(paths["real"], str(external / "realcorpus"))
        self.assertEqual(paths["runner_synthetic"], str(external / "corpus"))
        self.assertEqual(paths["runner_real"], str(external / "realcorpus"))
        self.assertEqual(paths["task_mirror_source"], str(external / "realcorpus"))
        self.assertEqual(paths["context_cost"], str(external / "corpus"))
        self.assertEqual(paths["task_verifier"], str(external / "corpus"))
        self.assertEqual(paths["task_verifier_bin"], str(candidate))
        self.assertEqual(paths["recall_audit"], str(external / "realcorpus"))
        self.assertEqual(paths["summary_quality"], str(external / "realcorpus"))
        self.assertEqual(
            paths["tracked_candidates"],
            str(repository / "bench/agent_efficiency/realcorpus/candidates.json"),
        )

    def test_run_manifest_hashes_the_manifest_from_the_shared_root(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            realcorpus = pathlib.Path(directory)
            manifest = realcorpus / "MANIFEST.json"
            manifest.write_text('{"repos": {}}\n', encoding="utf-8")
            record = parallel_acceptance_run.repository_manifest_record(realcorpus)

        self.assertEqual(record["path"], "realcorpus/MANIFEST.json")
        self.assertEqual(
            record["sha256"],
            "8702eb2099943bddf524714216050145eea44bca178d8db2772601bcdd8518c0",
        )

    def test_synthetic_selection_does_not_require_real_corpus_manifest(self) -> None:
        class CorpusLayout:
            REAL_REPOS = {"serde"}
            REALCORPUS = pathlib.Path("/missing-real-corpus")

        record = parallel_acceptance_run.selected_repository_manifest(
            [{"id": "r113", "repo": "python_large"}], CorpusLayout
        )

        self.assertIsNone(record)

    def test_real_selection_still_requires_real_corpus_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            realcorpus = pathlib.Path(directory)
            (realcorpus / "MANIFEST.json").write_text(
                '{"repos": {}}\n', encoding="utf-8"
            )

            class CorpusLayout:
                REAL_REPOS = {"serde"}
                REALCORPUS = realcorpus

            record = parallel_acceptance_run.selected_repository_manifest(
                [{"id": "r001", "repo": "serde"}], CorpusLayout
            )

        self.assertEqual(record["path"], "realcorpus/MANIFEST.json")

    def test_synthetic_verifier_failure_is_the_orchestrator_exit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = pathlib.Path(directory) / "run"
            argv = [
                "parallel_acceptance_run.py",
                "--skip-build",
                "--index-only",
                "--output-dir",
                str(output),
                "r113",
            ]
            with (
                mock.patch.object(sys, "argv", argv),
                mock.patch.dict(
                    os.environ,
                    {"GREPPY_STORE_DIR": str(pathlib.Path(directory) / "store")},
                ),
                mock.patch.object(parallel_acceptance_run, "ensure_api_key"),
                mock.patch.object(parallel_acceptance_run, "write_run_manifest"),
                mock.patch.object(
                    parallel_acceptance_run, "run_logged", return_value=7
                ),
                mock.patch.object(parallel_acceptance_run, "write_summary"),
                mock.patch.object(
                    parallel_acceptance_run, "index_task_repos"
                ) as index_repos,
            ):
                status = parallel_acceptance_run.main()

        self.assertEqual(status, 7)
        index_repos.assert_not_called()


if __name__ == "__main__":
    unittest.main()
