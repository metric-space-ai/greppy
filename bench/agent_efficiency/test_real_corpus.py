import json
import os
import pathlib
import re
import subprocess
import sys
import tempfile
import unittest

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
    "recall_audit": str(recall_audit.REALCORPUS),
    "summary_quality": str(summary_quality.REALCORPUS),
    "tracked_candidates": str(gen_real_tasks.CANDIDATES),
}))
"""
        with tempfile.TemporaryDirectory() as directory:
            external = pathlib.Path(directory).resolve()
            env = dict(os.environ, GREPPY_BENCH_CORPUS_HOME=str(external))
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


if __name__ == "__main__":
    unittest.main()
