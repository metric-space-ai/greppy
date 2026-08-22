import json
import pathlib
import tempfile
import unittest
from unittest import mock

import parallel_acceptance_run


def agent_result(*, error=None) -> dict:
    return {
        "answer": "done" if error is None else "",
        "error": error,
        "tool_calls": 1,
        "wall_s": 1.0,
    }


class InvalidProviderRecoveryTests(unittest.TestCase):
    def test_invalid_agent_detection_names_only_failed_arm(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            result_path = pathlib.Path(directory) / "r001.json"
            result_path.write_text(
                json.dumps(
                    [
                        {
                            "grep": agent_result(),
                            "greppy": agent_result(error="429 rate limit"),
                            "explorer": agent_result(),
                        }
                    ]
                ),
                encoding="utf-8",
            )
            self.assertEqual(
                parallel_acceptance_run.invalid_agents_in_worker_result(
                    result_path, ("grep", "greppy", "explorer")
                ),
                ("greppy",),
            )

    def test_parallel_run_recovers_isolated_invalid_session(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            worker_dir = root / "workers"
            raw_dir = root / "raw"
            logs_dir = root / "logs"
            worker_dir.mkdir()
            calls = []

            def fake_run_one_task(task, worker, raw, logs, rerun, agents,
                                  provider, tasks_path, log_suffix=""):
                calls.append((rerun, log_suffix))
                invalid = len(calls) == 1
                (worker / f"{task['id']}.json").write_text(
                    json.dumps(
                        [
                            {
                                "grep": agent_result(),
                                "greppy": agent_result(
                                    error="429 rate limit" if invalid else None
                                ),
                                "explorer": agent_result(),
                            }
                        ]
                    ),
                    encoding="utf-8",
                )
                return 0

            with mock.patch.object(
                parallel_acceptance_run, "run_one_task", side_effect=fake_run_one_task
            ):
                status = parallel_acceptance_run.run_parallel_bench(
                    [{"id": "r001"}],
                    1,
                    worker_dir,
                    raw_dir,
                    logs_dir,
                    False,
                )

            self.assertEqual(status, 0)
            self.assertEqual(calls, [(False, ""), (False, ".recovery-1")])


if __name__ == "__main__":
    unittest.main()
