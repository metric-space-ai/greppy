"""Real Linux process-family/PSS contract, including a shared file mapping."""
import json
import mmap
import os
from pathlib import Path
import select
import signal
import subprocess
import sys
import tempfile
import unittest

from process_memory import ProcessTreeMemorySampler


@unittest.skipUnless(sys.platform.startswith("linux"), "Linux /proc PSS contract")
class ProcessMemoryContract(unittest.TestCase):
    def test_private_family_and_shared_mapping_are_accounted_without_rss_double_count(self):
        child = r"""
import mmap, sys
with open(sys.argv[1], "r+b") as f:
    shared = mmap.mmap(f.fileno(), 32 * 1024 * 1024)
    private = bytearray(24 * 1024 * 1024)
    for i in range(0, len(private), 4096): private[i] = 1
    total = sum(shared[i] for i in range(0, len(shared), 4096))
    print("ready", flush=True)
    sys.stdin.read()
"""
        parent = r"""
import json, mmap, subprocess, sys
with open(sys.argv[1], "r+b") as f:
    shared = mmap.mmap(f.fileno(), 32 * 1024 * 1024)
    private = bytearray(24 * 1024 * 1024)
    for i in range(0, len(private), 4096): private[i] = 1
    total = sum(shared[i] for i in range(0, len(shared), 4096))
    child = subprocess.Popen([sys.executable, "-c", sys.argv[2], sys.argv[1]],
                             stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    try:
        assert child.stdout.readline() == b"ready\n"
        print(json.dumps({"child_pid": child.pid}), flush=True)
        sys.stdin.read()
    finally:
        child.communicate(timeout=5)
"""
        with tempfile.TemporaryDirectory(prefix="greppy-memory-contract-") as directory:
            path = Path(directory) / "shared.bin"
            with path.open("w+b") as stream:
                stream.truncate(32 * 1024 * 1024)
                with mmap.mmap(stream.fileno(), 32 * 1024 * 1024) as shared:
                    shared[:] = b"x" * len(shared)
            root = subprocess.Popen([sys.executable, "-c", parent, str(path), child],
                                    stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE, start_new_session=True)
            try:
                self.assertTrue(select.select([root.stdout], [], [], 15)[0], "fixture readiness timeout")
                line = root.stdout.readline()
                self.assertTrue(line, "fixture exited before readiness")
                child_pid = json.loads(line)["child_pid"]
                sampler = ProcessTreeMemorySampler(root.pid)
                measured = sampler.sample()
                self.assertTrue(measured["supported"])
                self.assertTrue(measured["complete_for_observed_processes"], measured)
                self.assertEqual({p["pid"] for p in measured["processes"]}, {root.pid, child_pid})
                self.assertGreaterEqual(measured["private_bytes"], 48 * 1024 * 1024)
                self.assertGreaterEqual(measured["pss_bytes"], 48 * 1024 * 1024)
                self.assertLess(measured["pss_bytes"], measured["rss_sum_bytes"] - 16 * 1024 * 1024)
                root.communicate(timeout=10)
                self.assertEqual(root.returncode, 0)
                self.assertEqual(sampler.sample()["processes"], [])
                print("PROCESS_MEMORY_CONTRACT_OK", json.dumps(measured), flush=True)
            finally:
                if root.poll() is None:
                    os.killpg(root.pid, signal.SIGTERM)
                    try:
                        root.communicate(timeout=5)
                    except subprocess.TimeoutExpired:
                        os.killpg(root.pid, signal.SIGKILL)
                        root.communicate(timeout=5)


if __name__ == "__main__":
    unittest.main()
