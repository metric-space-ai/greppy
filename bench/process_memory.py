"""Sample owned Linux process families without double-counting shared RSS.

No processes/threads are launched or stopped here. Call sample() from an
existing bounded monitor. Register verified PID/starttime roots directly or
use the exclusive Linux child-subreaper scope to adopt daemonized descendants.
Very short-lived unobserved children can
be missed: sampled peaks are lower bounds, never OS-enforced memory ceilings.
"""

import ctypes
import os
from pathlib import Path
import secrets
import signal
import sys
import threading
import time


class ProcessFamilyCustody:
    """One existing monitor adopts and reaps only its new descendant family.

    The caller must launch no unrelated children until close(). No processes or
    threads are created here. Direct children cannot reuse their PID before this
    monitor reaps them. Sampling still cannot establish unsampled memory peaks.
    """

    _exclusive = threading.Lock()

    def __init__(self):
        if not self._exclusive.acquire(blocking=False):
            raise RuntimeError("concurrent process-family measurement is not supported")
        self.supported = False
        self.reason = "Linux subreaper custody is unavailable on this platform"
        self._previous = ctypes.c_int()
        self._libc = None
        self._closed = False
        if not sys.platform.startswith("linux"):
            return
        if threading.current_thread() is not threading.main_thread():
            self.reason = "process-family custody requires the monitor main thread"
            return
        if signal.getsignal(signal.SIGCHLD) != signal.SIG_DFL:
            self.reason = "process-family custody requires the default SIGCHLD disposition"
            return
        try:
            self._libc = ctypes.CDLL(None, use_errno=True)
            self._prctl(37, ctypes.byref(self._previous))  # PR_GET_CHILD_SUBREAPER
            if self._direct_children():
                self.reason = "process-family custody refuses pre-existing children"
                return
            self._prctl(36, ctypes.c_ulong(1))  # PR_SET_CHILD_SUBREAPER
            self.supported = True
            self.reason = None
        except (OSError, RuntimeError, AttributeError) as error:
            self.reason = str(error)

    def _prctl(self, option, value):
        if self._libc.prctl(ctypes.c_int(option), value, ctypes.c_ulong(0),
                            ctypes.c_ulong(0), ctypes.c_ulong(0)) != 0:
            raise OSError(ctypes.get_errno(), "cannot configure process-family custody")

    def _direct_children(self):
        pids = set()
        for task in Path("/proc/self/task").iterdir():
            try:
                pids.update(int(value) for value in (task / "children").read_text().split())
            except FileNotFoundError:
                continue  # An HTTP fixture thread ended during the observation.
        children = {}
        for pid in pids:
            identity = _stat(pid)
            if identity is None or identity[0] != os.getpid():
                raise RuntimeError("owned child identity changed before it could be registered")
            children[pid] = identity[1]
        return children

    def children(self):
        return self._direct_children() if self.supported else {}

    def close(self):
        if self._closed:
            return
        try:
            if self.supported:
                if self.children():
                    raise RuntimeError("owned process family still has unreaped children")
                self._prctl(36, ctypes.c_ulong(self._previous.value))
        finally:
            self._closed = True
            self._exclusive.release()


def _stat(pid):
    try:
        fields = (Path("/proc") / str(pid) / "stat").read_text().rsplit(")", 1)[1].split()
        return int(fields[1]), int(fields[19])  # PPID, starttime ticks
    except (OSError, ValueError, IndexError):
        return None


def _rollup(pid):
    values = {}
    for line in (Path("/proc") / str(pid) / "smaps_rollup").read_text().splitlines():
        parts = line.split()
        if len(parts) == 3 and parts[2] == "kB":
            values[parts[0].rstrip(":")] = int(parts[1]) * 1024
    for name in ("Rss", "Pss", "Private_Clean", "Private_Dirty"):
        if name not in values:
            raise ValueError("missing smaps_rollup field " + name)
    return values


class ProcessTreeMemorySampler:
    """Track PID+starttime identities, retaining known children after reparenting."""

    def __init__(self, root_pid):
        self.supported = sys.platform.startswith("linux") and Path("/proc").is_dir()
        self.identities = {}
        self.registration_failed = False
        self.has_observed_memory = False
        self.peak_pss_bytes = None
        self.peak_private_bytes = None
        self.samples = 0
        self.incomplete_samples = 0
        self.register_root(root_pid)

    def register_root(self, pid, expected_starttime=None):
        if not self.supported:
            return False
        if not isinstance(pid, int) or pid <= 0:
            raise ValueError("a positive owned process ID is required")
        identity = _stat(pid)
        if identity is None or (expected_starttime is not None and identity[1] != expected_starttime):
            self.registration_failed = True
            return False
        self.identities[pid] = identity[1]
        return True

    def sample(self):
        started = time.monotonic_ns()
        if not self.supported:
            return {"supported": False, "platform": sys.platform,
                    "pss_bytes": None, "private_bytes": None, "rss_sum_bytes": None}
        table = {}
        for directory in Path("/proc").iterdir():
            if directory.name.isdecimal():
                pid = int(directory.name)
                identity = _stat(pid)
                if identity is not None:
                    table[pid] = identity
        children = {}
        for pid, (parent, _) in table.items():
            children.setdefault(parent, []).append(pid)
        live = {pid for pid, start in self.identities.items()
                if pid in table and table[pid][1] == start}
        # Dead/reused PIDs are history, not resident monitoring state.
        self.identities = {pid: self.identities[pid] for pid in live}
        pending = list(live)
        while pending:
            for child in children.get(pending.pop(), []):
                if child not in live:
                    live.add(child)
                    self.identities[child] = table[child][1]
                    pending.append(child)
        pss_bytes = 0
        rss_bytes = 0
        private_bytes = 0
        swap_pss_bytes = 0
        measured = []
        unreadable = []
        for pid in sorted(live):
            try:
                metrics = _rollup(pid)
                if _stat(pid) != table[pid]:
                    raise ValueError("process identity changed during sample")
            except (OSError, ValueError):
                unreadable.append(pid)
                continue
            measured.append({"pid": pid, "starttime_ticks": table[pid][1]})
            pss_bytes += metrics["Pss"]
            rss_bytes += metrics["Rss"]
            private_bytes += metrics["Private_Clean"] + metrics["Private_Dirty"]
            swap_pss_bytes += metrics.get("SwapPss", 0)
        self.has_observed_memory |= bool(measured)
        complete = not unreadable and self.has_observed_memory and not self.registration_failed
        self.samples += 1
        self.incomplete_samples += not complete
        if complete:
            self.peak_pss_bytes = max(self.peak_pss_bytes or 0, pss_bytes)
            self.peak_private_bytes = max(self.peak_private_bytes or 0, private_bytes)
        return {
            "supported": True,
            "complete_for_observed_processes": complete,
            "sample_started_ns": started,
            "sample_finished_ns": time.monotonic_ns(),
            "processes": measured,
            "unreadable_pids": unreadable,
            "pss_bytes": pss_bytes if complete else None,
            "private_bytes": private_bytes if complete else None,
            "rss_sum_bytes": rss_bytes if complete else None,
            "swap_pss_bytes": swap_pss_bytes if complete else None,
            "peak_sampled_pss_bytes": self.peak_pss_bytes,
            "peak_sampled_private_bytes": self.peak_private_bytes,
            "scope": "observed process PSS; excludes kernel, GPU and unmapped filesystem cache",
        }


class OwnedKernelGroup:
    """Fresh benchmark-only cgroup below an explicitly delegated monitor.

    The caller must already run alone in the parent's 'monitor' leaf. Never
    elevates, moves a foreign process, changes ancestor limits or creates a
    service. The existing scheduler/delegation is the resource authority.
    See https://docs.kernel.org/admin-guide/cgroup-v2.html.
    """

    def __init__(self, parent, observation):
        self.parent = Path(parent).resolve()
        self.path = None
        self.identity = None
        self.enabled_memory = False
        self.closed = False
        ancestors = observation.get("visible_ancestors", [])
        mount = observation.get("mount_point")
        if (not sys.platform.startswith("linux") or not ancestors or not mount
                or observation.get("errors")):
            raise RuntimeError("kernel group requires an observed Linux cgroup-v2 delegation")
        self.monitor = Path(ancestors[0]["path"]).resolve()
        mount = Path(mount).resolve()
        if (self.parent == mount or not self.parent.is_relative_to(mount)
                or self.parent.stat().st_dev != mount.stat().st_dev
                or self.monitor.name != "monitor" or self.monitor.parent != self.parent):
            raise RuntimeError("kernel group requires the explicit parent of this worker's monitor leaf")
        if (self.parent / "cgroup.type").read_text().strip() != "domain":
            raise RuntimeError("kernel group requires a normal delegated domain")
        if ((self.parent / "cgroup.procs").read_text().split()
                or set(map(int, (self.monitor / "cgroup.procs").read_text().split())) != {os.getpid()}
                or self._children() != {self.monitor}):
            raise RuntimeError("kernel group refuses a shared or foreign-populated delegation")
        try:
            enabled = (self.parent / "cgroup.subtree_control").read_text().split()
            if "memory" not in enabled:
                (self.parent / "cgroup.subtree_control").write_text("+memory")
                self.enabled_memory = True
            self.path = self.parent / ("greppy-meter-" + secrets.token_hex(8))
            self.path.mkdir()
            metadata = self.path.stat()
            self.identity = metadata.st_dev, metadata.st_ino
            # Fresh counters, not a historic worker-wide peak or an FD reset
            # accidentally read through a different descriptor.
            if (int((self.path / "memory.peak").read_text()) != 0
                    or self._values("cpu.stat")["usage_usec"] != 0
                    or self._values("cgroup.events")["populated"] != 0):
                raise RuntimeError("new kernel group already contains resource history")
        except BaseException:
            self.close()
            raise

    def _children(self):
        return {path for path in self.parent.iterdir() if path.is_dir()}

    def _check_identity(self):
        if self.closed or self.path is None:
            raise RuntimeError("kernel group is not live")
        metadata = self.path.stat()
        if (metadata.st_dev, metadata.st_ino) != self.identity:
            raise RuntimeError("owned kernel group identity changed")

    def _values(self, name):
        return {key: int(value) for key, value in
                (line.split() for line in (self.path / name).read_text().splitlines())}

    def wrap_command(self, command):
        self._check_identity()
        if not command:
            raise ValueError("kernel group requires a command")
        # Fixed shell source, positional arguments only. The helper enters
        # before target exec; target descendants inherit membership. No Python
        # pre_exec callback runs in the multithreaded HTTP-fixture monitor.
        return ["/bin/sh", "-c",
                'set -eu; echo "$$" > "$1/cgroup.procs"; shift; exec "$@"',
                "greppy-owned-kernel-group", str(self.path), *command]

    def snapshot(self):
        self._check_identity()
        populated = self._values("cgroup.events")["populated"]
        if populated != 0:
            raise RuntimeError("owned kernel group still contains live descendants")
        cpu = self._values("cpu.stat")
        return {
            "schema": "greppy.kernel-resource-group.v1",
            "path": str(self.path), "device": self.identity[0], "inode": self.identity[1],
            "fresh_group": True, "populated": False,
            "target_entry_observed": cpu["usage_usec"] > 0,
            "memory_current_bytes": int((self.path / "memory.current").read_text()),
            "memory_peak_bytes": int((self.path / "memory.peak").read_text()),
            "cpu_usage_seconds": cpu["usage_usec"] / 1_000_000,
            "cpu_user_seconds": cpu["user_usec"] / 1_000_000,
            "cpu_system_seconds": cpu["system_usec"] / 1_000_000,
            "memory_events": self._values("memory.events"),
            "memory_max": (self.path / "memory.max").read_text().strip(),
            "scope": "fresh owned cgroup charges through target exit and cleanup; includes charged "
                     "kernel/cache and descendants, including auto-reaped CPU; shared/preexisting "
                     "pages may be charged elsewhere; not PSS or all physical RAM; helper work "
                     "before attachment and LLM/providers outside this group are excluded",
        }

    def close(self):
        if self.closed:
            return
        if self.path is not None:
            self._check_identity()
            if self._values("cgroup.events")["populated"] != 0:
                raise RuntimeError("refusing to remove a populated owned kernel group")
            self.path.rmdir()
            self.path = None
        if self.enabled_memory:
            if self._children() != {self.monitor}:
                raise RuntimeError("refusing to change a delegation with a new foreign subgroup")
            (self.parent / "cgroup.subtree_control").write_text("-memory")
            self.enabled_memory = False
        self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
