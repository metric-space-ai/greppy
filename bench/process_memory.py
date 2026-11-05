"""Sample owned Linux process families without double-counting shared RSS.

No processes/threads are launched or stopped here. Call sample() from an
existing bounded monitor. Explicitly register detached runtime PIDs obtained
from their supported owner receipt. Very short-lived unobserved children can
be missed: sampled peaks are lower bounds, never OS-enforced memory ceilings.
"""

from pathlib import Path
import sys
import time


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

    def register_root(self, pid):
        if not self.supported:
            return False
        if not isinstance(pid, int) or pid <= 0:
            raise ValueError("a positive owned process ID is required")
        identity = _stat(pid)
        if identity is None:
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
