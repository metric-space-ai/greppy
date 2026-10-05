"""Finite, authenticated Unix-socket witness for an existing heavy-job lease.

The admission runner remains the lock owner and bounds this service to the
admitted child's lifetime. Callers verify the server's kernel peer PID.
"""
import fcntl
import json
import os
import socket
import threading


class LeaseWitness:
    def __init__(self, lease, lock_path):
        self.lease = lease
        self.path = os.fspath(lock_path) + "." + str(os.getpid()) + ".sock"
        self.server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.stop = threading.Event()
        self.thread = None

    def __enter__(self):
        self.server.bind(self.path)
        os.chmod(self.path, 0o600)
        self.server.listen(64)
        self.server.settimeout(0.2)
        self.thread = threading.Thread(target=self._serve, daemon=True)
        self.thread.start()
        return self

    def _serve(self):
        while not self.stop.is_set():
            try:
                client, _ = self.server.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            with client:
                try:
                    client.settimeout(0.2)
                    data = b""
                    while b"\n" not in data and len(data) < 2048:
                        part = client.recv(2048 - len(data))
                        if not part:
                            break
                        data += part
                    request = json.loads(data)
                    challenge = request.get("challenge")
                    if not isinstance(challenge, str) or len(challenge) > 256:
                        continue
                    owns = False
                    try:
                        # The same locked open-file description succeeds;
                        # an unrelated holder makes this operation fail.
                        fcntl.flock(self.lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        owns = True
                    except BlockingIOError:
                        pass
                    stat = os.fstat(self.lease.fileno())
                    response = dict(challenge=challenge, owns_lease=owns,
                                    dev=stat.st_dev, ino=stat.st_ino)
                    client.sendall(json.dumps(response).encode() + b"\n")
                except (OSError, ValueError, TypeError):
                    continue

    def __exit__(self, *_):
        self.stop.set()
        self.server.close()
        if self.thread:
            self.thread.join(timeout=1)
        try:
            os.unlink(self.path)
        except FileNotFoundError:
            pass
