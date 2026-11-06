"""Runs one long job at a time (dataset build, LoRA training, DVC push) as a subprocess.

Why a subprocess with a log file and not a function call: training takes hours.
The UI only polls the log, so closing the browser tab does not kill the run, and
every run leaves outputs/jobs/<time>_<name>.log behind as a record.

One job at a time because Magi and kohya each want the whole 8 GB GPU.
"""
from __future__ import annotations

import os
import signal
import subprocess
import threading
from datetime import datetime
from pathlib import Path

from .config import ROOT


class Busy(RuntimeError):
    pass


class JobRunner:
    def __init__(self, root: Path = ROOT):
        self.root = Path(root)
        self.proc: subprocess.Popen | None = None
        self.name = ""
        self.log: Path | None = None
        self._lock = threading.Lock()
        self.before_start = None   # hook: the model service frees the GPU here

    # -------------------------------------------------------------- state
    def running(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def status(self) -> str:
        if self.proc is None:
            return "idle"
        code = self.proc.poll()
        if code is None:
            return f"running: {self.name}"
        return f"finished: {self.name} ({'ok' if code == 0 else f'FAILED, exit {code}'})"

    def tail(self, lines: int = 60) -> str:
        if not self.log or not self.log.exists():
            return ""
        # read only the end: a 12k-step kohya log grows to megabytes of progress bars
        with open(self.log, "rb") as f:
            f.seek(max(0, f.seek(0, 2) - 64_000))
            text = f.read().decode("utf-8", errors="replace")
        # kohya redraws its progress bar with \r; keep only the last state of each line
        rows = [r.split("\r")[-1] for r in text.splitlines()]
        return "\n".join(rows[-lines:])

    # ------------------------------------------------------------ control
    def start(self, name: str, cmd: list[str]) -> Path:
        with self._lock:
            if self.running():
                raise Busy(f"'{self.name}' is still running - stop it or wait")
            if self.before_start:
                self.before_start()
            d = self.root / "outputs" / "jobs"
            d.mkdir(parents=True, exist_ok=True)
            self.log = d / f"{datetime.now():%Y%m%d-%H%M%S}_{name}.log"
            fh = open(self.log, "w", encoding="utf-8")
            fh.write("$ " + " ".join(map(str, cmd)) + "\n\n")
            fh.flush()
            env = {**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8"}
            kw = ({"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP} if os.name == "nt"
                  else {"start_new_session": True})
            self.proc = subprocess.Popen(list(map(str, cmd)), cwd=self.root, stdout=fh,
                                         stderr=subprocess.STDOUT, env=env, **kw)
            fh.close()      # the child holds its own handle
            self.name = name
            return self.log

    def stop(self) -> None:
        """Kill the job and its children (accelerate spawns the real trainer)."""
        if not self.running():
            return
        if os.name == "nt":
            subprocess.run(["taskkill", "/T", "/F", "/PID", str(self.proc.pid)],
                           capture_output=True)
        else:
            os.killpg(self.proc.pid, signal.SIGTERM)
        self.proc.wait(timeout=30)
