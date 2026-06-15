"""Persistence: a small JSON document plus an append-only completion log.

Concurrency model
-----------------
The whole bot runs in a single asyncio event loop, so there is no OS-thread
parallelism to guard against. The only hazards are:

  1. Two coroutines interleaving a read-modify-write across an ``await`` point.
  2. A crash midway through writing the file, leaving it truncated/corrupt.

We handle (1) with an ``asyncio.Lock`` held across each transaction, and (2)
by always writing to a temp file, ``fsync``-ing it, and ``os.replace``-ing it
over the real file (an atomic rename on POSIX). The in-memory ``self.data`` is
the source of truth during a run; it is flushed to disk after every change.

Transactions deliberately contain **no** ``await`` of network I/O — callers do
Discord work outside the lock and only re-enter a short transaction to commit
the resulting state. That keeps the lock held for microseconds.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import pathlib
import tempfile
from typing import Any

EMPTY: dict[str, Any] = {"configs": {}, "tasks": {}, "messages": {}}


class Store:
    def __init__(self, path: pathlib.Path, log_path: pathlib.Path) -> None:
        self.path = path
        self.log_path = log_path
        self._lock = asyncio.Lock()
        self.data: dict[str, Any] = json.loads(json.dumps(EMPTY))

    # -- loading ------------------------------------------------------------
    def load(self) -> None:
        """Read the store from disk (call once at startup, before the loop)."""
        if self.path.exists():
            self.data = json.loads(self.path.read_text(encoding="utf-8"))
        for key, default in EMPTY.items():
            self.data.setdefault(key, json.loads(json.dumps(default)))

    # -- atomic write -------------------------------------------------------
    def _flush(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=str(self.path.parent), suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(self.data, f, indent=2, ensure_ascii=False)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, self.path)  # atomic on POSIX
        except BaseException:
            with contextlib.suppress(FileNotFoundError):
                os.unlink(tmp)
            raise

    # -- transactions -------------------------------------------------------
    @contextlib.asynccontextmanager
    async def txn(self):
        """Mutate ``self.data`` under the lock; flush atomically on clean exit.

        Keep the body free of network ``await``s. On exception the change is
        not flushed (so a failed handler can't half-write the store)."""
        async with self._lock:
            yield self.data
            self._flush()

    async def snapshot(self) -> dict[str, Any]:
        """A deep copy that callers can read freely without holding the lock."""
        async with self._lock:
            return json.loads(json.dumps(self.data))

    # -- completion log (append-only JSONL) ---------------------------------
    async def log_completion(self, record: dict[str, Any]) -> None:
        line = json.dumps(record, ensure_ascii=False)
        async with self._lock:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
                f.flush()
                os.fsync(f.fileno())

    def read_completions(self) -> list[dict[str, Any]]:
        """Read every logged completion (tolerating a torn final line)."""
        out: list[dict[str, Any]] = []
        if not self.log_path.exists():
            return out
        for line in self.log_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return out
