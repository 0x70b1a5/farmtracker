"""Lightweight smoke tests — run with: uv run python tests/smoke.py

No pytest dependency; just asserts. Covers the parts most likely to break:
importing the bot (validates all slash-command registrations), the DST-aware
recurrence math, and the store's transaction + atomic-write behavior.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import pathlib
import tempfile
from zoneinfo import ZoneInfo

# Importing the bot module runs every @bot.tree.command decorator and builds
# the FarmBot instance — a real smoke test of the command definitions.
import farmtracker.bot  # noqa: F401
from farmtracker import models as m
from farmtracker.store import Store

UTC = dt.timezone.utc


def test_emoji_key() -> None:
    assert m.emoji_key("ℹ️") == m.emoji_key("ℹ")
    assert m.emoji_key("✅") == "✅"


def test_time_parsing() -> None:
    assert m.parse_hhmm("8:00") == (8, 0)
    assert m.parse_hhmm("23:59") == (23, 59)
    assert m.normalise_hhmm("8:5".replace("5", "05")) == "08:05"
    for bad in ("24:00", "8:60", "abc", "8", "08-00"):
        try:
            m.parse_hhmm(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"{bad!r} should have failed")


def test_first_due() -> None:
    tz = ZoneInfo("Europe/Berlin")
    # 09:00 local "now"; a 08:00 task already passed -> tomorrow.
    now = dt.datetime(2026, 6, 14, 9, 0, tzinfo=tz).astimezone(UTC)
    due = m.compute_first_due(now, tz, "08:00").astimezone(tz)
    assert (due.date() - now.astimezone(tz).date()).days == 1
    assert (due.hour, due.minute) == (8, 0)
    # a 10:00 task is still ahead -> today.
    due2 = m.compute_first_due(now, tz, "10:00").astimezone(tz)
    assert due2.date() == now.astimezone(tz).date()
    assert (due2.hour, due2.minute) == (10, 0)


def test_roll_forward_skips_backlog() -> None:
    tz = ZoneInfo("Europe/Berlin")
    prev = dt.datetime(2026, 6, 10, 8, 0, tzinfo=tz).astimezone(UTC)  # due 5 days ago
    now = dt.datetime(2026, 6, 14, 9, 0, tzinfo=tz).astimezone(UTC)
    # daily: next slot strictly after now -> tomorrow 08:00, NOT a backlog of 5.
    nxt = m.roll_forward(prev, tz, "08:00", 1, now).astimezone(tz)
    assert nxt.date() == dt.date(2026, 6, 15) and (nxt.hour, nxt.minute) == (8, 0)
    # every 2 days, anchored on the 10th -> 10,12,14,16; next after the 14th 09:00 is the 16th.
    nxt2 = m.roll_forward(prev, tz, "08:00", 2, now).astimezone(tz)
    assert nxt2.date() == dt.date(2026, 6, 16) and (nxt2.hour, nxt2.minute) == (8, 0)


def test_roll_forward_dst() -> None:
    # Spring-forward in Berlin is 2026-03-29 (02:00 -> 03:00). A daily 08:00 task
    # must stay at wall-clock 08:00 across the boundary, even though the UTC
    # offset changes from +01:00 to +02:00.
    tz = ZoneInfo("Europe/Berlin")
    prev = dt.datetime(2026, 3, 28, 8, 0, tzinfo=tz).astimezone(UTC)
    now = dt.datetime(2026, 3, 28, 8, 30, tzinfo=tz).astimezone(UTC)
    nxt = m.roll_forward(prev, tz, "08:00", 1, now).astimezone(tz)
    assert nxt.date() == dt.date(2026, 3, 29)
    assert (nxt.hour, nxt.minute) == (8, 0)
    assert prev.astimezone(tz).utcoffset() != nxt.utcoffset()  # offset really changed


def test_oneoff_parse() -> None:
    tz = ZoneInfo("Europe/Berlin")
    due = m.parse_oneoff("2026-06-20 14:00", tz)
    assert due == dt.datetime(2026, 6, 20, 14, 0, tzinfo=tz).astimezone(UTC)
    assert m.parse_oneoff("2026-06-20T14:00", tz) == due  # T separator allowed
    for bad in ("2026-13-01 10:00", "not a date", "2026-06-20"):
        try:
            m.parse_oneoff(bad, tz)
        except ValueError:
            pass
        else:
            raise AssertionError(f"{bad!r} should have failed")


async def test_store() -> None:
    with tempfile.TemporaryDirectory() as d:
        path = pathlib.Path(d) / "store.json"
        store = Store(path, pathlib.Path(d) / "log.jsonl")
        store.load()

        async with store.txn() as data:
            data["tasks"]["abc"] = {"brief": "feed", "guild_id": 1}
        assert path.exists(), "store file should be written atomically"

        snap = await store.snapshot()
        snap["tasks"]["abc"]["brief"] = "MUTATED"  # must not affect the store
        again = await store.snapshot()
        assert again["tasks"]["abc"]["brief"] == "feed", "snapshot must be a deep copy"

        # Exceptions inside a txn must NOT be flushed.
        try:
            async with store.txn() as data:
                data["tasks"]["abc"]["brief"] = "half-written"
                raise RuntimeError("boom")
        except RuntimeError:
            pass
        store2 = Store(path, pathlib.Path(d) / "log.jsonl")
        store2.load()
        assert store2.data["tasks"]["abc"]["brief"] == "feed", "rolled-back change leaked to disk"

        await store.log_completion({"user_id": 7, "ts": "2026-06-14T08:00:00+00:00"})
        await store.log_completion({"user_id": 7, "ts": "2026-06-14T09:00:00+00:00"})
        recs = store.read_completions()
        assert len(recs) == 2 and recs[0]["user_id"] == 7


def main() -> None:
    test_emoji_key()
    test_time_parsing()
    test_first_due()
    test_roll_forward_skips_backlog()
    test_roll_forward_dst()
    test_oneoff_parse()
    asyncio.run(test_store())
    print("✅ all smoke tests passed")


if __name__ == "__main__":
    main()
