"""A graceful proxy restart must not cost warm conversations their prompt cache.

Production, 2026-10-19: five graceful restarts in one day (app relaunches for
updates) each made the next request of every active Claude Code session read
only the ~25k-token tools + system prefix and re-write the rest, 247k-432k
tokens at a time, about a third of that day's cache writes. After a restart
the proxy has no ``PrefixCacheTracker`` for the conversation, so it compresses
the whole history afresh and the early messages come out as different bytes
than the prefix it forwarded before the restart.

``HEADROOM_PREFIX_STATE_FILE`` (opt-in) saves the tracker store on graceful
shutdown and restores it at startup, so the first request after a restart
replays the exact prefix the provider cached.
"""

from __future__ import annotations

import contextvars
import json
import os
import stat
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("fastapi")

from headroom.cache import prefix_tracker
from headroom.cache.compression_store import reset_compression_store
from headroom.cache.prefix_tracker import PREFIX_STATE_FILE_ENV, SessionTrackerStore
from headroom.transforms.pipeline import TransformPipeline
from tests.test_anthropic_fork_lineage_cache import (
    _drive,
    _main,
    _PromptCache,
    _send,
    _strip,
    _summary_fork,
)


def _wire(messages: list[dict[str, Any]]) -> str:
    """The bytes the provider keys its prefix cache on (breakpoints excluded)."""
    return json.dumps(_strip(messages))


def _with_fake_kompress(monkeypatch: pytest.MonkeyPatch) -> None:
    """A deterministic stand-in for the Kompress ML stage (off in this suite).

    A cold start forwards the fast pass, which skips Kompress, while warm
    turns compress their new messages with it. So after a restart without
    state, the cold fast pass re-emits every message Kompressed after the
    first turn in a different form, which re-writes the cache from there."""
    real_apply = TransformPipeline.apply

    def apply(self, messages, model, **kwargs):  # noqa: ANN001, ANN003, ANN202
        skip = kwargs.get("skip_kompress", False)
        frozen = kwargs.get("frozen_message_count", 0) or 0
        result = real_apply(self, messages, model, **kwargs)
        if skip:
            return result
        out = []
        for i, message in enumerate(result.messages):
            content = message.get("content")
            if i >= frozen and isinstance(content, list):
                blocks = []
                for block in content:
                    text = block.get("content") if isinstance(block, dict) else None
                    if block.get("type") == "tool_result" and isinstance(text, str):
                        if not text.endswith(" [kompressed]"):
                            block = {**block, "content": text[: len(text) // 2] + " [kompressed]"}
                    blocks.append(block)
                message = {**message, "content": blocks}
            out.append(message)
        result.messages = out
        return result

    monkeypatch.setattr(TransformPipeline, "apply", apply)


def _read_turn(i: int) -> list[dict[str, Any]]:
    tid = f"toolu_r{i:02d}"
    body = "\n".join(f"{n:5d}  line {n} of file {i}: some code here" for n in range(300))
    return [
        {
            "role": "assistant",
            "content": [
                {"type": "tool_use", "id": tid, "name": "Read", "input": {"file_path": f"/f{i}.py"}}
            ],
        },
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": tid, "content": body}]},
    ]


def _reads(turns: int) -> list[dict[str, Any]]:
    """One Read per turn; Claude Code's breakpoint rides the newest block."""
    messages: list[dict[str, Any]] = [
        {"role": "user", "content": [{"type": "text", "text": "Review the change."}]}
    ]
    for i in range(turns):
        messages += _read_turn(i)
    messages[-1]["content"][-1]["cache_control"] = {"type": "ephemeral"}
    return messages


# (history builder, turns before the restart, proxy config, fake Kompress)
_SCENARIOS = {
    # Turns 1-2, restart, turn 3: Kompressed messages come back fast-passed.
    "kompress": (_main, 2, {}, True),
    # Reads held while active then cached verbatim stay verbatim; a fresh
    # process with an empty CompressionCache freezes nothing and matures them.
    "read-maturation": (
        _reads,
        5,
        {
            "read_maturation": True,
            "read_maturation_quiesce_turns": 2,
            "read_maturation_min_size_bytes": 100,
        },
        False,
    ),
}


def _restart_between_turns(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, scenario: str
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
    """Some turns in one proxy, the next turn in a fresh one, one warm
    provider cache. Returns (last forwarded before, first forwarded after,
    its read hit)."""
    build, turns, config, kompress = _SCENARIOS[scenario]
    monkeypatch.setenv("HEADROOM_CCR_BACKEND", "memory")
    # As the desktop app runs it: a cold start forwards the fast pass.
    monkeypatch.setenv("HEADROOM_BACKGROUND_COMPRESSION", "1")
    monkeypatch.setenv("HEADROOM_BACKGROUND_COMPRESSION_MIN_TOKENS", "1")
    monkeypatch.chdir(tmp_path)
    if kompress:
        _with_fake_kompress(monkeypatch)
    cache = _PromptCache()
    reset_compression_store()
    before = _drive([build(n) for n in range(1, turns + 1)], cache, **config)
    reset_compression_store()  # nothing in process memory survives a restart
    (after,) = _drive([build(turns + 1)], cache, **config)
    return before[-1][0], after[0], after[1]


@pytest.mark.parametrize("scenario", list(_SCENARIOS))
def test_first_request_after_a_restart_replays_the_cached_prefix(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, scenario: str
) -> None:
    state = tmp_path / "prefix-state.json"
    monkeypatch.setenv(PREFIX_STATE_FILE_ENV, str(state))

    previous, current, hit = _restart_between_turns(monkeypatch, tmp_path, scenario)

    assert _wire(current[: len(previous)]) == _wire(previous)
    assert hit >= len(previous) - 1, f"read only {hit + 1}/{len(previous)} cached messages"


@pytest.mark.parametrize("scenario", list(_SCENARIOS))
def test_without_the_state_file_a_restart_rewrites_the_prefix(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, scenario: str
) -> None:
    """Default behavior is unchanged, and it is the bug the opt-in fixes, so
    the test above is not vacuous."""
    monkeypatch.delenv(PREFIX_STATE_FILE_ENV, raising=False)

    previous, current, hit = _restart_between_turns(monkeypatch, tmp_path, scenario)

    assert _wire(current[: len(previous)]) != _wire(previous)
    assert hit < len(previous) - 1
    assert list(tmp_path.iterdir()) == [], "nothing may be written without the opt-in"


def _two_conversations() -> SessionTrackerStore:
    store = SessionTrackerStore()
    contextvars.Context().run(_send, store, _strip(_main(3)))
    contextvars.Context().run(_send, store, [{"role": "user", "content": "unrelated"}])
    return store


def test_restored_lineages_resume_and_new_ones_get_fresh_keys(tmp_path: Path) -> None:
    """Store level: a fork answered last still leaves the main turn resumable,
    and a new conversation never lands on a restored lineage's key."""
    store = _two_conversations()
    base = _strip(_main(3))
    contextvars.Context().run(_send, store, _strip(_summary_fork(3)))
    path = str(tmp_path / "state.json")
    assert store.save_state(path) == 2

    restored = SessionTrackerStore()
    assert restored.load_state(path) == 2
    main = contextvars.Context().run(
        restored.resolve_tracker, "sid", "anthropic", messages=_strip(_main(4))
    )
    new = contextvars.Context().run(
        restored.resolve_tracker, "sid", "anthropic", messages=[{"role": "user", "content": "new"}]
    )

    assert main.get_frozen_message_count() == len(base)
    assert main.get_last_original_messages() == base
    assert restored.active_sessions == 3
    assert new is not main and new.get_frozen_message_count() == 0


def test_expired_trackers_are_not_restored(tmp_path: Path) -> None:
    path = tmp_path / "state.json"
    _two_conversations().save_state(str(path))
    data = json.loads(path.read_text())
    data["trackers"][0]["meta"]["last_activity"] -= 3600  # past the 600s retention
    path.write_text(json.dumps(data))

    assert SessionTrackerStore().load_state(str(path)) == 1


def _bad_ref(path: Path) -> None:
    data = json.loads(path.read_text())
    data["trackers"][0]["meta"]["state"][3] = [-1]
    path.write_text(json.dumps(data))


def _bad_maturation(path: Path) -> None:
    data = json.loads(path.read_text())
    data["trackers"][-1]["meta"]["maturation"] = {"matured": [1], "booked": [], "deltas": {}}
    path.write_text(json.dumps(data))


@pytest.mark.parametrize(
    "corrupt",
    [
        lambda p: p.write_text('{"version": 1, "trackers": [{"meta"'),
        lambda p: p.write_text('{"version": 99, "trackers": []}'),
        lambda p: p.write_text("[1, 2]"),
        _bad_ref,
        _bad_maturation,
    ],
    ids=["truncated", "version", "shape", "bad-ref", "bad-maturation"],
)
def test_a_bad_state_file_starts_empty(tmp_path: Path, caplog, corrupt) -> None:  # noqa: ANN001
    path = tmp_path / "state.json"
    _two_conversations().save_state(str(path))
    corrupt(path)

    store = SessionTrackerStore()
    assert store.load_state(str(path)) == 0
    assert store.active_sessions == 0
    assert "ignoring prefix state" in caplog.text


def test_state_file_is_private_and_removed_once_loaded(tmp_path: Path) -> None:
    path = tmp_path / "state.json"
    path.write_text("stale")
    path.chmod(0o644)

    assert _two_conversations().save_state(str(path)) == 2
    if os.name == "posix":
        assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert SessionTrackerStore().load_state(str(path)) == 2
    assert not path.exists()
    assert SessionTrackerStore().load_state(str(path)) == 0  # missing file: cold start


def test_save_keeps_the_newest_trackers_within_the_cap(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    store = _two_conversations()
    newest = max(store._trackers.values(), key=lambda t: t._last_activity)
    monkeypatch.setattr(prefix_tracker, "_PREFIX_STATE_MAX_TRACKERS", 1)
    path = str(tmp_path / "state.json")
    assert store.save_state(path) == 1

    restored = SessionTrackerStore()
    restored.load_state(path)
    (kept,) = restored._trackers.values()
    assert kept.get_last_original_messages() == newest.get_last_original_messages()

    monkeypatch.setattr(prefix_tracker, "_PREFIX_STATE_MAX_BYTES", 10)
    assert store.save_state(path) == 0


def test_stateless_mode_writes_nothing(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HEADROOM_STATELESS", "1")
    path = tmp_path / "state.json"
    assert _two_conversations().save_state(str(path)) == 0
    assert not path.exists()


def test_read_maturation_state_survives_a_restart(tmp_path: Path) -> None:
    """Matured Reads stay booked once: without this a restart books their
    removal again on every later request of the conversation."""
    from headroom.config import ReadMaturationConfig
    from headroom.transforms.read_maturation import MaturedRead, ReadMaturationManager

    store = _two_conversations()
    manager = ReadMaturationManager(ReadMaturationConfig(enabled=True))
    manager._matured["toolu_r00"] = MaturedRead(marker="[Read of /f0.py ...]", ccr_hash="abc")
    manager._booked.add("toolu_r00")
    manager._replay_token_deltas["toolu_r00"] = 640
    newest = max(store._trackers.values(), key=lambda t: t._last_activity)
    newest.read_maturation_manager = manager
    path = str(tmp_path / "state.json")
    store.save_state(path)

    restored = SessionTrackerStore()
    restored.load_state(path)
    states = [t.restored_maturation_state for t in restored._trackers.values()]
    assert all(t.restored_from_disk for t in restored._trackers.values())
    (state,) = [s for s in states if s is not None]
    adopted = ReadMaturationManager(ReadMaturationConfig(enabled=True))
    adopted.restore_state(state)

    assert adopted._matured == manager._matured
    assert adopted._booked == manager._booked
    assert adopted._replay_token_deltas == manager._replay_token_deltas
