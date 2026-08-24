"""Tests for src.common.tracker."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from src.common import tracker as tracker_mod
from src.common.tracker import (
    SCHEMA_VERSION,
    TRACKER_FILENAME,
    TrackerStep,
    _empty_tracker,
    get_or_assign_lr_model_id,
    get_or_assign_sr_model_id,
    load_tracker,
    mark_completed,
    record_error,
    record_warning,
    save_tracker,
    tracker_path,
    update_coverage,
)


@pytest.fixture(autouse=True)
def _isolated_processed_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect tracker IO at a tmp processed-bug dir.

    Patches ``get_processed_dir`` *inside the tracker module*, since that's the
    only reference the tracker uses.
    """
    bug_dir = tmp_path / "Proj" / "1"
    bug_dir.mkdir(parents=True)

    def fake_get_processed_dir(project: str, bug_id, *, dataset: str = "defects4j") -> Path:
        return bug_dir

    monkeypatch.setattr(tracker_mod, "get_processed_dir", fake_get_processed_dir)
    return bug_dir


# ---------------------------------------------------------------------------
# Empty / load / save
# ---------------------------------------------------------------------------


def test_empty_tracker_shape() -> None:
    t = _empty_tracker()
    assert t["schema_version"] == SCHEMA_VERSION
    assert t["extraction"] == {"completed": [], "warnings": {}, "errors": {}}
    assert t["fl"] == {"completed": [], "warnings": {}, "errors": {}}
    assert t["sr"] == {}
    assert t["lr"] == {}
    assert t["coverage"] == {}


def test_load_missing_returns_empty_when_create_true() -> None:
    t = load_tracker("Proj", "1")
    assert t == _empty_tracker()


def test_load_missing_raises_when_create_false() -> None:
    with pytest.raises(FileNotFoundError):
        load_tracker("Proj", "1", create=False)


def test_save_then_load_roundtrips(_isolated_processed_dir: Path) -> None:
    t = _empty_tracker()
    mark_completed(t, "extraction", "properties")
    save_tracker(t, "Proj", "1")
    assert (_isolated_processed_dir / TRACKER_FILENAME).exists()
    t2 = load_tracker("Proj", "1")
    assert t2["extraction"]["completed"] == ["properties"]


def test_save_is_atomic_no_tmp_left_behind(_isolated_processed_dir: Path) -> None:
    save_tracker(_empty_tracker(), "Proj", "1")
    files = sorted(p.name for p in _isolated_processed_dir.iterdir())
    assert files == [TRACKER_FILENAME]


def test_tracker_path_uses_get_processed_dir(_isolated_processed_dir: Path) -> None:
    assert tracker_path("Proj", "1") == _isolated_processed_dir / TRACKER_FILENAME


# ---------------------------------------------------------------------------
# mark_completed / record_warning / record_error idempotency
# ---------------------------------------------------------------------------


def test_mark_completed_is_idempotent() -> None:
    t = _empty_tracker()
    mark_completed(t, "extraction", "gzoltar")
    mark_completed(t, "extraction", "gzoltar")
    mark_completed(t, "fl", "ochiai")
    assert t["extraction"]["completed"] == ["gzoltar"]
    assert t["fl"]["completed"] == ["ochiai"]


def test_record_warning_appends_per_step() -> None:
    t = _empty_tracker()
    record_warning(t, "extraction", "faults", "missing source for X")
    record_warning(t, "extraction", "faults", "missing source for Y")
    record_warning(t, "extraction", "bug_report", "no URL")
    assert t["extraction"]["warnings"] == {
        "faults": ["missing source for X", "missing source for Y"],
        "bug_report": ["no URL"],
    }


def test_record_error_appends_per_step() -> None:
    t = _empty_tracker()
    record_error(t, "fl", "boostn", "RuntimeError: x")
    assert t["fl"]["errors"] == {"boostn": ["RuntimeError: x"]}


# ---------------------------------------------------------------------------
# get_or_assign_sr_model_id
# ---------------------------------------------------------------------------


def _common_kwargs(**overrides):  # type: ignore[no-untyped-def]
    base = {
        "model": "llama3.1:8b",
        "temperature": 0.0,
        "iterations": 10,
        "base_url": "http://localhost:11434",
        "input_keys": ["bug_report", "ochiai", "boostn", "sbir"],
    }
    base.update(overrides)
    return base


def test_assigns_default_slug_for_first_run() -> None:
    t = _empty_tracker()
    mid = get_or_assign_sr_model_id(t, **_common_kwargs())
    assert mid == "llama3.1_8b"
    assert "llama3.1_8b" in t["sr"]
    assert t["sr"]["llama3.1_8b"]["base_url"] == "http://localhost:11434"


def test_reuses_existing_id_for_matching_config() -> None:
    t = _empty_tracker()
    mid1 = get_or_assign_sr_model_id(t, **_common_kwargs())
    mid2 = get_or_assign_sr_model_id(t, **_common_kwargs())
    assert mid1 == mid2 == "llama3.1_8b"
    assert len(t["sr"]) == 1


def test_reuses_existing_id_when_only_base_url_differs() -> None:
    """base_url is informational only — it must NOT participate in identity."""
    t = _empty_tracker()
    mid1 = get_or_assign_sr_model_id(t, **_common_kwargs(base_url="http://a"))
    mid2 = get_or_assign_sr_model_id(t, **_common_kwargs(base_url="http://b"))
    assert mid1 == mid2
    assert len(t["sr"]) == 1


def test_assigns_suffix_when_temperature_differs() -> None:
    t = _empty_tracker()
    mid1 = get_or_assign_sr_model_id(t, **_common_kwargs(temperature=0.0))
    mid2 = get_or_assign_sr_model_id(t, **_common_kwargs(temperature=0.7))
    assert mid1 == "llama3.1_8b"
    assert mid2 == "llama3.1_8b__1"
    assert set(t["sr"]) == {"llama3.1_8b", "llama3.1_8b__1"}


def test_assigns_suffix_when_iterations_differ() -> None:
    t = _empty_tracker()
    mid1 = get_or_assign_sr_model_id(t, **_common_kwargs(iterations=10))
    mid2 = get_or_assign_sr_model_id(t, **_common_kwargs(iterations=20))
    assert mid1 == "llama3.1_8b"
    assert mid2 == "llama3.1_8b__1"


def test_assigns_suffix_when_input_set_differs() -> None:
    t = _empty_tracker()
    mid1 = get_or_assign_sr_model_id(t, **_common_kwargs(input_keys=["bug_report", "ochiai"]))
    mid2 = get_or_assign_sr_model_id(t, **_common_kwargs(input_keys=["bug_report"]))
    assert mid1 != mid2
    assert mid2 == "llama3.1_8b__1"


def test_input_order_does_not_matter_for_identity() -> None:
    t = _empty_tracker()
    mid1 = get_or_assign_sr_model_id(t, **_common_kwargs(input_keys=["a", "b", "c"]))
    mid2 = get_or_assign_sr_model_id(t, **_common_kwargs(input_keys=["c", "a", "b"]))
    assert mid1 == mid2


def test_suffix_skips_already_used_indices() -> None:
    t = _empty_tracker()
    get_or_assign_sr_model_id(t, **_common_kwargs(temperature=0.0))
    get_or_assign_sr_model_id(t, **_common_kwargs(temperature=0.5))
    get_or_assign_sr_model_id(t, **_common_kwargs(temperature=0.7))
    assert set(t["sr"]) == {"llama3.1_8b", "llama3.1_8b__1", "llama3.1_8b__2"}


# ---------------------------------------------------------------------------
# update_coverage
# ---------------------------------------------------------------------------


def test_update_coverage_replaces_subentries() -> None:
    t = _empty_tracker()
    update_coverage(t, s_faults={"count": 3, "ochiai": 2}, r_tests={"count": 12})
    assert t["coverage"]["s_faults"] == {"count": 3, "ochiai": 2}
    assert t["coverage"]["r_tests"] == {"count": 12}
    assert "m_faults" not in t["coverage"]

    update_coverage(t, s_faults={"count": 5})  # replaces, doesn't merge
    assert t["coverage"]["s_faults"] == {"count": 5}


def test_update_coverage_none_leaves_subentry_unchanged() -> None:
    t = _empty_tracker()
    update_coverage(t, s_faults={"count": 1})
    update_coverage(t, m_faults={"count": 2})
    assert t["coverage"]["s_faults"] == {"count": 1}
    assert t["coverage"]["m_faults"] == {"count": 2}


# ---------------------------------------------------------------------------
# TrackerStep
# ---------------------------------------------------------------------------


def test_step_marks_completed_on_clean_exit() -> None:
    with TrackerStep("Proj", "1", section="extraction", step="properties"):
        pass
    t = load_tracker("Proj", "1")
    assert "properties" in t["extraction"]["completed"]
    assert t["extraction"]["warnings"] == {}
    assert t["extraction"]["errors"] == {}


def test_step_does_not_mark_completed_on_exception() -> None:
    with (
        pytest.raises(RuntimeError, match="boom"),
        TrackerStep("Proj", "1", section="fl", step="boostn"),
    ):
        raise RuntimeError("boom")
    t = load_tracker("Proj", "1")
    assert t["fl"]["completed"] == []
    assert "boostn" in t["fl"]["errors"]
    assert any("boom" in m for m in t["fl"]["errors"]["boostn"])


def test_step_captures_log_warnings_from_named_logger() -> None:
    log = logging.getLogger("src.somewhere")
    with TrackerStep("Proj", "1", section="fl", step="ochiai"):
        log.warning("first warn")
        log.error("oops")
        log.info("ignored")  # below WARNING threshold
    t = load_tracker("Proj", "1")
    assert t["fl"]["warnings"]["ochiai"] == ["first warn"]
    assert t["fl"]["errors"]["ochiai"] == ["oops"]
    assert "ochiai" in t["fl"]["completed"]


def test_step_ignores_loggers_outside_logger_names() -> None:
    third_party = logging.getLogger("requests.session")
    with TrackerStep("Proj", "1", section="fl", step="boostn"):
        third_party.warning("third party noise")
    t = load_tracker("Proj", "1")
    assert "boostn" not in t["fl"]["warnings"]


def test_explicit_record_error_survives_swallowed_exception() -> None:
    with TrackerStep("Proj", "1", section="fl", step="sbir") as ts:
        try:
            raise ValueError("bad inputs")
        except ValueError as exc:  # caller swallows
            ts.record_error(repr(exc))
    t = load_tracker("Proj", "1")
    # Exception did not escape the with-block, so sbir should still be marked completed,
    # but the explicit error must be recorded.
    assert "sbir" in t["fl"]["completed"]
    assert any("bad inputs" in m for m in t["fl"]["errors"]["sbir"])


def test_step_handler_is_detached_after_exit() -> None:
    log = logging.getLogger("src.somewhere")
    with TrackerStep("Proj", "1", section="fl", step="ochiai"):
        pass
    log.warning("after-exit warn")  # must not be captured
    t = load_tracker("Proj", "1")
    assert t["fl"]["warnings"] == {}


def test_sr_step_records_into_model_bucket() -> None:
    t = _empty_tracker()
    mid = get_or_assign_sr_model_id(t, **_common_kwargs())
    save_tracker(t, "Proj", "1")

    log = logging.getLogger("src.agent4sr")
    with TrackerStep("Proj", "1", section="sr", step="run", model_id=mid):
        log.warning("slow tool call")

    t2 = load_tracker("Proj", "1")
    assert t2["sr"][mid]["warnings"] == ["slow tool call"]
    # SR section has no "completed" list — it's keyed presence only.
    assert "completed" not in t2["sr"][mid]


def test_sr_step_requires_model_id() -> None:
    with pytest.raises(ValueError, match="model_id"):
        TrackerStep("Proj", "1", section="sr", step="run")


def test_step_persists_tracker_to_disk_during_exit(_isolated_processed_dir: Path) -> None:
    with TrackerStep("Proj", "1", section="fl", step="ochiai"):
        # tracker.json should not exist yet — write happens on __exit__
        assert not (_isolated_processed_dir / TRACKER_FILENAME).exists()
    assert (_isolated_processed_dir / TRACKER_FILENAME).exists()


def test_step_preserves_pre_existing_sections(_isolated_processed_dir: Path) -> None:
    """A step writing to fl must not clobber the extraction section."""
    t = _empty_tracker()
    mark_completed(t, "extraction", "gzoltar")
    record_warning(t, "extraction", "bug_report", "no URL")
    save_tracker(t, "Proj", "1")

    with TrackerStep("Proj", "1", section="fl", step="ochiai"):
        pass

    t2 = load_tracker("Proj", "1")
    assert t2["extraction"]["completed"] == ["gzoltar"]
    assert t2["extraction"]["warnings"] == {"bug_report": ["no URL"]}
    assert "ochiai" in t2["fl"]["completed"]


def test_load_handles_missing_schema_version(_isolated_processed_dir: Path) -> None:
    """Forward-compatible load: legacy files without schema_version still parse."""
    raw = {
        "extraction": {"completed": ["properties"], "warnings": {}, "errors": {}},
        "fl": {"completed": [], "warnings": {}, "errors": {}},
        "sr": {},
        "coverage": {},
    }
    (_isolated_processed_dir / TRACKER_FILENAME).write_text(json.dumps(raw))
    t = load_tracker("Proj", "1")
    assert t["schema_version"] == SCHEMA_VERSION
    assert t["extraction"]["completed"] == ["properties"]
    assert t["lr"] == {}  # v1 → v2 migration added the empty lr bucket


def test_v1_to_v2_migration_adds_lr_bucket(_isolated_processed_dir: Path) -> None:
    """An existing v1 tracker.json must be loadable and get an empty lr bucket."""
    raw = {
        "schema_version": 1,
        "extraction": {"completed": [], "warnings": {}, "errors": {}},
        "fl": {"completed": [], "warnings": {}, "errors": {}},
        "sr": {
            "llama3.1_8b": {
                "model": "llama3.1:8b",
                "temperature": 0.0,
                "iterations": 10,
                "base_url": "",
                "input": ["bug_report"],
                "warnings": [],
                "errors": [],
            },
        },
        "coverage": {},
    }
    (_isolated_processed_dir / TRACKER_FILENAME).write_text(json.dumps(raw))
    t = load_tracker("Proj", "1")
    assert t["schema_version"] == 2
    assert t["lr"] == {}
    assert "llama3.1_8b" in t["sr"]


# ---------------------------------------------------------------------------
# LR model_id allocation
# ---------------------------------------------------------------------------


_PLANNER = {"role": "planner", "provider": "openai", "model": "gpt-4.1-mini"}
_TOOL = {
    "role": "tool_caller",
    "provider": "ollama",
    "model": "llama3.1:8b",
    "iterations": 10,
}
_FINISHER = {"role": "finisher", "provider": "openai", "model": "gpt-4.1-mini"}


def _lr_kwargs(**over):
    base = dict(
        config_name="my_config",
        agent_chain=[dict(_PLANNER), dict(_TOOL), dict(_FINISHER)],
        sr_model_id="llama3.1_8b",
        candidate_source="FlexFL/SR/rankings/top20/llama3.1_8b.txt",
        input_keys=["bug_report", "trigger_test"],
    )
    base.update(over)
    return base


def test_get_or_assign_lr_model_id_uses_config_name_slug() -> None:
    t = _empty_tracker()
    mid = get_or_assign_lr_model_id(t, **_lr_kwargs())
    assert mid == "my_config"
    assert t["lr"][mid]["sr_model_id"] == "llama3.1_8b"


def test_get_or_assign_lr_model_id_reuses_on_identical_identity() -> None:
    t = _empty_tracker()
    mid1 = get_or_assign_lr_model_id(t, **_lr_kwargs())
    mid2 = get_or_assign_lr_model_id(t, **_lr_kwargs())
    assert mid1 == mid2
    assert len(t["lr"]) == 1


def test_get_or_assign_lr_model_id_suffixes_when_chain_differs() -> None:
    t = _empty_tracker()
    mid1 = get_or_assign_lr_model_id(t, **_lr_kwargs())
    alt_tool = dict(_TOOL, model="llama3.1:70b")
    mid2 = get_or_assign_lr_model_id(
        t, **_lr_kwargs(agent_chain=[dict(_PLANNER), alt_tool, dict(_FINISHER)])
    )
    assert mid1 == "my_config"
    assert mid2 == "my_config__1"


def test_get_or_assign_lr_model_id_adhoc_uses_chain_hash() -> None:
    t = _empty_tracker()
    mid = get_or_assign_lr_model_id(t, **_lr_kwargs(config_name="<adhoc>"))
    # 12 hex chars
    assert len(mid) == 12
    assert all(c in "0123456789abcdef" for c in mid)


def test_get_or_assign_lr_model_id_input_order_does_not_matter() -> None:
    t = _empty_tracker()
    mid1 = get_or_assign_lr_model_id(t, **_lr_kwargs(input_keys=["bug_report", "trigger_test"]))
    mid2 = get_or_assign_lr_model_id(t, **_lr_kwargs(input_keys=["trigger_test", "bug_report"]))
    assert mid1 == mid2


def test_lr_step_records_into_bucket() -> None:
    t = _empty_tracker()
    mid = get_or_assign_lr_model_id(t, **_lr_kwargs())
    save_tracker(t, "Proj", "1")

    log = logging.getLogger("src.agent4lr")
    with TrackerStep("Proj", "1", section="lr", step="run", model_id=mid):
        log.warning("ranker complained")

    t2 = load_tracker("Proj", "1")
    assert t2["lr"][mid]["warnings"] == ["ranker complained"]


def test_lr_step_requires_model_id() -> None:
    with pytest.raises(ValueError, match="model_id"):
        TrackerStep("Proj", "1", section="lr", step="run")
