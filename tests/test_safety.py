"""Tests for file safety, repair, and edit operations."""

import tempfile
from pathlib import Path

from engram.models import Fact, FactCategory
from engram.store import FactStore


def _make_store() -> FactStore:
    tmp = Path(tempfile.mkdtemp())
    return FactStore(data_dir=tmp)


def test_repair_recovers_from_corrupt_lines():
    store = _make_store()
    store.append_facts(
        [
            Fact(id="good1", category=FactCategory.preference, content="Valid fact"),
        ]
    )
    # Append a corrupt line
    with store.facts_path.open("a") as f:
        f.write('{"broken json\n')
        f.write("not even json\n")

    result = store.repair()
    assert result["facts_valid"] == 1
    assert result["facts_corrupt"] == 2

    # Store should work normally after repair
    facts = store.load_facts()
    assert len(facts) == 1
    assert facts[0].id == "good1"


def test_repair_no_corruption():
    store = _make_store()
    store.append_facts(
        [
            Fact(category=FactCategory.preference, content="Clean fact"),
        ]
    )
    result = store.repair()
    assert result["facts_corrupt"] == 0
    assert result["facts_valid"] == 1


def test_repair_empty_store():
    store = _make_store()
    result = store.repair()
    assert result["facts_valid"] == 0
    assert result["facts_corrupt"] == 0


def test_edit_fact_preserves_id_and_timestamps():
    store = _make_store()
    fact = Fact(id="edit1", category=FactCategory.preference, content="Old content")
    store.append_facts([fact])
    original_created = fact.created_at

    updated = store.update_fact("edit1", content="New content", tags=["updated"])
    assert updated is not None
    assert updated.id == "edit1"
    assert updated.content == "New content"
    assert updated.tags == ["updated"]
    assert updated.created_at == original_created
    assert updated.updated_at > original_created


def test_recall_log_roundtrip():
    from engram.models import RecallRecord

    store = _make_store()
    record = RecallRecord(
        query="test query",
        tier=1,
        prefilter_count=10,
        latency_ms=150.5,
        quality="medium",
    )
    store.log_recall(record)

    loaded = store.load_recall_log()
    assert len(loaded) == 1
    assert loaded[0].query == "test query"
    assert loaded[0].tier == 1
    assert loaded[0].quality == "medium"


def test_load_facts_skips_corrupt_line():
    """A single corrupt line does not prevent other facts from loading."""
    store = _make_store()
    store.append_facts(
        [
            Fact(id="good1", category=FactCategory.preference, content="Valid fact 1"),
            Fact(id="good2", category=FactCategory.preference, content="Valid fact 2"),
        ]
    )
    # Inject a corrupt line between the two valid ones
    lines = store.facts_path.read_text().splitlines()
    lines.insert(1, '{"broken json')
    store.facts_path.write_text("\n".join(lines) + "\n")

    facts = store.load_facts()
    assert len(facts) == 2
    assert {f.id for f in facts} == {"good1", "good2"}


def test_load_facts_fully_corrupt_returns_empty():
    store = _make_store()
    store.facts_path.write_text("not json at all\nalso not json\n")
    assert store.load_facts() == []


def test_load_candidates_skips_corrupt_line():
    """A corrupt candidate line does not crash load_candidates."""
    from engram.models import MemoryCandidate

    store = _make_store()

    store.append_candidates(
        [
            MemoryCandidate(
                id="cand1",
                category=FactCategory.preference,
                content="Pending candidate",
            )
        ]
    )
    with store.candidates_path.open("a") as f:
        f.write('{"corrupt\n')

    candidates = store.load_candidates()
    assert len(candidates) == 1
    assert candidates[0].id == "cand1"


def test_append_facts_adds_newline_if_missing():
    """append_facts writes a leading newline if the file lacks a trailing one."""
    store = _make_store()
    # Manually write a valid JSON line without a trailing newline
    fact1 = Fact(id="f1", category=FactCategory.preference, content="Fact 1")
    store.facts_path.write_text(fact1.model_dump_json())  # no trailing \n

    fact2 = Fact(id="f2", category=FactCategory.preference, content="Fact 2")
    store.append_facts([fact2])

    loaded = store.load_facts()
    assert len(loaded) == 2
    assert {f.id for f in loaded} == {"f1", "f2"}


def test_append_facts_normal_file_no_extra_blank_line():
    """append_facts does not add extra blank lines when trailing newline exists."""
    store = _make_store()
    fact1 = Fact(id="f1", category=FactCategory.preference, content="Fact 1")
    store.append_facts([fact1])

    fact2 = Fact(id="f2", category=FactCategory.preference, content="Fact 2")
    store.append_facts([fact2])

    lines = [ln for ln in store.facts_path.read_text().splitlines() if ln.strip()]
    assert len(lines) == 2


def test_edit_fact_refuses_forgotten_fact():
    """update_fact returns None when fact is forgotten (confidence == 0.0)."""
    store = _make_store()
    fact = Fact(id="gone1", category=FactCategory.preference, content="Old pref")
    store.append_facts([fact])
    store.forget("gone1", reason="outdated")

    result = store.update_fact("gone1", content="New pref")
    assert result is None

    # Confirm fact is still forgotten and content unchanged
    all_facts = store.load_facts()
    stored = next(f for f in all_facts if f.id == "gone1")
    assert stored.confidence == 0.0
    assert stored.content == "Old pref"


def test_edit_fact_works_on_active_fact():
    """update_fact still works normally for active facts."""
    store = _make_store()
    fact = Fact(id="active1", category=FactCategory.preference, content="Original")
    store.append_facts([fact])

    updated = store.update_fact("active1", content="Updated")
    assert updated is not None
    assert updated.content == "Updated"


def test_atomic_rewrite_preserves_data():
    """Rewrite uses tmp+rename for atomicity."""
    store = _make_store()
    facts = [
        Fact(id=f"f{i}", category=FactCategory.preference, content=f"Fact {i}")
        for i in range(100)
    ]
    store.append_facts(facts)

    # Batch update all facts
    updates = {f"f{i}": {"content": f"Updated {i}"} for i in range(100)}
    store.batch_update_facts(updates)

    loaded = store.load_facts()
    assert len(loaded) == 100
    assert all("Updated" in f.content for f in loaded)
    # No .tmp file should remain
    assert not store.facts_path.with_suffix(".tmp").exists()
