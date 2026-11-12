import json
import logging
from pathlib import Path

import pytest
from docling_builders import DocBuilder
from docling_core.types.doc import DoclingDocument, SectionHeaderItem, TableItem

from app.retrieval.parsed import load_parsed


def _save(parsed_dir: Path, paper_id: str, name: str | None = None) -> DoclingDocument:
    doc = (
        DocBuilder(name=name or paper_id)
        .title("A Paper")
        .heading("2.1 Datasets", page=2)
        .text("Hello.", page=2)
        .table(["Dataset", "Size"], [["X", "10"]], caption="Table 1: Sizes.", page=3)
        .build()
    )
    doc.save_as_json(parsed_dir / f"{paper_id}.json")
    return doc


def test_round_trip_of_docling_json(tmp_path: Path) -> None:
    saved = _save(tmp_path, "p1")

    doc = load_parsed(tmp_path, "p1")

    assert isinstance(doc, DoclingDocument)
    assert doc.export_to_dict() == saved.export_to_dict()
    items = [item for item, _ in doc.iterate_items()]
    heading = next(i for i in items if isinstance(i, SectionHeaderItem))
    assert (heading.text, heading.prov[0].page_no) == ("2.1 Datasets", 2)
    table = next(i for i in items if isinstance(i, TableItem))
    assert table.caption_text(doc) == "Table 1: Sizes."


def test_missing_file_returns_none_without_warning(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.WARNING):
        assert load_parsed(tmp_path, "absent") is None
    assert caplog.records == []


def test_missing_directory_returns_none(tmp_path: Path) -> None:
    assert load_parsed(tmp_path / "does_not_exist", "p1") is None


@pytest.mark.parametrize(
    "content",
    [b"{not json", b"\xff\xfe\x00bad", b'["a", "list"]', b'{"name": 5}'],
    ids=["invalid-json", "non-utf8", "not-an-object", "not-a-docling-document"],
)
def test_unusable_file_returns_none_with_warning(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, content: bytes
) -> None:
    (tmp_path / "p1.json").write_bytes(content)

    with caplog.at_level(logging.WARNING):
        assert load_parsed(tmp_path, "p1") is None
    assert "p1.json" in caplog.text


def test_wrong_schema_field_returns_none_with_warning(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    _save(tmp_path, "p1")
    path = tmp_path / "p1.json"
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["texts"][0]["prov"][0]["page_no"] = "not a page"
    path.write_text(json.dumps(raw), encoding="utf-8")

    with caplog.at_level(logging.WARNING):
        assert load_parsed(tmp_path, "p1") is None
    assert "p1.json" in caplog.text


def test_document_name_mismatch_returns_none_with_warning(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    _save(tmp_path, "wrong_name", name="some_other_paper")

    with caplog.at_level(logging.WARNING):
        assert load_parsed(tmp_path, "wrong_name") is None
    assert "some_other_paper" in caplog.text


def test_empty_document_is_valid(tmp_path: Path) -> None:
    DoclingDocument(name="p1").save_as_json(tmp_path / "p1.json")

    doc = load_parsed(tmp_path, "p1")

    assert doc is not None
    assert list(doc.iterate_items()) == []


@pytest.mark.parametrize("paper_id", ["../p1", "sub/p1", "..", ""])
def test_path_traversal_paper_id_is_refused(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, paper_id: str
) -> None:
    parsed_dir = tmp_path / "parsed"
    parsed_dir.mkdir()
    _save(tmp_path, "p1")  # one level above parsed_dir

    with caplog.at_level(logging.WARNING):
        assert load_parsed(parsed_dir, paper_id) is None
    assert "invalid paper_id" in caplog.text
