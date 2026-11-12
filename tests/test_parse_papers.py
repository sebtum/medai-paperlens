import logging
from pathlib import Path

import pytest
from docling_builders import DocBuilder
from docling_core.types.doc import DoclingDocument

from app.retrieval.parsed import load_parsed
from scripts import parse_papers
from scripts.parse_papers import parse_papers as run_parse


def _pdfs(pdf_dir: Path, *paper_ids: str) -> None:
    pdf_dir.mkdir(parents=True, exist_ok=True)
    for paper_id in paper_ids:
        (pdf_dir / f"{paper_id}.pdf").write_bytes(b"%PDF-1.7 fake")


class FakeConverter:
    """Stands in for Docling; records calls and names the document after the PDF."""

    def __init__(self, fail_on: set[str] | None = None) -> None:
        self.calls: list[str] = []
        self.fail_on = fail_on or set()

    def __call__(self, pdf_path: Path) -> DoclingDocument:
        self.calls.append(pdf_path.stem)
        if pdf_path.stem in self.fail_on:
            raise RuntimeError("boom")
        return DocBuilder(name="whatever").title("T").text("Body.", page=1).build()


def test_writes_one_loadable_cache_file_per_pdf(tmp_path: Path) -> None:
    pdf_dir, parsed_dir = tmp_path / "pdfs", tmp_path / "parsed"
    _pdfs(pdf_dir, "a", "b")

    failures = run_parse(pdf_dir, parsed_dir, FakeConverter())

    assert failures == 0
    assert sorted(p.name for p in parsed_dir.iterdir()) == ["a.json", "b.json"]
    # doc.name is set to the paper id, which load_parsed checks.
    assert load_parsed(parsed_dir, "a") is not None
    assert load_parsed(parsed_dir, "b") is not None


def test_skips_existing_unless_force(tmp_path: Path) -> None:
    pdf_dir, parsed_dir = tmp_path / "pdfs", tmp_path / "parsed"
    _pdfs(pdf_dir, "a", "b")
    run_parse(pdf_dir, parsed_dir, FakeConverter())

    skipped = FakeConverter()
    run_parse(pdf_dir, parsed_dir, skipped)
    assert skipped.calls == []

    forced = FakeConverter()
    run_parse(pdf_dir, parsed_dir, forced, force=True)
    assert forced.calls == ["a", "b"]


def test_only_restricts_the_run(tmp_path: Path) -> None:
    pdf_dir, parsed_dir = tmp_path / "pdfs", tmp_path / "parsed"
    _pdfs(pdf_dir, "a", "b", "c")
    converter = FakeConverter()

    failures = run_parse(pdf_dir, parsed_dir, converter, only=["b"])

    assert failures == 0
    assert converter.calls == ["b"]
    assert [p.name for p in parsed_dir.iterdir()] == ["b.json"]


def test_only_unknown_id_is_a_failure(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    pdf_dir, parsed_dir = tmp_path / "pdfs", tmp_path / "parsed"
    _pdfs(pdf_dir, "a")

    with caplog.at_level(logging.ERROR):
        failures = run_parse(pdf_dir, parsed_dir, FakeConverter(), only=["nope"])

    assert failures == 1
    assert "nope" in caplog.text


def test_one_failure_does_not_stop_the_others_or_leave_a_partial_file(
    tmp_path: Path,
) -> None:
    pdf_dir, parsed_dir = tmp_path / "pdfs", tmp_path / "parsed"
    _pdfs(pdf_dir, "a", "b", "c")

    failures = run_parse(pdf_dir, parsed_dir, FakeConverter(fail_on={"b"}))

    assert failures == 1
    assert sorted(p.name for p in parsed_dir.iterdir()) == ["a.json", "c.json"]


def test_missing_parser_prints_install_hint_and_exits_1(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    _pdfs(tmp_path / "pdfs", "a")

    def no_parser() -> None:
        raise ImportError("No module named 'docling'")

    monkeypatch.setattr(parse_papers, "build_converter", no_parser)
    monkeypatch.setattr(
        "sys.argv",
        [
            "parse_papers.py",
            "--pdf-dir",
            str(tmp_path / "pdfs"),
            "--parsed-dir",
            str(tmp_path / "parsed"),
        ],
    )

    with caplog.at_level(logging.ERROR), pytest.raises(SystemExit) as exc:
        parse_papers.main()

    assert exc.value.code == 1
    assert "pip.exe install" in caplog.text
    assert not (tmp_path / "parsed").exists()
