import json
from pathlib import Path

import httpx
import pytest

from scripts.download_papers import USER_AGENT, download_papers

PAPERS_DIR = Path(__file__).parent.parent / "data" / "papers"


def _write_manifest(papers_dir: Path, paper_id: str, url: str | None) -> None:
    manifest = {"title": paper_id}
    if url is not None:
        manifest["pdf_url"] = url
    (papers_dir / f"{paper_id}.json").write_text(json.dumps(manifest), encoding="utf-8")


def _client(handler: httpx.MockTransport) -> httpx.Client:
    return httpx.Client(transport=handler, headers={"User-Agent": USER_AGENT})


def test_real_manifests_have_arxiv_id_and_pdf_url() -> None:
    manifests = sorted(PAPERS_DIR.glob("*.json"))
    assert len(manifests) == 10
    for path in manifests:
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["pdf_url"] == f"https://arxiv.org/pdf/{data['arxiv_id']}"
        assert "v" in data["arxiv_id"], f"{path.stem}: arXiv ID must be versioned"


@pytest.mark.parametrize("paper_id", ["conch", "cxr_vlm_generation"])
def test_preprint_manifests_carry_source_note(paper_id: str) -> None:
    data = json.loads((PAPERS_DIR / f"{paper_id}.json").read_text(encoding="utf-8"))
    assert data["source_note"]


def test_downloads_pdf_and_sends_user_agent(tmp_path: Path) -> None:
    papers, pdfs = tmp_path / "papers", tmp_path / "pdfs"
    papers.mkdir()
    _write_manifest(papers, "a", "https://example.org/a.pdf")
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, content=b"%PDF-1.7 body")

    with _client(httpx.MockTransport(handler)) as client:
        failures = download_papers(papers, pdfs, client, sleep=lambda _: None)

    assert failures == 0
    assert (pdfs / "a.pdf").read_bytes() == b"%PDF-1.7 body"
    assert not list(pdfs.glob("*.part"))
    assert seen[0].headers["user-agent"] == USER_AGENT


def test_skips_existing_files_unless_forced(tmp_path: Path) -> None:
    papers, pdfs = tmp_path / "papers", tmp_path / "pdfs"
    papers.mkdir()
    pdfs.mkdir()
    _write_manifest(papers, "a", "https://example.org/a.pdf")
    (pdfs / "a.pdf").write_bytes(b"%PDF-old")
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(200, content=b"%PDF-new")

    with _client(httpx.MockTransport(handler)) as client:
        assert download_papers(papers, pdfs, client, sleep=lambda _: None) == 0
        assert calls == 0
        assert (pdfs / "a.pdf").read_bytes() == b"%PDF-old"
        assert (
            download_papers(papers, pdfs, client, force=True, sleep=lambda _: None) == 0
        )

    assert calls == 1
    assert (pdfs / "a.pdf").read_bytes() == b"%PDF-new"


def test_bad_magic_bytes_is_a_failure_and_writes_nothing(tmp_path: Path) -> None:
    papers, pdfs = tmp_path / "papers", tmp_path / "pdfs"
    papers.mkdir()
    _write_manifest(papers, "a", "https://example.org/a.pdf")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"<html>not a pdf</html>")

    with _client(httpx.MockTransport(handler)) as client:
        failures = download_papers(papers, pdfs, client, sleep=lambda _: None)

    assert failures == 1
    assert not list(pdfs.iterdir())


def test_http_error_and_missing_url_count_as_failures_but_others_continue(
    tmp_path: Path,
) -> None:
    papers, pdfs = tmp_path / "papers", tmp_path / "pdfs"
    papers.mkdir()
    _write_manifest(papers, "a_missing_url", None)
    _write_manifest(papers, "b_error", "https://example.org/b.pdf")
    _write_manifest(papers, "c_ok", "https://example.org/c.pdf")

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/b.pdf":
            return httpx.Response(503)
        return httpx.Response(200, content=b"%PDF-1.7")

    with _client(httpx.MockTransport(handler)) as client:
        failures = download_papers(papers, pdfs, client, sleep=lambda _: None)

    assert failures == 2
    assert (pdfs / "c_ok.pdf").exists()


def test_waits_between_requests_only(tmp_path: Path) -> None:
    papers, pdfs = tmp_path / "papers", tmp_path / "pdfs"
    papers.mkdir()
    for pid in ("a", "b", "c"):
        _write_manifest(papers, pid, f"https://example.org/{pid}.pdf")
    sleeps: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"%PDF-1.7")

    with _client(httpx.MockTransport(handler)) as client:
        download_papers(papers, pdfs, client, delay=3.0, sleep=sleeps.append)

    assert sleeps == [3.0, 3.0]
