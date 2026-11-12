"""Tests for the HybridChunker-based chunker (ADR-0021).

Documents are built in code (docling_builders) and token counts come from a
whitespace tokenizer, so no model is downloaded (R7).
"""

import pytest
from docling_builders import DocBuilder, WhitespaceTokenizer, words
from docling_core.types.doc import DocItemLabel

from app.retrieval.chunking import Chunk, chunk_paper

MANIFEST = {"title": "My Paper.", "abstract": "We study things.  It works."}
TITLE = "My Paper"


def _chunk(doc_builder: DocBuilder | None, max_tokens: int = 60, **manifest: str):
    tokenizer = WhitespaceTokenizer(max_tokens=max_tokens)
    doc = doc_builder.build() if doc_builder else None
    return chunk_paper({**MANIFEST, **manifest}, doc, tokenizer, paper_id="p1")


def _sections(chunks: list[Chunk]) -> list[str]:
    return [c.section for c in chunks]


def _body(chunks: list[Chunk]) -> str:
    return " ".join(c.text for c in chunks)


# --- Abstract -----------------------------------------------------------------------


def test_no_parsed_doc_gives_abstract_only() -> None:
    [chunk] = _chunk(None)

    assert chunk == Chunk(
        paper_id="p1",
        chunk_index=0,
        chunk_type="text",
        section="Abstract",
        page=1,
        text="We study things. It works.",
        embed_text="My Paper\nAbstract\nWe study things. It works.",
    )


def test_long_abstract_is_split_within_budget() -> None:
    chunks = _chunk(None, max_tokens=20, abstract=words("a", 50))

    assert len(chunks) > 1
    assert {c.section for c in chunks} == {"Abstract"}
    assert _body(chunks).split() == words("a", 50).split()
    assert all(len(c.embed_text.split()) <= 20 for c in chunks)


def test_no_abstract_and_no_doc_gives_nothing() -> None:
    assert _chunk(None, abstract="") == []


def test_title_too_long_raises() -> None:
    with pytest.raises(ValueError, match="no room"):
        _chunk(None, max_tokens=3, title="one two three four")


# --- Section labels -----------------------------------------------------------------


def test_numbered_headings_give_two_level_path() -> None:
    builder = (
        DocBuilder()
        .title(TITLE)
        .heading("1 Introduction")
        .text("intro text")
        .heading("2 Methods")
        .text("methods text")
        .heading("2.1 Datasets")
        .text("dataset text")
        .heading("2.1.1 Splits")
        .text("split text")
    )

    chunks = _chunk(builder)

    assert _sections(chunks) == [
        "Abstract",
        "Introduction",
        "Methods",
        "Methods > Datasets",
        "Methods > Datasets",
    ]
    assert chunks[3].embed_text == "My Paper\n2 Methods\n2.1 Datasets\ndataset text"
    # The depth-3 heading stays in the embed text but not in the label.
    assert chunks[4].embed_text.startswith("My Paper\n2 Methods\n2.1 Datasets\n2.1.1")


def test_unnumbered_headings_canonical_top_level_and_children() -> None:
    builder = (
        DocBuilder()
        .heading("Introduction")
        .text("intro text")
        .heading("Our Approach")
        .text("approach text")
        .heading("Results")
        .text("results text")
        .heading("Ablations")
        .text("ablation text")
    )

    assert _sections(_chunk(builder))[1:] == [
        "Introduction",
        "Introduction > Our Approach",
        "Results",
        "Results > Ablations",
    ]


def test_first_unnumbered_noncanonical_heading_is_top_level() -> None:
    builder = (
        DocBuilder()
        .heading("Overview")
        .text("overview text")
        .heading("Details")
        .text("detail text")
    )

    assert _sections(_chunk(builder))[1:] == ["Overview", "Overview > Details"]


def test_online_methods_is_a_top_level_section() -> None:
    builder = (
        DocBuilder()
        .heading("Discussion")
        .text("discussion text")
        .heading("Online Methods")
        .text("methods text")
        .heading("Pretraining dataset curation")
        .text("curation text")
    )

    assert _sections(_chunk(builder))[1:] == [
        "Discussion",
        "Online Methods",
        "Online Methods > Pretraining dataset curation",
    ]


@pytest.mark.parametrize(
    "false_heading", ["(continued from previous page)", "Contents"]
)
def test_false_headings_do_not_become_section_labels(false_heading: str) -> None:
    builder = (
        DocBuilder()
        .heading("Results")
        .text("before text")
        .heading(false_heading)
        .text("after text")
    )

    chunks = _chunk(builder)

    assert set(_sections(chunks)[1:]) == {"Results"}
    assert all(false_heading not in c.embed_text for c in chunks)
    assert "after text" in _body(chunks)


def test_heading_that_merely_starts_with_continued_is_kept() -> None:
    builder = (
        DocBuilder()
        .heading("Introduction")
        .text("intro text")
        .heading("Continued Pretraining")
        .text("pretraining text")
    )

    assert "Introduction > Continued Pretraining" in _sections(_chunk(builder))


def test_all_caps_heading_is_title_cased_and_numbering_stripped() -> None:
    builder = (
        DocBuilder()
        .heading("3 RESULTS AND DISCUSSION")
        .text("results text")
        .heading("3.1 EVALUATION OF THE MODEL")
        .text("eval text")
    )

    assert _sections(_chunk(builder))[1:] == [
        "Results and Discussion",
        "Results and Discussion > Evaluation of the Model",
    ]


def test_roman_and_letter_numbering_are_top_level() -> None:
    builder = (
        DocBuilder()
        .heading("IV. Experiments")
        .text("experiment text")
        .heading("A. Additional Results")
        .text("extra text")
        .heading("A.1 More Tables")
        .text("more text")
    )

    assert _sections(_chunk(builder))[1:] == [
        "Experiments",
        "Additional Results",
        "Additional Results > More Tables",
    ]


def test_bare_capital_letter_is_not_numbering() -> None:
    builder = DocBuilder().heading("A Comparison of Models").text("text")

    assert _sections(_chunk(builder))[1:] == ["A Comparison of Models"]


# --- Dropped content ----------------------------------------------------------------


def test_front_matter_and_parsed_abstract_are_dropped() -> None:
    builder = (
        DocBuilder()
        .title(TITLE)
        .text("Alice Smith, Bob Jones")
        .text("University of Somewhere")
        .heading("Abstract")
        .text("parsed abstract text")
        .heading("1 Introduction")
        .text("intro text")
    )

    chunks = _chunk(builder)

    assert _sections(chunks) == ["Abstract", "Introduction"]
    assert "Alice" not in _body(chunks)
    assert "parsed abstract" not in _body(chunks)


def test_title_reported_as_heading_counts_as_title() -> None:
    builder = (
        DocBuilder()
        .heading("MY PAPER")
        .text("Alice Smith")
        .heading("1 Introduction")
        .text("intro text")
    )

    chunks = _chunk(builder)

    assert _sections(chunks) == ["Abstract", "Introduction"]
    assert chunks[1].embed_text == "My Paper\n1 Introduction\nintro text"


def test_references_dropped_and_appendix_after_them_kept() -> None:
    builder = (
        DocBuilder()
        .heading("1 Introduction")
        .text("intro text")
        .heading("References")
        .text("[1] Some reference.")
        .heading("Supplementary Notes")
        .text("still references")
        .heading("Appendix A: Extra Results")
        .text("appendix text")
    )

    chunks = _chunk(builder)

    assert _sections(chunks)[1:] == ["Introduction", "Appendix A: Extra Results"]
    assert "reference" not in _body(chunks).lower()


def test_dotless_appendix_letters_after_references_are_kept() -> None:
    # Real Docling output for llava_med: "A Data", "B Prompts" after the references.
    builder = (
        DocBuilder()
        .heading("A Comparison of Models")  # before back matter: not numbering
        .text("comparison text")
        .heading("References")
        .text("[1] Some reference.")
        .heading("A Data")
        .text("appendix data text")
        .heading("B Prompts")
        .text("prompt text")
        .heading("Prompting GPT-4 to generate data")
        .text("prompting text")
    )

    chunks = _chunk(builder)

    assert _sections(chunks)[1:] == [
        "A Comparison of Models",
        "Data",
        "Prompts",
        "Prompts > Prompting GPT-4 to generate data",
    ]
    assert "reference" not in _body(chunks).lower()


def test_dropped_subsection_does_not_hide_its_sibling() -> None:
    builder = (
        DocBuilder()
        .heading("5 Discussion")
        .text("discussion text")
        .heading("5.2 Funding")
        .text("funded by someone")
        .heading("5.3 Limitations")
        .text("limitations text")
    )

    chunks = _chunk(builder)

    assert _sections(chunks)[1:] == ["Discussion", "Discussion > Limitations"]
    assert "funded" not in _body(chunks)


@pytest.mark.parametrize(
    "heading",
    ["Acknowledgements", "Acknowledgments", "Author Contributions", "Bibliography"],
)
def test_other_back_matter_is_dropped(heading: str) -> None:
    builder = (
        DocBuilder()
        .heading("1 Introduction")
        .text("intro text")
        .heading(heading)
        .text("back matter text")
    )

    assert "back matter" not in _body(_chunk(builder))


def test_page_furniture_and_footnotes_are_not_indexed() -> None:
    builder = (
        DocBuilder()
        .heading("1 Introduction")
        .furniture("arXiv:1234.5678 page header")
        .text("intro text")
        .text("1 A footnote.", label=DocItemLabel.FOOTNOTE)
    )

    body = _body(_chunk(builder))

    assert "arXiv" not in body
    assert "footnote" not in body


# --- Tables and captions ------------------------------------------------------------


def test_table_is_markdown_with_caption() -> None:
    builder = (
        DocBuilder()
        .heading("4 Results")
        .text(words("r", 40))
        .table(["Model", "AUC"], [["A", "0.9"]], caption="Table 1: Scores.", page=3)
    )

    table = next(c for c in _chunk(builder) if c.chunk_type == "table")

    assert table.section == "Results"
    assert table.page == 3
    assert table.text.startswith("Table 1: Scores.\n\n| Model | AUC |")
    assert "| A | 0.9 |" in table.text


def test_large_table_split_repeats_header_row() -> None:
    rows = [[f"m{i}", f"0.{i}"] for i in range(30)]
    builder = DocBuilder().heading("4 Results").table(["Model", "AUC"], rows)

    tables = [c for c in _chunk(builder, max_tokens=40) if c.chunk_type == "table"]

    assert len(tables) > 1
    assert all(t.text.startswith("| Model | AUC |") for t in tables)
    joined = "\n".join(t.text for t in tables)
    assert all(f"| m{i} | 0.{i} |" in joined for i in range(30))
    assert all(len(t.embed_text.split()) <= 40 for t in tables)


def test_picture_caption_becomes_caption_chunk() -> None:
    builder = (
        DocBuilder()
        .heading("3 Method")
        .text(words("m", 54))  # too full for the caption to be merged in
        .picture("Figure 2: Model overview.", page=4)
    )

    caption = next(c for c in _chunk(builder) if c.chunk_type == "caption")

    assert caption.text == "Figure 2: Model overview."
    assert (caption.section, caption.page) == ("Method", 4)


# --- Sizes, order, identity ---------------------------------------------------------


def _long_paper() -> DocBuilder:
    return (
        DocBuilder()
        .title(TITLE)
        .text("Alice Smith")
        .heading("1 Introduction", page=1)
        .text(words("i", 70), page=1)
        .text(words("j", 30), page=2)
        .heading("2 Methods", page=2)
        .heading("2.1 Data", page=2)
        .text(words("d", 120), page=3)
        .table(["k", "v"], [[f"k{i}", f"v{i}"] for i in range(25)], page=4)
        .picture("Figure 1: Overview.", page=4)
        .heading("References", page=5)
        .text("[1] ref", page=5)
    )


def test_every_embed_text_fits_budget_and_index_runs_in_order() -> None:
    chunks = _chunk(_long_paper(), max_tokens=50)

    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))
    assert all(len(c.embed_text.split()) <= 50 for c in chunks)
    assert all(c.embed_text.startswith("My Paper\n") for c in chunks)
    assert all(c.embed_text.endswith(c.text) for c in chunks)
    assert {c.paper_id for c in chunks} == {"p1"}


def test_no_text_lost_outside_dropped_sections() -> None:
    chunks = _chunk(_long_paper(), max_tokens=50)
    text_words = " ".join(c.text for c in chunks if c.chunk_type == "text").split()

    expected = [
        *words("i", 70).split(),
        *words("j", 30).split(),
        *words("d", 120).split(),
    ]
    assert [w for w in text_words if w[0] in "ijd" and w[1:].isdigit()] == expected


def test_page_is_the_page_of_the_first_item() -> None:
    chunks = _chunk(_long_paper(), max_tokens=50)

    data = [
        c for c in chunks if c.section == "Methods > Data" and c.chunk_type == "text"
    ]
    assert data and {c.page for c in data} == {3}
    assert chunks[1].page == 1


def test_parsed_doc_is_not_modified() -> None:
    doc = _long_paper().build()
    before = doc.export_to_dict()

    chunk_paper(MANIFEST, doc, WhitespaceTokenizer(max_tokens=50), paper_id="p1")

    assert doc.export_to_dict() == before


def test_small_paragraphs_in_one_section_are_merged() -> None:
    builder = (
        DocBuilder()
        .heading("1 Introduction")
        .text("first short paragraph")
        .text("second short paragraph")
    )

    [_, intro] = _chunk(builder)

    assert intro.text == "first short paragraph\nsecond short paragraph"


def test_stray_title_mid_document_keeps_text_under_last_section() -> None:
    builder = (
        DocBuilder()
        .heading("1 Introduction")
        .text(words("i", 54))
        .title("Some figure text misread as a title")
        .text("text after the stray title")
    )

    chunks = _chunk(builder)

    assert chunks[-1].text == "text after the stray title"
    assert chunks[-1].section == "Introduction"


# --- Embedder tokenizer -------------------------------------------------------------


def test_make_embedder_tokenizer_leaves_room_for_special_tokens() -> None:
    from types import SimpleNamespace

    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from tokenizers.processors import BertProcessing
    from transformers import PreTrainedTokenizerFast

    from app.retrieval.chunking import make_embedder_tokenizer

    vocab = {"[UNK]": 0, "[CLS]": 1, "[SEP]": 2, "hello": 3}
    backend = Tokenizer(WordLevel(vocab, unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    backend.post_processor = BertProcessing(("[SEP]", 2), ("[CLS]", 1))
    hf = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="[UNK]",
        cls_token="[CLS]",
        sep_token="[SEP]",
    )
    model = SimpleNamespace(tokenizer=hf, max_seq_length=512)

    tokenizer = make_embedder_tokenizer(model)  # type: ignore[arg-type]

    assert tokenizer.get_max_tokens() == 510
    assert tokenizer.count_tokens("hello hello") == 2
