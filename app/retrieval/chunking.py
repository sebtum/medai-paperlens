"""Chunking with Docling's HybridChunker (ADR-0021).

Turns a manifest abstract plus an optional parsed `DoclingDocument` into `Chunk`s that
carry a section label and a page, so a citation can point at paper + section + page.

Docling does the chunking itself: splitting by document item and then by text
(semchunk), tokenizer-aware sizing, repeating a table's header row when it is split,
merging undersized neighbours with the same headings, skipping page headers and
footers (furniture layer), and serializing pictures as their captions. What this module
adds, and why Docling does not cover it:

- **Heading levels.** Docling's PDF pipeline reports every heading at level 1
  (ADR-0020), so its heading path would be the leaf heading only. The levels are
  rewritten from the numbering and canonical section names (U7) on a copy, and the
  chunker then builds the real path.
- **Dropped sections.** Docling does not detect bibliographies or other back matter, so
  chunks under References, Acknowledgements, Funding, ... (and the parsed abstract,
  which the manifest abstract replaces) are filtered out, as is front matter.
- **Manifest abstract** as chunk 0: curated metadata, and the fallback when there is no
  parsed file.
- **Token budget.** Docling counts tokens without the special tokens and does not know
  about the title line we prepend, so the budget is reduced by both.
- **Section label and `chunk_type`**: our citation contract.

Chunks are embedded as `"{title}\\n{heading path}\\n{body}"` (Docling's `contextualize`
layout with the manifest title as the first line), and that whole string stays within
the tokenizer's budget, so the embedder never truncates.
"""

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

from docling_core.transforms.chunker.doc_chunk import DocChunk
from docling_core.transforms.chunker.hierarchical_chunker import (
    ChunkingDocSerializer,
    ChunkingSerializerProvider,
)
from docling_core.transforms.chunker.hybrid_chunker import HybridChunker
from docling_core.transforms.chunker.tokenizer.base import BaseTokenizer
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
from docling_core.transforms.serializer.base import BaseDocSerializer
from docling_core.transforms.serializer.markdown import (
    MarkdownParams,
    MarkdownTableSerializer,
)
from docling_core.types.doc import (
    DocItemLabel,
    DoclingDocument,
    SectionHeaderItem,
    TitleItem,
)

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

ChunkType = Literal["text", "table", "caption"]

ABSTRACT_SECTION = "Abstract"

# Top-level names that start a new section even without a number (ADR-0021).
_CANONICAL_TOP_LEVEL = frozenset(
    {
        "abstract",
        "introduction",
        "background",
        "related work",
        "related works",
        "method",
        "methods",
        "methodology",
        "online methods",
        "materials and methods",
        "experiments",
        "results",
        "discussion",
        "conclusion",
        "conclusions",
        "limitations",
        "appendix",
    }
)

# Sections that are not indexed. The parsed abstract is dropped because the manifest
# abstract is chunk 0.
_DROPPED_SECTION = re.compile(
    r"^(abstract|references?|bibliography|acknowledg(e)?ments?"
    r"|author contributions?|competing interests?|funding)\b"
)

# Headings Docling invents from page furniture (`(continued from previous page)` on a
# table that runs over a page, a table-of-contents label). They are not sections: they
# are left out of the label and the embed text, and the chunk keeps its parent's.
_FALSE_HEADING = re.compile(
    r"^\(?(continued( from previous page)?|(table of )?contents)\)?$", re.IGNORECASE
)

# Heading numbering: `3`, `3.2`, `A.1`, `A.`, `IV.`. A bare capital letter without a
# following dot or bracket is not numbering ("A Comparison of Models").
_NUMBERING = re.compile(
    r"^(?P<num>\d+(?:\.\d+)*|[A-Z](?:\.\d+)+|[IVXL]{1,5}(?=[.)]\s)|[A-Z](?=[.)]\s))"
    r"(?:[.)])?\s+"
)

# Appendix lettering without a dot (`A Data`, `B Prompts`), read only in back matter.
_APPENDIX_LETTER = re.compile(r"^[A-Z]\s+(?=[A-Z0-9])")

_SMALL_WORDS = frozenset(
    {"a", "an", "and", "as", "at", "for", "in", "of", "on", "or", "the", "to", "with"}
)

# Docling's chunking serializer, with two settings changed: tables as compact markdown
# (readable as a citation excerpt, and far fewer tokens than the default
# "row, column = value" triplets), and footnotes left out (they are in the body layer).
_SERIALIZER_PARAMS: MarkdownParams = ChunkingDocSerializer.model_fields[
    "params"
].default.model_copy(
    update={
        "labels": ChunkingDocSerializer.model_fields["params"].default.labels
        - {DocItemLabel.FOOTNOTE},
        "compact_tables": True,
    }
)


@dataclass(frozen=True)
class Chunk:
    paper_id: str
    chunk_index: int
    chunk_type: ChunkType
    section: str
    page: int
    text: str  # the body: payload `text` and `Citation.excerpt`
    embed_text: str  # "{title}\n{heading path}\n{body}": what the embedder sees


class _MarkdownChunkingSerializerProvider(ChunkingSerializerProvider):
    """Docling's documented way to swap the chunker's table serializer."""

    def get_serializer(self, doc: DoclingDocument) -> BaseDocSerializer:
        return ChunkingDocSerializer(
            doc=doc,
            table_serializer=MarkdownTableSerializer(),
            params=_SERIALIZER_PARAMS,
        )


class _ReservedTokenizer(BaseTokenizer):
    """A tokenizer whose budget leaves room for the title line we prepend."""

    inner: BaseTokenizer
    max_tokens: int

    def count_tokens(self, text: str) -> int:
        return self.inner.count_tokens(text)

    def get_max_tokens(self) -> int:
        return self.max_tokens

    def get_tokenizer(self) -> Any:
        return self.inner.get_tokenizer()


def make_embedder_tokenizer(model: SentenceTransformer) -> HuggingFaceTokenizer:
    """Docling tokenizer for `model` whose budget is what the model really accepts.

    Docling counts `tokenizer.tokenize(text)`, which leaves out `[CLS]`/`[SEP]`, so
    `max_seq_length` itself would let the embedder truncate.
    """
    tokenizer = model.tokenizer
    budget = model.max_seq_length - tokenizer.num_special_tokens_to_add(pair=False)
    return HuggingFaceTokenizer(tokenizer=tokenizer, max_tokens=budget)


# --- Section labels -----------------------------------------------------------------


@dataclass(frozen=True)
class _Heading:
    depth: int | None  # None: unnumbered, not a canonical top-level name
    title: str
    dropped: bool


def _smart_title(text: str) -> str:
    words = text.lower().split()
    return " ".join(
        w if (i and w in _SMALL_WORDS) else w[:1].upper() + w[1:]
        for i, w in enumerate(words)
    )


def _parse_heading(raw: str, *, appendix_letters: bool = False) -> _Heading:
    """`appendix_letters`: also read `A Data` as appendix numbering (back matter)."""
    text = " ".join(raw.split())
    depth: int | None = None
    if match := _NUMBERING.match(text):
        depth = match.group("num").count(".") + 1
        text = text[match.end() :]
    elif appendix_letters and (match := _APPENDIX_LETTER.match(text)):
        depth = 1
        text = text[match.end() :]
    text = text.strip().rstrip(".:").strip()
    if any(c.isalpha() for c in text) and text.isupper():
        text = _smart_title(text)

    lowered = text.lower()
    dropped = bool(_DROPPED_SECTION.match(lowered))
    if depth is None and (
        dropped or lowered in _CANONICAL_TOP_LEVEL or lowered.startswith("appendix")
    ):
        depth = 1
    return _Heading(depth=depth, title=text, dropped=dropped)


def _title_key(text: str) -> str:
    return re.sub(r"[\W_]+", " ", text).strip().casefold()


def _normalise_heading_levels(
    doc: DoclingDocument, title_key: str
) -> dict[str, _Heading]:
    """Rewrite `SectionHeaderItem.level` in place from numbering and names (U7).

    Numbered headings get their depth (`3.2` → 2). Canonical names start a top level.
    Any other unnumbered heading is a child of the current top level, or the top level
    itself if none has started yet. A heading that repeats the paper title is put at
    the top level, so it closes nothing below it. After the back matter has started
    (References, ...), `A Data` counts as appendix numbering; before it, a leading
    capital stays part of the text ("A Comparison of Models").

    Returns each raw heading text's parse, so labels follow the same reading.
    """
    parsed: dict[str, _Heading] = {}
    have_top = False
    in_back_matter = False
    for item, _ in doc.iterate_items():
        if not isinstance(item, SectionHeaderItem):
            continue
        if _title_key(item.text) == title_key:
            item.level = 1
            continue
        heading = _parse_heading(item.text, appendix_letters=in_back_matter)
        parsed[item.text] = heading
        if heading.depth is not None:
            item.level = min(heading.depth, 100)
        else:
            item.level = 2 if have_top else 1
        have_top = True
        if heading.dropped and item.level == 1 and heading.title.lower() != "abstract":
            in_back_matter = True
    return parsed


def _front_matter_refs(doc: DoclingDocument, title_key: str) -> set[str]:
    """Refs of body items before the first real section heading (authors etc.)."""
    refs: set[str] = set()
    for item, _ in doc.iterate_items():
        if isinstance(item, SectionHeaderItem) and _title_key(item.text) != title_key:
            break
        refs.add(item.self_ref)
    return refs


def _section_headings(headings: list[str] | None, title_keys: set[str]) -> list[str]:
    """The chunk's heading path minus the title and false headings (`title_keys`)."""
    return [h for h in headings or [] if _title_key(h) not in title_keys]


def _section_label(path: list[_Heading]) -> str:
    """Depth ≤ 2 label: `"Methods > Datasets"`; deeper headings map to their parent."""
    if not path:
        return ""
    if len(path) >= 2 and path[0].title != path[1].title:
        return f"{path[0].title} > {path[1].title}"
    return path[0].title


def _doc_chunks(chunker: HybridChunker, doc: DoclingDocument) -> list[DocChunk]:
    chunks = list(chunker.chunk(doc))
    if not all(isinstance(chunk, DocChunk) for chunk in chunks):
        raise TypeError("HybridChunker returned a chunk that is not a DocChunk")
    return chunks  # type: ignore[return-value]


_TABLE_LABELS = frozenset({DocItemLabel.TABLE, DocItemLabel.DOCUMENT_INDEX})
_CAPTION_LABELS = frozenset(
    {DocItemLabel.PICTURE, DocItemLabel.CHART, DocItemLabel.CAPTION}
)


def _chunk_type(chunk: DocChunk) -> ChunkType:
    # By label, not isinstance: `DocMeta.doc_items` is `list[DocItem]`, so pydantic
    # hands back base `DocItem`s, not `TableItem`/`PictureItem`.
    labels = {item.label for item in chunk.meta.doc_items}
    if labels & _TABLE_LABELS:
        return "table"
    if labels <= _CAPTION_LABELS:
        return "caption"
    return "text"


def _page(chunk: DocChunk) -> int:
    for item in chunk.meta.doc_items:
        if item.prov:
            return item.prov[0].page_no
    return 1


def _abstract_doc(paper_id: str, abstract: str) -> DoclingDocument:
    doc = DoclingDocument(name=paper_id)
    doc.add_heading(ABSTRACT_SECTION, level=1)
    doc.add_text(label=DocItemLabel.TEXT, text=abstract)
    return doc


# --- Public API ---------------------------------------------------------------------


def chunk_paper(
    manifest: Mapping[str, Any],
    doc: DoclingDocument | None,
    tokenizer: BaseTokenizer,
    *,
    paper_id: str,
) -> list[Chunk]:
    """Chunk one paper: the manifest abstract first, then the parsed full text.

    `tokenizer.get_max_tokens()` is the budget for the whole embed text (see
    `make_embedder_tokenizer`). With `doc=None` only the abstract chunks are returned.
    `doc` is not modified. Raises `ValueError` if the title leaves no room for text.
    """
    title = " ".join(str(manifest.get("title", "")).split()).rstrip(".")
    title_key = _title_key(title)
    reserved = tokenizer.count_tokens(f"{title}\n") if title else 0
    budget = tokenizer.get_max_tokens() - reserved
    if budget <= 0:
        raise ValueError(
            f"title {title!r} leaves no room within {tokenizer.get_max_tokens()} tokens"
        )
    chunker = HybridChunker(
        tokenizer=_ReservedTokenizer(inner=tokenizer, max_tokens=budget),
        serializer_provider=_MarkdownChunkingSerializerProvider(),
        merge_peers=True,
        repeat_table_header=True,
    )
    chunks: list[Chunk] = []

    def add(chunk: DocChunk, section: str, page: int, headings: list[str]) -> None:
        body = chunk.text.strip()
        if not body:
            return
        embed_text = "\n".join([*([title] if title else []), *headings, body])
        chunks.append(
            Chunk(
                paper_id=paper_id,
                chunk_index=len(chunks),
                chunk_type=_chunk_type(chunk),
                section=section,
                page=page,
                text=body,
                embed_text=embed_text,
            )
        )

    abstract = " ".join(str(manifest.get("abstract", "")).split())
    if abstract:
        for chunk in _doc_chunks(chunker, _abstract_doc(paper_id, abstract)):
            add(chunk, ABSTRACT_SECTION, 1, [ABSTRACT_SECTION])

    if doc is None:
        return chunks

    doc = doc.model_copy(deep=True)
    parsed_headings = _normalise_heading_levels(doc, title_key)
    front_matter = _front_matter_refs(doc, title_key)
    title_keys = {title_key} | {
        _title_key(item.text)
        for item, _ in doc.iterate_items()
        if isinstance(item, TitleItem)
        or (
            isinstance(item, SectionHeaderItem)
            and _FALSE_HEADING.match(item.text.strip())
        )
    }

    last_section: str | None = None
    for chunk in _doc_chunks(chunker, doc):
        if all(item.self_ref in front_matter for item in chunk.meta.doc_items):
            continue
        headings = _section_headings(chunk.meta.headings, title_keys)
        path = [
            heading
            for raw in headings
            if (heading := parsed_headings.get(raw) or _parse_heading(raw)).title
        ]
        if any(h.dropped for h in path):
            continue
        # An empty path after the front matter means a stray title-like item reset
        # Docling's heading stack: keep the text under the last section seen.
        section = _section_label(path) or last_section
        if section is None:
            continue
        last_section = section
        add(chunk, section, _page(chunk), headings)
    return chunks
