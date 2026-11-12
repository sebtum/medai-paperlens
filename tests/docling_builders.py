"""Helpers that build small `DoclingDocument`s in code, plus a model-free tokenizer.

Shared by test_chunking.py and test_parsed.py. Nothing here downloads a model.
"""

from collections.abc import Callable
from typing import Any

from docling_core.transforms.chunker.tokenizer.base import BaseTokenizer
from docling_core.types.doc import (
    BoundingBox,
    ContentLayer,
    DocItemLabel,
    DoclingDocument,
    ProvenanceItem,
    TableCell,
    TableData,
)


class WhitespaceTokenizer(BaseTokenizer):
    """One token per whitespace-separated word; semchunk accepts the callable."""

    max_tokens: int

    def count_tokens(self, text: str) -> int:
        return len(text.split())

    def get_max_tokens(self) -> int:
        return self.max_tokens

    def get_tokenizer(self) -> Callable[[str], int]:
        return self.count_tokens


def prov(page: int) -> ProvenanceItem:
    return ProvenanceItem(
        page_no=page, bbox=BoundingBox(l=0, t=0, r=1, b=1), charspan=(0, 1)
    )


def words(prefix: str, n: int) -> str:
    """`n` distinct words, so lost or duplicated text is easy to spot."""
    return " ".join(f"{prefix}{i}" for i in range(n))


def table_data(header: list[str], rows: list[list[str]]) -> TableData:
    cells = [
        TableCell(
            text=text,
            start_row_offset_idx=r,
            end_row_offset_idx=r + 1,
            start_col_offset_idx=c,
            end_col_offset_idx=c + 1,
            column_header=(r == 0),
        )
        for r, row in enumerate([header, *rows])
        for c, text in enumerate(row)
    ]
    return TableData(table_cells=cells, num_rows=len(rows) + 1, num_cols=len(header))


class DocBuilder:
    """Fluent builder; every item gets a page so `Chunk.page` can be checked."""

    def __init__(self, name: str = "p1") -> None:
        self.doc = DoclingDocument(name=name)

    def title(self, text: str, page: int = 1) -> DocBuilder:
        self.doc.add_title(text, prov=prov(page))
        return self

    def heading(self, text: str, page: int = 1) -> DocBuilder:
        # Docling's PDF pipeline reports every heading at level 1 (ADR-0020).
        self.doc.add_heading(text, level=1, prov=prov(page))
        return self

    def text(
        self, text: str, page: int = 1, label: DocItemLabel = DocItemLabel.TEXT
    ) -> DocBuilder:
        self.doc.add_text(label=label, text=text, prov=prov(page))
        return self

    def furniture(self, text: str, page: int = 1) -> DocBuilder:
        self.doc.add_text(
            label=DocItemLabel.PAGE_HEADER,
            text=text,
            prov=prov(page),
            content_layer=ContentLayer.FURNITURE,
        )
        return self

    def picture(self, caption: str, page: int = 1) -> DocBuilder:
        cap = self.doc.add_text(
            label=DocItemLabel.CAPTION, text=caption, prov=prov(page)
        )
        self.doc.add_picture(caption=cap, prov=prov(page))
        return self

    def table(
        self,
        header: list[str],
        rows: list[list[str]],
        caption: str | None = None,
        page: int = 1,
    ) -> DocBuilder:
        cap: Any = None
        if caption:
            cap = self.doc.add_text(
                label=DocItemLabel.CAPTION, text=caption, prov=prov(page)
            )
        self.doc.add_table(data=table_data(header, rows), caption=cap, prov=prov(page))
        return self

    def build(self) -> DoclingDocument:
        return self.doc
