# ADR-0021: Chunking with Docling's HybridChunker, chunk/citation identity

## Status

Accepted (Phase 8 Plan); **revised in Phase 8 Execute**. The owner replaced the custom
chunker with Docling's HybridChunker. Each custom part that stays is listed below with
its reason. The first version is under Alternatives.

## Context

Up to Phase 7, each paper was indexed as a single point holding its abstract, with an
integer point ID taken from its position in the batch. Phase 8 indexes the full text of
the papers. They are parsed with Docling (ADR-0020) and cached as Docling's own
`DoclingDocument` JSON. That raises five questions: how to cut the text into pieces that
fit the embedder, how to handle tables and figures, how to label a piece so a citation
can point to it, how to keep re-ingestion idempotent, and how `Citation` changes. Phase 9
labels its golden set at paper + section level, so the section label has to be stable.

## Decision

**Docling does the chunking** (`app/retrieval/chunking.py`): `HybridChunker(merge_peers=True,
repeat_table_header=True)` over the cached `DoclingDocument`, with
`HuggingFaceTokenizer` over the embedder's own tokenizer. From Docling, unchanged:
- split by document item first, then by text (semchunk) for an item over the limit;
- merge undersized neighbours that share the same headings;
- a table over the limit is split by rows, and the header row is repeated in every piece;
- page headers and footers are skipped (the furniture content layer);
- pictures are serialized as their caption;
- the embedding text uses `contextualize()`'s layout: heading path and text, joined by
  newlines;
- the page comes from the first item's provenance.

Two settings change on Docling's chunking serializer. Tables are rendered as compact
markdown instead of the default `row, column = value` triplets, because markdown reads
well as a citation excerpt and uses far fewer tokens on wide tables. Footnotes are left
out: they sit in the body layer, and the serializer would otherwise index them.

**What stays custom, and why**:
1. **Heading levels.** Docling's PDF pipeline reports every heading at level 1
   (ADR-0020), so its heading path would hold only the leaf heading. On a copy of the
   document, `SectionHeaderItem.level` is rewritten from the numbering (`3.2` → 2,
   `A.1` → 2) and the canonical names. Abstract, Introduction, Background, Related work,
   Method(s), Methodology, Online methods, Materials and methods, Experiments, Results, Discussion,
   Conclusion(s), Limitations and Appendix start a top level. Any other unnumbered
   heading becomes a child of the current top level. Headings Docling invents from page
   furniture (`(continued from previous page)`, `Contents`) are left out of the label and
   the embed text. Once the back matter has started,
   a dotless appendix letter (`A Data`) counts as numbering; before it, `A Comparison of
   Models` stays text. HybridChunker then builds the real path. Docling has no built-in
   fix. The third-party `docling-hierarchical-pdf` was not evaluated, because it would be
   one more dependency to replace about 40 lines.
2. **Dropped content.** Docling's PDF pipeline never emits the `reference` label, so it
   can't tell a bibliography from body text. A chunk is dropped if any heading on its path
   is References, Bibliography, Acknowledg(e)ments, Author contributions, Competing
   interests, Funding or Abstract. The parsed abstract is dropped because the manifest
   abstract replaces it. Front matter (items before the first real section heading:
   authors, affiliations) is dropped too. Since levels are normalised, "resume after
   References" follows from the heading stack: `A Data` or `Appendix` (top level) is
   kept, and a dropped `5.2 Funding` does not hide `5.3 Limitations`.
3. **Manifest abstract = chunk 0**, `section = "Abstract"`, page 1. It is curated
   metadata, not PDF content, and it is the abstract-only fallback when there is no
   parsed file (a fresh clone). It is split by the same chunker if it is too long.
4. **Section label**: a path of at most two levels, `"Methods > Datasets"`. Numbering is
   stripped and ALL-CAPS is title-cased (U7, owner decision); deeper headings map to their
   depth-2 ancestor. A leaf label is ambiguous (several "Evaluation" subsections), and a
   top-level label alone is too coarse for Phase 9. The cache keeps the raw heading text,
   so labels can be derived again later.
5. **Token budget.** Docling counts `tokenizer.tokenize()`, which leaves out
   `[CLS]`/`[SEP]`. So the budget is `max_seq_length − num_special_tokens_to_add()` (510
   for bge-small, ADR-0022), minus the tokens of the manifest-title line that we put
   first. The embed text is `"{manifest title}\n{heading path}\n{body}"`, with Docling's
   own detected title left out because it is unreliable (ADR-0020). It always fits, so the
   embedder never truncates. On `llava_med`, the longest chunk is 502 tokens including
   special tokens.
6. **`chunk_type`** comes from the items' labels: a table → `table`; only pictures and
   captions → `caption`; otherwise `text`. It goes by label because `DocMeta.doc_items`
   comes back as base `DocItem`s.

**Size.** Chunks fill up to the budget. There is no separate body target and no overlap,
because HybridChunker has neither. Phase 9 tunes the size, by passing a lower budget, and
measures whether overlap helps.

**Identity.** `chunk_index` counts 0..n through the paper in reading order. Point ID =
`uuid5(PAPERLENS_NS, f"{paper_id}:{chunk_index}")`, with `paper_id` = the manifest file
stem. Ingest deletes a paper's points by a `paper_id` payload filter and then upserts. Re-runs give the same
IDs, and no stale chunks are left behind when a paper gets fewer chunks. Payload:
`title, authors, year, doi, paper_id, section, page, chunk_index, chunk_type, text,
embedding_model`.

**`Citation` contract.** New optional fields: `paper_id: str | None`, `section: str |
None`, `page: int | None` and `chunk_index: int | None`, all defaulting to `None`. One
citation per retrieved chunk, so several citations can come from the same paper. The
`/query` and `/query/stream` shapes otherwise stay the same, and old payloads and test
fixtures without the fields still validate.

## Alternatives Considered

### Custom structure-aware chunker (first version of this ADR)

A parser-agnostic block cache (`ParsedPaper`) with its own packer: paragraphs, then
sentences, a body target of about 350 tokens, about 50 tokens of sentence overlap
within a section, tables and captions deferred to the end of their section, and its own
row splitter for tables.

Pros:
- Overlap and a body target separate from the embedder limit.
- Not tied to one parser.

Cons:
- About 500 lines of our own splitting and packing code, which Docling already ships
  and maintains.
- Its own cache schema plus a mapping step from Docling's output, which loses
  information. The native `DoclingDocument` is lossless, and labels or chunks can be
  derived again from it.

### Fixed-size sliding window over the whole text

Pros:
- Simple, no reliance on heading detection.

Cons:
- Chunks cross section boundaries, so there is no clean section label for citations or
  for Phase 9 labels. Tables are cut apart.

### Docling's heading path without level normalisation

Pros:
- No custom heading code.

Cons:
- With flat levels the path is just the leaf heading, which is ambiguous (see item 4).

### Keep integer point IDs

Pros:
- No change.

Cons:
- IDs shift when a paper is added or re-parsed, so upserts overwrite the wrong points.

## Consequences

Positive:
- Citations point to paper + section + page. Re-ingest is idempotent and cleans up after
  itself.
- The splitting, table handling and token sizing are Docling's, so we maintain only the
  rules that are specific to scientific papers.
- A missing cache file degrades to abstract-only, so a fresh clone still runs.

Negative:
- `docling-core[chunking]` is a runtime dependency of ingestion (in the Docker image):
  docling-core, semchunk and tree-sitter with its grammars. The parser itself (`docling`)
  stays host-only.
- No overlap; a sentence cut at a chunk boundary has no context on either side.
- Most figure captions are not linked to their picture by Docling (1 of 7 on
  `llava_med`). Unlinked captions are indexed as text, and `merge_peers` can merge a
  short caption into the text around it, which then counts as `text`. So
  `chunk_type == "caption"` undercounts captions.
- Section labels depend on heading detection. False headings from Docling ("Caption:",
  text inside figures) become labels without any error (T12 inspects them).
- One paper can fill all top-k slots (accepted; Phase 9 decides on a per-paper cap).

## Execute results

Real-corpus run (T12): 595 chunks over the 10 papers (52 / 52 / 40 / 18 / 40 / 38 / 104 /
89 / 57 / 105 per paper; 28 % tables), longest embed text 512 tokens including special
tokens, median 338–402 per paper, pages from 1 to each paper's last page. Heading checks
changed two things: `Online methods` became a canonical top level (`conch`'s Methods had
come out as `Discussion > …`), and the false headings `(continued from previous page)`
and `Contents` are left out of label and embed text. Known and left alone: `Input` /
`Target` (`med_palm_m`) and `Caption:` / `Example input:` (`llava_med`) are labels
inside examples; `radfm` has a long title-variant top-level heading; `maira_2` has 60+
distinct labels from its appendix.

## Follow-up

- Phase 9: tune chunk size, measure overlap, table-chunk retrieval and the embedding
  header.
- Phase 9: per-paper cap on citations if top-k gets crowded by a single paper.
- Phase 9: decide whether the remaining odd labels need fixing; none breaks retrieval or
  citations.
