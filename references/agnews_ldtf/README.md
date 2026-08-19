# `agnews_ldtf/` — Reference (read-only)

This directory holds **reference copies** of four files from the original
`agnews_ldtf/` project (located at `../../agnews_ldtf/agnews_ldtf/models/`).

> The original project remains intact at the repository root. The copies here are
> only used as a "look-up" while writing our own production code in
> `../../src/models/`. Do not import from this directory.

## Why keep a reference here?

* We re-implement the **class-conditioned attention** idea in `../../src/`.
* Having a side-by-side reference makes it easy to compare math, shapes, and
  defaults when something feels off in our own implementation.

## File mapping

| Reference file in this folder | Original in `../../agnews_ldtf/` | Used as input for |
|---|---|---|
| `bert_backbone.py` | `agnews_ldtf/models/bert_backbone.py` | how BERT layers are exposed |
| `token_router.py`  | `agnews_ldtf/models/token_router.py`  | scaled class-conditioned token attention |
| `depth_router.py`  | `agnews_ldtf/models/depth_router.py`  | layer-wise attention over `[L]` |
| `label_query_bank.py` | `agnews_ldtf/models/label_query_bank.py` | learnable per-class query vectors |

## How this directory is organized

Files are kept **identical** to the originals (modulo optional rewording in
docstrings) so we can diff them against our `src/` version when we want to
check behaviour.

## Should I edit these files?

**No.** Edit `../../src/models/` instead. The reference is here purely so
you can read the original code without leaving the project.