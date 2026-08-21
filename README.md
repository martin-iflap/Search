# Search

A local CLI document search tool based on TF-IDF and cosine similarity.

It indexes `.txt`, `.docx`, and `.pdf` files from a target directory, ranks files by relevance to a query, and can optionally surface matching sentences from selected results.

## Tech stack

- Python
- `pymupdf` for PDF text extraction
- `python-docx` for DOCX parsing
- Optional spaCy lemmatization (`spacy` + `en_core_web_sm`)

## Setup

This project is managed with `uv`.

```bash
uv sync
```

If lemmatization is enabled and the model is missing, install it:

```bash
uv run python -m spacy download en_core_web_sm
```

## Run

From the repository root:

```bash
uv run python -m src.search
```

Optionally override the configured directory:

```bash
uv run python -m src.search --dir "C:\path\to\documents"
```

## Configuration

Main settings are in `config.json`:

- `search.dir_path`: default root directory
- `search.supported_file_types`: indexed extensions
- `search.search_threshold`: minimum similarity score
- `search.exclude_patterns`: skip patterns
- `vector_search.*`: ranking/lemmatization behavior
- `stopwords`: language stopword lists

## Tests

```bash
uv run pytest -q
```