# Examples

Runnable examples. Not optimized and designed to illustrate principles.
Run from this directory with:

```bash
python <filename>.py
```

Example: `python understand_llm_models.py`

## Data directory

Running the examples may create a **`data/`** directory for:

- **Logs**: Request/response dumps (e.g. `data/YYYYMMDD/<timestamp>-openai-response.json`).
- **Persistent storage**: e.g. SQLite DBs like `data/embeddings_vector_search.db`.

`data/` is in `.gitignore` and is not committed. Delete it anytime to clear logs and stored data.
