# Source: examples that support the documentation

This directory contains runnable examples that demonstrate the patterns described in the [eng-dev-patterns](../../eng-dev-patterns/) documentation. Use them to explore prompt engineering, structured output, tool use, schema-driven inference, embeddings/vector search, and model behavior (chat, reasoning, embedding-based context).

## Data directory (not in version control)

When you run these examples, a **`data/`** directory may be created for:

- **Logs**: Request/response dumps from `OpenAILog` (e.g. `data/YYYYMMDD/<timestamp>-openai-response.json`).
- **Persistent storage**: e.g. the SQLite vector store used by `embeddings_vector_search` (`data/embeddings_vector_search.db`).

`data/` is listed in `.gitignore` and is **opaque to GitHub**—it is not committed. You can delete it at any time to clear logs and stored data.

---

## Runnable examples

Run from the project root with `python -m src.scripts.examples.<module>`.

### Understanding model demos (interactive)

Interactive chats that illustrate how different model types behave; each has an accompanying `understand_*.md` that explains the theory.

| Script | What it demonstrates |
|--------|----------------------|
| `understand_llm_models` | Chat model with tool use (`fetch_url`); tool-call loop and message roles. |
| `understand_reasoning_model` | Reasoning model (e.g. o3-mini) with reasoning in content and tool use. |
| `understand_embedding_models` | Embedding-based context: retrieve similar past turns instead of full history. |

```bash
python -m src.scripts.examples.understand_llm_models [--model MODEL] [--temperature T] [--max-tokens N]
python -m src.scripts.examples.understand_reasoning_model [--model MODEL] [--temperature T] [--max-tokens N]
python -m src.scripts.examples.understand_embedding_models [--model MODEL] [--embedding-model EMB] [--retrieve-after N] [--retrieve-k K]
```

### Pattern examples (learning progression)

Examples that map to the [learning progression](../../eng-dev-patterns/learning_progression.md) and detailed pattern docs.

| Script | Pattern(s) | What it does |
|--------|------------|--------------|
| `system_prompt_example` | Prompt engineering | System vs user prompts; stateless vs stateful; token comparison. |
| `language_assessment_structured_output` | Structured output | Pydantic + parse; enums (language/region codes); nested and list fields. |
| `tool_use_patterns` | Function calling / tool use | Sequential, parallel, interleaved tool use; language analysis tool. |
| `schema_driven_translation` | Schema-driven inference | Minimal prompt; tools + structured output; schema as implicit instructions. |
| `embeddings_vector_search` | Embeddings / vector search | SQLite vector store; similar-past-translations few-shot; canonical language/region. |
| `pg_vector_search` | Embeddings / vector search | Postgres + pgvector; translation_targets + translations; exact-prompt embedding; cost/token tracking. |

```bash
python -m src.scripts.examples.system_prompt_example [--model MODEL]
python -m src.scripts.examples.language_assessment_structured_output [--model MODEL] [--prompt PHRASE] [--example-phrases]
python -m src.scripts.examples.tool_use_patterns [--model MODEL] [--mode sequential|parallel|interleaved] [--target LANG] [--prompt PHRASE]
python -m src.scripts.examples.schema_driven_translation [--model MODEL] [--prompt TEXT] [--target LANG]
python -m src.scripts.examples.embeddings_vector_search [--model MODEL] [--embedding-model EMB] [--db PATH] [--top-k N]
python -m src.scripts.examples.pg_vector_search [--model MODEL] [--embedding-model EMB] [--db-url URL] [--top-k N]
```

---

## Support files

- **`utils.py`** – Shared helpers: `create_client()`, `OpenAILog`, `print_indented()`. Used by all examples.
- **`db/`** – Postgres + pgvector helpers and notes for `pg_vector_search` (`db/utils.py`, `db/pg_user.md`, `db/postgres_pgvector_notes.md`).
- **`AGENTS.md`** – Conventions for adding or changing example scripts (logging, CLI, output, etc.).
- **`understand_*.md`** – Theory and walkthroughs for the understand_* demos (no separate runner).

Documentation lives in [eng-dev-patterns](../../eng-dev-patterns/): [README](../../eng-dev-patterns/README.md) (index), [learning_progression.md](../../eng-dev-patterns/learning_progression.md) (path with translation use case), and the individual pattern docs linked from there.
