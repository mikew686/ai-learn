# ai-learn

A self-learning project to deepen my understanding of AI development: how modern LLMs and reasoning models work, how to use them as components in software, and which engineering patterns and examples support building with them.

## What’s in this project

- **Conference notes (2025)** — [DeepLearning.AI Dev Convention 2025](deeplearning-ai-dev-2025/README.md): themes, keynotes (e.g. Andrew Ng), context engineering, and technical directions from the conference.
- **AI research notes** — [ai-research/](ai-research/): foundations (neural nets → LLMs → GPTs, embeddings & vector search, reasoning, tool calling), how modern LLMs work, and how LLMs write code. Written to solidify conceptual understanding.
- **Dev patterns & examples** — [eng-dev-patterns/](eng-dev-patterns/) documents patterns (prompt engineering, structured output, function calling, schema-driven inference, embeddings/vector search, model selection). [src/examples/](src/examples/) holds runnable examples that demonstrate those patterns (chat vs reasoning models, tool use, embeddings, vector search with pgvector).

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Environment

Examples can use either **OPENROUTER** or **OPENAI** tokens. Set the appropriate variables in your environment:

- `OPENROUTER_API_KEY` for OpenRouter
- `OPENAI_API_KEY` for OpenAI

```bash
export OPENROUTER_API_KEY="your-openrouter-key"
# or
export OPENAI_API_KEY="your-openai-key"
```

---

## License

Code samples are licensed under the [MIT License](LICENSE).

Documentation, notes, and written content are licensed under [CC BY 4.0](docs/LICENSE).
