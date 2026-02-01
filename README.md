# ai-learn

A personal learning project for understanding the fundamentals of developing software using LLMs and reasoning models. This project focuses on **AI technology as a component**—what it takes to build products with these tools—rather than techniques for AI-assisted software development.

I got interested after attending the [DeepLearning.AI developer conference in 2025](deeplearning-ai-dev-2025/README.md). This project is an attempt to understand the building blocks of AI development and develop a learning approach for what is required to build products using this new technology.

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
