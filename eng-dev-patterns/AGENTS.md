# Agent guide: eng-dev-patterns documentation style

This directory holds **pattern documentation** for AI engineering: model selection, prompt design, tool use, structured output, RAG, and related topics. The docs are written for humans learning or referencing patterns; this file describes the **documentation style** so agents and maintainers can add or edit content consistently.

## Document roles

- **README.md** – Entry point and index. Short overview of each pattern with **Description**, **Key Concepts** (or **Key Techniques** / **Key Approaches**), **Best Practices**, **Use Cases**, and a link to the detailed doc: `📖 [Detailed Documentation](./filename.md)`.
- **learning_progression.md** – Ordered learning path. Each pattern is a section with a fixed block: **Description**, **Technology**, **Key Concepts**, **Translation Example**, **Example Implementation** (bullets), and when applicable **Example** (link to `../src/example.py`). Uses translation as the running use case so patterns build on each other.
- **Detailed docs** (e.g. `understanding_models.md`, `schema_driven_inference.md`, `prompt_engineering.md`) – Long-form references. Structure: **Overview** → **Description** (and/or **How It Works**) → subsections with **###** and **####**, code blocks, **Best Practices**, **Documentation Links**, etc. They are linked from README and from learning_progression.

Other files (`llm_vs_reasoning_problems.md`, `function_calling_tool_use.md`, `structured_output.md`) follow the same long-form style when they are the “detailed” doc for a pattern.

## Pattern block structure (learning_progression and README)

When adding or editing a pattern section, use this structure so the index and progression stay consistent:

1. **Heading**: `## Pattern N: Pattern Name` (learning_progression) or `## Pattern Name` (README).
2. **Description**: One or two sentences; bold label `**Description**:`.
3. **Technology** (learning_progression only): e.g. `OpenAI SDK + Pydantic`, `OpenAI SDK (embeddings + chat) + Local vector storage`.
4. **Key Concepts** (or **Key Techniques** / **Key Approaches**): Bullet list of concepts.
5. **Translation Example** (learning_progression only): How this pattern applies to the translation use case in one sentence.
6. **Example Implementation**: Bullet list of concrete implementation points.
7. **Example** (learning_progression only, when there is a script): `**Example**: [script_name.py](../src/script_name.py)` — relative link from `eng-dev-patterns/` to `src/`.
8. **Best Practices** (README often; detailed docs): Bullet list.
9. **Use Cases** (README often): Bullet list.
10. **Related Patterns** (optional): Short list with links or names.
11. **Detailed doc link** (README): `📖 [Detailed Documentation](./filename.md)` at the top of the section.

Use `---` between major pattern sections in learning_progression and README.

## Detailed doc structure

Long-form docs typically include:

- **Title**: `# Pattern Name` or `# Understanding Models`-style.
- **Overview**: Short summary paragraph.
- **Description** (or **How It Works**): Explanation; can be split into numbered steps or subsections.
- **Subsections**: `###` for main divisions, `####` for sub-divisions (e.g. **Technical Usage** → **Basic Chat Completion**).
- **Code blocks**: Use ` ```python ` (and other languages when relevant). Keep snippets self-contained or clearly scoped; mention SDK, Pydantic, etc. as in the rest of the repo.
- **Best Practices**: Bullet or short paragraphs.
- **Documentation Links** (or **Links**): External references (e.g. OpenAI docs, Anthropic docs) at the end of a section or doc.

Preserve existing heading levels when inserting new sections.

## Links and paths

- **Same directory**: `./understanding_models.md`, `./learning_progression.md`.
- **From eng-dev-patterns to src**: `../src/embeddings_vector_search.py`, `../src/schema_driven_translation.py`.
- **From learning_progression**: It lives in `eng-dev-patterns/`, so “detailed documentation” links use `../eng-dev-patterns/prompt_engineering.md` when the path is from repo root, or `./prompt_engineering.md` when the path is from eng-dev-patterns. Check the file’s actual location; learning_progression is under eng-dev-patterns, so links to other eng-dev-patterns docs are `./filename.md`.
- Use markdown link form: `[link text](./path)` or `[link text](../src/file.py)`.

## Formatting conventions

- **Bold** for subsection labels: `**Description**:`, `**Key Concepts**:`, `**Example**:`.
- Emoji for doc links in README / progression: `📖` for detailed docs, `📚` for the learning progression guide.
- Bullet lists for concepts, practices, use cases, and example implementation items.
- Code: language-tagged fenced blocks; prefer real SDK/schema snippets that match the repo (e.g. OpenAI client, Pydantic, JSON Schema).

## Tone and content

- **Audience**: Engineers and learners building applications with LLMs and related services.
- **Tone**: Neutral, practical, instructional. Avoid marketing; name vendors and products when they illustrate the pattern (OpenAI, Anthropic, LangChain, etc.).
- **Translation**: In learning_progression, translation is the shared use case; each pattern section should include a **Translation Example** and, when applicable, an **Example** link to a script in `src/`.
- **Consistency**: When adding a new pattern, add or update the corresponding section in README and, if it’s part of the path, in learning_progression, and add or update the detailed doc so the three stay aligned.

## Summary

- **README**: Index with short pattern blocks and `📖` links to detailed docs.
- **learning_progression**: Ordered path with Description, Technology, Key Concepts, Translation Example, Example Implementation, and **Example** link to `../src/` when there is a script.
- **Detailed docs**: Long-form with Overview, Description/How It Works, subsections, code, Best Practices, and links.
- Use **bold** labels, relative links (`./` and `../src/`), and the same pattern-block structure so the documentation stays consistent and easy to navigate.
