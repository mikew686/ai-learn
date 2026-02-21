# Agent guide: eng-dev-patterns documentation style

This directory holds **pattern documentation** for AI engineering: model selection, prompt design, tool use, structured output, RAG, and related topics. The docs are written for humans learning or referencing patterns; this file describes the **documentation style** so agents and maintainers can add or edit content consistently.

## Document roles

- **README.md** – Entry point and index. Short overview of each pattern with **Description**, **Key Concepts** (or **Key Techniques** / **Key Approaches**), **Best Practices**, **Use Cases**, and a link to the detailed doc: `📖 [Detailed Documentation](./filename.md)`. Include a **Recommended order** note under Learning Progression that points to learning_progression.md and states: core principles first (Understanding Models, Prompts, Schemas, Tools, Schema-Driven Inference, Embeddings), then tying together patterns, then production patterns.



- **learning_progression.md** – Ordered learning path. **Preferred learning order**: (1) Understanding Models, (2) Prompt Engineering, (3) Structured Output, (4) Function Calling / Tool Use, (5) Schema-Driven Inference, (6) Embeddings / Vector Search, then (7+) patterns that tie them together (Few-Shot, RAG, Chain-of-Thought, etc.), then production patterns (Streaming, Caching, Memory, Guardrails, Agents, Orchestration, Evaluation, Advanced). Schema-Driven Inference comes before Embeddings as it is more fundamental to core inference flow. Each pattern section uses the fixed block: **Description**, **Technology**, **Key Concepts**, **Translation Example**, **Example Implementation** (bullets), and when applicable **Example** (link to `../scripts/examples/script.py`). Only link to **Example** when a corresponding script exists in `scripts/examples/`; use the actual filename (e.g. `system_prompt_example.py`, `tool_use_patterns.py`, `embeddings_vector_search.py`, `schema_driven_translation.py`, `language_assessment_structured_output.py`, `understand_llm_models.py`). Uses translation as the running use case so patterns build on each other.
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
7. **Example** (learning_progression only, when there is a script in `scripts/examples/`): The **primary** example — one script that best demonstrates the pattern. Format: `**Example**: [script_name.py](../scripts/examples/script_name.py)`. Use the actual filename that exists in the repo. If no script exists for the pattern, omit the **Example** line.
8. **Related** (learning_progression only, optional): Other `scripts/examples/` scripts that use this pattern as a **secondary** technology (the pattern appears in them but is not the main focus). Format: `**Related**: [script_a.py](../scripts/examples/script_a.py), [script_b.py](../scripts/examples/script_b.py)`. Omit if there are no such scripts.
9. **Best Practices** (README often; detailed docs): Bullet list.
10. **Use Cases** (README often): Bullet list.
11. **Related Patterns** (optional): Short list with links or names.
12. **Detailed doc link** (README): `📖 [Detailed Documentation](./filename.md)` at the top of the section.

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

## Requirements for future learning docs

When adding or revising **detailed docs** (long-form pattern or learning documentation), include the following so new content stays consistent and discoverable:

1. **Related patterns** (required for pattern docs)
   - Add a **Related patterns** section with links to other eng-dev-patterns docs that this pattern builds on, extends, or combines with.
   - Use relative links: `[link text](./filename.md)` for docs in this directory, `[link text](../scripts/examples/script.py)` for example scripts.
   - Example: *"**Schema-Driven Inference**: Tool schema descriptions act as implicit prompts; see [schema_driven_inference.md](./schema_driven_inference.md)."*

2. **Learning path** (recommended)
   - State where this pattern sits in the learning order (e.g. "Pattern 3 in the [learning progression](./learning_progression.md)").
   - Mention prerequisites (e.g. "Learn after Prompt Engineering and Structured Output") and, when applicable, link to an example script in `scripts/examples/`.

3. **Practical technologies** (when relevant)
   - For patterns that involve infrastructure, libraries, or tooling, add a **Practical technologies** (or **Popular solutions**) subsection: concrete options (e.g. SQLite, FAISS, Pinecone for vector storage; Pydantic, Outlines, Zod for structured output; LangChain, Jinja2 for prompts).
   - For fundamental patterns (e.g. vector search, RAG, tool use), include a short table or list of common choices (databases, SDKs, frameworks) with one-line notes so learners know what to use when.

4. **Best practices** and **Documentation links**
   - Keep **Best practices** as bullet or short paragraphs.
   - End with **Documentation links** (or **Links**) to official provider docs, specs, and community resources.

5. **Cross-updates**
   - When adding a new detailed doc, add or update the corresponding section in **README.md** and, if part of the path, in **learning_progression.md**.
   - Add a link to the new doc from **Related patterns** in any existing doc that logically relates to it.

This keeps the doc set navigable, ties each pattern into the progression, and gives learners concrete technologies and a clear path.

## Links and paths

- **Same directory**: `./understanding_models.md`, `./learning_progression.md`.
- **From eng-dev-patterns to examples**: `../scripts/examples/embeddings_vector_search.py`, `../scripts/examples/schema_driven_translation.py`.
- **From learning_progression**: It lives in `eng-dev-patterns/`, so “detailed documentation” links use `../eng-dev-patterns/prompt_engineering.md` when the path is from repo root, or `./prompt_engineering.md` when the path is from eng-dev-patterns. Check the file’s actual location; learning_progression is under eng-dev-patterns, so links to other eng-dev-patterns docs are `./filename.md`.
- Use markdown link form: `[link text](./path)` or `[link text](../scripts/examples/file.py)`.

## Formatting conventions

- **Bold** for subsection labels: `**Description**:`, `**Key Concepts**:`, `**Example**:`.
- Emoji for doc links in README / progression: `📖` for detailed docs, `📚` for the learning progression guide.
- Bullet lists for concepts, practices, use cases, and example implementation items.
- Code: language-tagged fenced blocks; prefer real SDK/schema snippets that match the repo (e.g. OpenAI client, Pydantic, JSON Schema).

## Tone and content

- **Audience**: Engineers and learners building applications with LLMs and related services.
- **Tone**: Neutral, practical, instructional. Avoid marketing; name vendors and products when they illustrate the pattern (OpenAI, Anthropic, LangChain, etc.).
- **Translation**: In learning_progression, translation is the shared use case; each pattern section should include a **Translation Example** and, when applicable, an **Example** link to a script in `scripts/examples/`.
- **Consistency**: When adding a new pattern, add or update the corresponding section in README and, if it’s part of the path, in learning_progression, and add or update the detailed doc so the three stay aligned.

## Learning progression best practices

- **Core principles before tying together**: The learning progression order must cover (1) Understanding Models (how to find/call models), (2) Prompts, (3) Schemas (Structured Output), (4) Tools (Function Calling), (5) Schema-Driven Inference before Embeddings (Schema-Driven is more fundamental). Then (6) Embeddings, then patterns that depend on vector search (Few-Shot with retrieval, RAG), then Chain-of-Thought. Do not remove patterns when reorganizing; renumber and reorder as needed.
- **Example links must point to existing files**: In learning_progression.md, **Example** (or **Example Files**) must reference only scripts that exist under `scripts/examples/`. Verify filenames (e.g. `system_prompt_example.py`, `tool_use_patterns.py`, `schema_driven_translation.py`, `embeddings_vector_search.py`, `language_assessment_structured_output.py`, `understand_llm_models.py`). If a pattern has no example script, omit the Example line; use **Example Implementation** bullets only.
- **Example vs. Related**: **Example** is the single primary script that best demonstrates the pattern. **Related** lists any other `scripts/examples/` scripts that use this pattern as a secondary technology (pattern appears but is not the main focus). When multiple scripts use the pattern, choose one as Example and list the rest under Related.
- **Preferred order documented in README**: README should state the recommended learning order (core → tying together → production) and link to learning_progression.md for the full sequence and example links.

## Summary

- **README**: Index with short pattern blocks and `📖` links to detailed docs; include recommended learning order under Learning Progression.
- **learning_progression**: Preferred order: Understanding Models → Prompts → Schemas → Tools → Schema-Driven Inference → Embeddings → tying together → production. Each pattern: Description, Technology, Key Concepts, Translation Example, Example Implementation; **Example** link only when script exists in `scripts/examples/`.
- **Detailed docs**: Long-form with Overview, Description/How It Works, subsections, code, Best Practices, and Documentation links. For **future learning docs**, also include: **Related patterns** (with links), **Learning path** (position in progression + prerequisites), and **Practical technologies** (when relevant).
- Use **bold** labels, relative links (`./` and `../scripts/examples/`), and the same pattern-block structure so the documentation stays consistent and easy to navigate.
