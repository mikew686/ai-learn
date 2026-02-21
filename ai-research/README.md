# AI Research Notes

Self-learning notes to deepen my understanding of LLMs and AI development. Any mistakes are mine or my LLMs. :)

---

## AI discussion

Reflection and practice on using AI in a way that strengthens thinking.

* [accomplishment_hallucination_article_summary.md](accomplishment_hallucination_article_summary.md) — Summary of “Accomplishment hallucination: when the tool uses you”: when speed feels like competence and output feels like accomplishment without the underlying thinking.
* [tools_for_thoughtful_ai_use.md](tools_for_thoughtful_ai_use.md) — Practical tools (structure, reflection, verification) so fluency supports thinking rather than replacing it; goal: use AI in a way that strengthens your thinking.

---

## Overview docs (`how_*`)

High-level explanations of how things work:

* [how_modern_llms_work.md](how_modern_llms_work.md) — How modern LLMs work: next-token prediction, scale, emergence, inference loop, tool use, and a clean mental model.
* [how_llms_write_code.md](how_llms_write_code.md) — Why code generation is pattern matching, how scale changes behavior, and how to get strong results.

---

## Prompt-engineering and practice

Foundations for building with LLMs: prompts, tools, and schema.

* [foundations_prompt_engineering.md](foundations_prompt_engineering.md) — What happens to system/user prompts, roles, context window, and how to steer the model’s probability.
* [foundations_tool_calling.md](foundations_tool_calling.md) — Tool calling as next-token prediction + external execution + context reintegration, with examples.
* [foundations_structured_output.md](foundations_structured_output.md) — Getting machine-parseable, schema-conforming output from LLMs for workflows, APIs, and tool routing.
* [foundations_behavioral_configuration_artifacts.md](foundations_behavioral_configuration_artifacts.md) — Repo-scoped AI playbooks: rules, architecture guardrails, skills, and conversational gating via context conditioning.

---

## Theory and mechanics

More technical / conceptual depth: architecture, embeddings, reasoning.

* [foundations_from_neural_nets_to_llms.md](foundations_from_neural_nets_to_llms.md) — From neural nets and backprop to RNNs, attention, and the transformer; evolution toward LLMs.
* [foundations_from_llms_to_gpts.md](foundations_from_llms_to_gpts.md) — How GPT-style models work: autoregressive LM, tokenization, embeddings, attention, MLP, residual stream, scaling.
* [foundations_embeddings_and_vector_search.md](foundations_embeddings_and_vector_search.md) — Embeddings as a coordinate system for meaning; vector search, ANN, and RAG.
* [foundations_reasoning.md](foundations_reasoning.md) — How reasoning emerges in LLMs from multi-step generation and “show your work” prompting, not symbolic logic.
