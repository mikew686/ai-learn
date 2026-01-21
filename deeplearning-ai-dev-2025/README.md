# DeepLearning.AI Dev Convention 2025  
### A Software Engineer’s Overview of Themes, Trends, and Technical Directions  

Event link: https://ai-dev.deeplearning.ai/

The DeepLearning.AI Dev Convention 2025 brought together engineers, researchers, platform teams, and AI-tooling companies to examine the rapidly shifting technical landscape shaped by agentic systems, multimodal models, AI-assisted coding, and modern ML infrastructure. Across keynotes, panels, and technical talks, a central theme emerged: **AI capabilities are accelerating faster than the surrounding processes—product development, governance, evaluation, and human feedback loops—can adapt.**

---

## Keynote Themes: Acceleration, Iteration, and New Bottlenecks

**This keynote was delivered by Andrew Ng, Founder of DeepLearning.AI, Co-founder of Coursera, and Adjunct Professor at Stanford University.** His keynote framed the central questions of the convention by examining the rapid acceleration of AI capabilities, the disruption of traditional software-development workflows, and the widening gap between technical progress and public trust. Ng’s perspective set the foundation for the rest of the sessions, establishing rapid iteration, shifting bottlenecks, and the changing role of engineers as recurring themes.

Andrew Ng opened the convention with a clear message: **AI progress is not slowing down**. He presented data showing:

- The complexity of tasks AI systems can complete is **doubling every seven months**.
- For AI coding tasks, capability is doubling roughly every **70 days**.
- AI accelerates prototyping by **10×**, though production-grade software still only sees about **1.5×** gains (prototyping is now near-zero cost compared to production).

This shift has moved the bottleneck away from engineering output and into **product management and user feedback loops**, which have not yet adapted to this new level of velocity. Ng warned that while workers increasingly want AI to automate repetitive tasks (workers want AI to automate **46.1%** of tasks on average, primarily to free time for high-value work and remove drudgery), public trust in AI remains low—driven by outsized fears about existential risks—creating pressure for regulation that may outpace technical understanding and harm innovation, especially in open-source ecosystems.

### The New Development Loop: Explore First, Define Later

Ng outlined a fundamental shift in development methodology. The traditional loop of **Define → Build → Validate → Iterate** is being replaced by an AI-era approach: **Build many prototypes → Filter quickly → Validate with users → Scale winners**. Requirements increasingly emerge **after exploration**, not before, shifting product discovery from prediction to experimentation. Teams should generate **10–20 prototypes per cycle**, with an updated cultural guidance: **Move fast and be responsible.**

### Additional Bottlenecks: Evaluation and Data Strategy

Beyond PM constraints, Ng identified **evaluation as the new technical bottleneck**. Human evaluation does not scale with model output volume, requiring automated scoring, rubric-based evaluation, synthetic test generation, and new "Evaluation Ops" roles. Evaluation velocity now determines product velocity. He also emphasized that **data strategy is now a critical differentiator**, with AI value shifting from generic large datasets to domain-specific, expert-curated, and organization-specific operational data. Best-performing systems combine foundation models with high-quality proprietary data and golden evaluation sets.

### Cost Dynamics and Organizational Adaptation

Ng noted that inference costs are dropping faster than expected, and **engineering cost now often exceeds compute cost**—enabling more prototypes, more evaluation, and rapid iteration at lower risk. This economic shift is driving organizational changes: new roles are emerging (Data Quality Owner, Evaluation Engineer, AI Operations), and teams are restructuring toward cross-functional AI teams with ML engineers embedded within product teams. Engineers must upskill with new AI-era skills: prompt engineering patterns, data curation, evaluation design, debugging LLM-driven behavior, and safety/guardrail design. Engineers mastering hybrid AI–software skills become force multipliers.

### Additional keynote insight and observations

Although the keynote was structured around data and broad themes, the tone in the room reflected a deeper underlying tension about what this acceleration means for engineering teams. Ng emphasized that while AI lowers the barrier to writing code, it raises the bar for engineering judgment—developers must think more deeply about architecture, data flows, and product intent. He also noted rising mental fatigue among AI-assisted developers, reflecting insecurity and imposter syndrome as expectations outpace clarity. The audience reacted strongly to his comments on PM bottlenecks, with engineers nodding as he described how prototype velocity now outstrips product feedback cycles. The keynote set a tone of **cautious optimism**: capabilities are skyrocketing, but the human and organizational structures around them have not yet caught up.


📄 **[Detailed keynote summary →](./andrew_ng_keynote_summary.md)**
---

## Cross-Convention Themes

---

### 1. AI Coding & New Development Workflows

**This theme drew primarily from the panel “Software Development in the Age of AI,” featuring Andrew Ng (DeepLearning.AI), Malte Ubl (CTO, Vercel), Laurence Moroney (Director of AI, Arm), and Fabian Hedin (CTO, Lovable), moderated by Ryan Keenan of DeepLearning.AI.** The panel explored how AI-assisted coding is reshaping engineering roles, the emerging mental workload of AI development, and the rising involvement of PMs and non-technical contributors in the software process.

AI-assisted coding is reshaping responsibilities across engineering, product management, and non-technical roles. Multiple panelists warned that:

- AI coding is **mentally demanding** and can induce imposter syndrome.
- Rapid prototyping can **accelerate technical debt** without strong engineering discipline.
- Teams must maintain fundamentals in architecture, data modeling, and testing.
- PMs and subject-matter experts are increasingly creating prototypes themselves, blurring traditional boundaries.

Rather than replacing engineers, AI places **greater emphasis on foundational engineering skills**.

#### Expanding PM Roles, Rapid Prototyping, and Shifting Team Dynamics

As AI accelerates prototyping, the role of **product managers is expanding and shifting**. Multiple speakers noted that the traditional PM → engineer → feedback loop is breaking down. With AI tools, PMs can now generate interactive prototypes, wireframes, and even partial implementations themselves—often before engineers are consulted. Malte Ubl (Vercel) observed that PMs today are far more “development-adjacent,” using AI copilots to explore product directions independently. This leads to **more prototypes per unit time**, but also creates pressure on engineers to evaluate or rewrite concepts that may lack architectural grounding.

Andrew Ng emphasized that engineering velocity has increased so dramatically that **PM-to-engineer ratios are moving toward 1:1**, because engineers now build faster than PMs can learn from users. My conference impression reinforced this: PMs now face the hardest question—**what to build**—because the cost of building anything (at prototype scale) is near zero. Several panelists compared this to the dot-com era, where high iteration speeds produced a flood of ideas but not always clear product direction.

The panel also warned of overconfidence in AI-generated prototypes: because LLMs can produce functional demos quickly, teams may mistake “working code” for “validated product.” Laurence Moroney noted that poorly grounded prototypes can mask deep architectural flaws, while Fabian Hedin emphasized that iteration must focus on *ideas*, not just code. In practice, AI is pulling PMs closer to engineering and pushing engineers closer to product strategy—a convergence still poorly understood, even by those experimenting with it firsthand.

#### Landing.ai: Workflow-Integrated Document Extraction

A notable thread within this theme was the rising importance of **AI-driven document ingestion and transformation**, demonstrated by Landing.ai’s multimodal document-extraction system. Their technology focuses on converting complex documents—PDFs, forms, tables, scanned pages, and semi-structured materials—into clean, structured machine-readable data. Instead of relying solely on generic OCR or vision-language models, Landing.ai combines computer vision techniques with fine-tuned multimodal models capable of identifying fields, segments, contextual groupings, and inter-document relationships.

What sets Landing.ai apart is its **workflow-centric integration philosophy**: their extraction engine is designed to sit inside real enterprise ingestion pipelines, not as an isolated OCR tool. This makes it directly relevant to operational contexts like curriculum ingestion, content normalization, and semi-structured data processing—needs that align with Kiddom’s content operations. Their system reflects a broader movement: building **task-specific, workflow-aware AI systems** that automate tedious and error-prone data-transformation steps.

---

### 2. Agentic Systems & Long-Horizon Task Automation

**This section is informed by multiple talks, including Nicholas Clegg’s “Model-Driven Agents with AWS Strands,” Kay Zhu’s discussion on Genspark’s super-agent architecture, and Ori Goshen’s analysis of reliability bottlenecks from AI21 Labs.** Together, these speakers explored the evolution from simple tool-calling LLMs to robust agentic systems capable of planning, delegation, and long-horizon task execution.

Agentic systems were one of the strongest engineering trends across the convention. Talks highlighted:

- Multi-agent orchestration  
- Model-driven control loops  
- Tool-based planning and delegation  
- Context routing and memory handling  
- Dynamic task decomposition  

#### AWS Strands as a Model-Driven Architecture

A central example was **AWS Strands**, an open-source SDK for building AI agents that originated inside AWS product teams (Amazon Q, AWS Glue, VPC Reachability Analyzer). Strands takes a **model-first** approach: developers define a model, system prompt, and tools, and the agent autonomously performs reasoning, planning, and multi-step execution. It supports:

- multiple collaborating agents  
- arbitrary tool creation using docstrings for agent understanding  
- dynamic task delegation  
- local-to-production scalability  

Strands reflects a movement away from brittle, manually orchestrated pipelines toward **LLM-directed execution**, where models decide what steps are required to accomplish a task. This shift mirrors how human operators behave, replacing hand-written flows with adaptive reasoning loops.

#### Complementary View: Redis LangCache

Redis introduced **LangCache**, highlighting how semantic caching supports agent workflows by reducing redundant token usage, decreasing latency, and improving response times for multi-call agent loops. In agentic settings, repeated queries for similar information can cause cascading token costs—LangCache mitigates this through vector similarity and probabilistic reuse.

---

### 3. Context Engineering as a Core Skill

**Insights in this section draw from Nitin Kanukolanu’s Redis talk on semantic caching, Jacky Liang’s TigerData session on Postgres hybrid retrieval, and David Loker’s CodeRabbit discussion on context-aware code review.** Each speaker highlighted the importance of controlling what goes into the model’s context window—emphasizing token efficiency, precision, retrieval correctness, and infrastructure-level optimization.

Context windows, token efficiency, and retrieval precision now shape both performance and cost in agent workflows. Context engineering is evolving into a discipline concerned with **how information flows through agents**, not just which documents get retrieved.

📄 **[Detailed context engineering highlights →](./context_engineering_highlights.md)**

Core topics included:

- Semantic caching  
- RAG assembly and context shaping  
- Hybrid retrieval (lexical + vector)  
- Knowledge-graph-assisted routing  
- Precision/recall management  
- Eliminating redundant LLM calls in multi-step loops  

#### Redis LangCache: Probabilistic, Semantic Caching for Efficiency

Redis’s **LangCache** provides semantic caching through vector embeddings, enabling intelligent reuse of past LLM results when new queries are sufficiently similar. This reduces cost, latency, and unnecessary inference calls. Because LangCache is **probabilistic**, developers must evaluate it with metrics like:

- precision  
- recall  
- F1 score  
- cache hit rate  
- false-positive impact  

The talk emphasized treating semantic caches **as machine learning components**, not deterministic lookup tables.

#### Postgres pg_textsearch (TigerData): Hybrid Search for Cleaner Context

TigerData demonstrated **pg_textsearch**, a Postgres extension that combines full-text search and vector similarity directly in the database. This hybrid retrieval approach:

- reduces hallucinations  
- improves contextual relevance  
- removes architecture complexity  
- avoids standalone vector databases  

By keeping retrieval inside Postgres, developers maintain tighter control over context assembly and significantly reduce the “garbage in, garbage out” problem that plagues naive RAG implementations.

---

### 4. Model Scale vs. Model Specialization

A major trend discussed across vendors was the movement from frontier-scale general models toward **domain-specific, task-optimized, small or mid-sized models**. Examples included:

- Google’s domain-tuned medical and clinical models outperforming larger general LLMs  
- AWS advocating for model-driven planning instead of scale-first thinking through Strands  
- Arm’s push for “Small AI”: efficient, on-device local inference  

The ecosystem is shifting toward **fit-for-purpose** architectures.

#### Baseten & OpenRouter: Routing and Serving for Specialist Models

Baseten and OpenRouter exemplified practical approaches for adopting specialized models:

- **Baseten** enables deployment and scaling of fine-tuned or distilled models specialized for narrow tasks.  
- **OpenRouter** enables routing across dozens of open and commercial LLMs, letting teams pick the best model per task based on accuracy, domain strength, latency, or cost.

These platforms make model specialization **operationally feasible**, encouraging developers to treat models as interchangeable components.

---

### 5. Governance, Public Perception, and Open Source

The governance panel highlighted the widening philosophical divide in the field:

- **Miriam Vogel (President & CEO, EqualAI)** pushed for stronger legal frameworks and enterprise accountability.  
- **Andrew Ng (Founder, DeepLearning.AI)** warned that excessive regulation driven by hype could damage open-source progress and restrict innovation.  
- **Nicholas Thompson (CEO, The Atlantic)** moderated the discussion.

### Additional governance insight and observations

The tone of this panel was more cautious than the technical sessions. Vogel's arguments aligned with enterprise risk frameworks and policy-first approaches, while Ng presented a builder-centric counterpoint. He criticized fear-driven narratives such as "AI might cause human extinction" as harmful distractions that push regulators toward overreach. Ng drew analogies to cultural resistance seen in Hollywood, where fears of technological replacement create friction and overcorrection. I observed that practitioners in the audience clearly sided with Ng's stance: the immediate danger they perceived was **overregulation**, not runaway AI. The panel underscored a growing divide between policy discourse and engineering reality.

---

## Overall Takeaway: The Field Is Maturing Unevenly

The convention emphasized that while AI capabilities are accelerating rapidly, the surrounding ecosystem—developer tools, evaluation frameworks, governance structures, product processes—lags behind. This creates a landscape where:

- **Prototypes are cheap; production is hard.**  
- **Agents are powerful but fragile.**  
- **Evaluation and context engineering are now core disciplines.**  
- **Open-source innovation is essential yet increasingly politicized.**  
- Engineers must think in **full-stack terms** across data, retrieval, UX, orchestration, safety, and model behavior.

Across sessions, the message was clear: **software engineering is being redefined, not replaced**. Teams that embrace disciplined engineering fundamentals, iterative product feedback loops, and thoughtful model/tool selection will be best positioned in this new era of AI-driven development.

---

## License

Code samples are licensed under the [MIT License](LICENSE).

Documentation, notes, and written content are licensed under [CC BY 4.0](docs/LICENSE).
