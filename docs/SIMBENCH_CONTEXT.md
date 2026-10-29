# SimBench: Project Context for Cursor

Status as of October 23, 2026. **Living document.** Update it as decisions are made.

> **Read this first.** The **research question is NOT yet decided** (see Section 5). Milestones V0 to V4 are question-agnostic and can be built now. Do not hard-code assumptions that only one research question would need.
>
> Status tags: `[DECIDED]` `[PROPOSED]` `[OPEN]` `[STRETCH]`

---

## 1. Project in one paragraph

**SimBench** is a web-based framework for creating **simulated users (SimUsers)**, running multi-turn conversations between a SimUser and an AI **agent under test (AUT)** inside a controlled mock environment, and **measuring how reliable and realistic that kind of testing is**.

A SimUser is a simulated person with a goal, limited knowledge, memory, constraints, and a communication style, who chooses its next message based on what the agent says. The AUT starts as a small customer-support agent we control (orders, refund policy, tools), so we can create versions that differ by construction.

The project is a **research project with a web app as its instrument**, built as a portfolio piece for SWE and AI-startup roles.

**Pitch:** SimBench is a web-based framework for creating stateful, goal-driven simulated users, testing AI agents through realistic interactions, and experimentally measuring simulation fidelity and evaluation reliability against simple prompt-based baselines.

---

## 2. Goals and framing

- `[DECIDED]` **Career goal:** stepping stone to SWE or AI-startup roles (including early / founding-engineer roles), combining production engineering, LLM/agent systems, and rigorous evaluation.
- `[DECIDED]` **Framing:** research project. The study is what gets judged: falsifiable question, protocol fixed before confirmatory runs, honest statistics, write-up that includes failures and limitations.
- `[DECIDED]` **Resume positioning:** list as open-source / independent research (creator and maintainer), not a founder title, unless real users or a pilot appear. Do not inflate claims.
- `[DECIDED]` **Not a startup right now.** Commercial simulation and eval platforms already exist. If anything, the defensible angle is measuring simulator fidelity and statistical reliability. Revisit only if real usage appears.
- `[DECIDED]` **"Shippable" means:** public repo with tests and CI, reproducible small experiment, written report, hosted demo in replay mode by default. Not a multi-tenant SaaS.

---

## 3. Source papers and what we borrow

Neither paper states formal hypotheses. Both test claims by comparing a baseline with a variant on a measurable outcome. Our study should have the same shape.

### RealUserSim (Zhu et al., Salesforce AI Research, preprint 2026)

- Grounds user simulators in real behavior from WildChat.
- **Executable Persona Manual:** Command + Example pairs (e.g. "use lowercase" with quoted samples).
- **PT3 (Paired Trajectory Turing Test):** an LLM judge compares a real and a simulated transcript on five dimensions.
- Anti-leakage cleaning of profile examples.
- Evaluated on tau-bench. Reports a "Formalism Ceiling" (unprompted simulators are overly polished) and "Directive Amplification" (one-line directives like "you are angry" behave very differently across simulator models).
- Reports a small mean task-success drop (about 3%) with persona injection, alongside large run-to-run variance (about 8 points mean range across 3 runs on 50 airline tasks).

**SimBench borrows:** Grounded SimUser variant (Command + Example profiles); deterministic behavior markers (message length, stage directions, frustration words); failure taxonomy (information loss, agent misinterpretation, compliance failure); dev/test separation and anti-leakage discipline.

### SimUSER (Bougie and Watanabe, ACL 2025 Industry)

- LLM agents as user proxies for recommender systems.
- Persona matching via a self-consistency check.
- Modules: persona, perception, memory (episodic memory with embedding retrieval plus a knowledge-graph memory), brain (multi-round decisions, reflection, causal action refinement).
- Ablations show memory and persona help human-likeness and rating prediction. Simulated A/B results correlated with real ones.

**SimBench borrows:** Stateful SimUser variant (persona plus memory/state kept separate from chat history, optional reflection); ablation-style experimental design; optional embedding-based retrieval memory.

### Observations (opportunities, not attacks)

- RealUserSim's headline persona effect (about 3%) is smaller than its own re-run noise (about 8 points). A careful study of *how many runs are needed* is a legitimate contribution.
- In PT3 the simulator, agent, and judge are all GPT-4o, and the judge does not appear to be validated against human ratings. Use a judge from a different model family and validate against human labels on a subset.
- WildChat is human-to-chatbot conversation, not customer-support data. Any "grounded" claim must state this scope. Check dataset license and terms before use.
- Several useful metrics need no LLM judge (leakage, fact coverage, task success via database-state check, behavior markers).

---

## 4. Decisions log

| Topic | Status | Decision / note |
|---|---|---|
| Project framing | `[DECIDED]` | Research project; web app is an instrument. Study quality over UI polish. |
| Architecture | `[DECIDED]` | Python core library (simulation, evaluation, adapters) + thin FastAPI layer + React/TypeScript UI. Core first, so a runnable study exists even if the web layer slips. |
| Agent loop | `[DECIDED]` | Hand-write the loop for V0 to V3. Later add a LangChain/LangGraph agent as one AUT to prove the adapter interface is agent-agnostic. |
| Storage | `[DECIDED]` | SQLite for local prototype; Postgres only if deployment needs it. |
| Demo mode | `[DECIDED]` | Replay mode by default (saved transcripts and results). Live runs only behind turn caps, global daily budget, per-IP rate limits, server-side keys. |
| SimUser variants | `[PROPOSED]` | Prompt baseline, Persona, Stateful, Grounded (Grounded is riskiest; stretch if data, license, or time is a problem). |
| Agent versions (known differences) | `[PROPOSED]` | A: basic prompt. B: better prompt. C: B plus tools. D: C with a deliberately injected flaw. B-prime: identical copy of B (A/A control). |
| Primary metric | `[PROPOSED]` | Deterministic task success via final database-state check, plus turns, tokens, latency, cost. |
| Judge | `[PROPOSED]` | LLM judge only for subjective qualities (realism). Different model family from simulator. Validate on a human-rated subset. |
| **Primary research question** | `[OPEN]` | **Not frozen.** Leaning toward RQ1 (Section 5); alternatives still under consideration. |
| Reference for "realistic" | `[OPEN]` | Needed only for fidelity claims. Options: WildChat-derived profiles, small human ratings, classifier test, or skip realism claims and focus on reliability. |
| Human rating work | `[OPEN]` | Include or not (affects RQ4 and judge validation feasibility). |
| Models and budget | `[DECIDED]` for now | Owner wants no surprise bill. Use the Gemini API free tier (default `gemini-3.5-flash-lite`) and never link a billing account to that project. The earlier $20 to $50 / $50 to $100 planning figures apply only if the owner later chooses a paid provider. Free-tier limits are per model, per minute and per day, are not published exactly, and are not guaranteed, so the V5 study must be sized to the quota or move to a local model. See the free-tier notes in Section 12. |
| Timeline and owner background | `[OPEN]` | Needed to scope. Ask the owner. |

---

## 5. Research question: STILL BEING DECIDED

> `[OPEN]` **The primary research question has not been chosen.** The items below are candidates, not commitments.
>
> **Assistant instruction:** do not write analysis code, metrics, or experiment configs that depend on a single question being final. Keep scenarios, variants, agents, and metrics configurable.

### Currently favored (not frozen)

**RQ1:** Can SimUser-based evaluation reliably detect known differences between agent versions, and does SimUser design change how many runs that takes?

- **"Run"** = one complete simulated conversation (one scenario, one SimUser variant, one agent version) from first message to termination, then scored. Conversations are random, so one run says little; a reliable comparison needs many.
- **"Known differences"** = agent versions built to differ by construction (A, B, C, D, B-prime), so the true ordering is known in advance.

### How reliability would be measured under RQ1

- **Detection rate (power):** if C truly beats A, how often does an N-run evaluation show it? Plot detection rate against N.
- **False-positive rate:** compare identical agents (B vs B-prime). A sound method flags a difference only about as often as the chosen significance level.
- **Ranking stability:** split runs into independent halves, rank agents in each, measure agreement (e.g. Kendall's tau) and agreement with the known ordering.
- **Runs needed:** smallest N reaching a target detection rate (e.g. 80%) for a given gap size, per SimUser variant, with confidence intervals.
- **Separation vs noise:** effect size between agents. A cooperative simulator may let every agent pass (ceiling effect) and hide differences.
- **Method:** collect a pool (e.g. 10 scenarios x 4 agents x 3 variants x 5 repeats), then bootstrap-resample to estimate power curves cheaply. Resample scenarios as well as repeats, because runs on the same scenario are not independent. Pilot with a small pool first and check prices.

### Alternative candidate questions

| ID | Question | Notes |
|---|---|---|
| RQ2 | Do stateful SimUsers (state separate from chat history) leak withheld facts and drift from goals less than prompt-only SimUsers? | Mostly deterministic metrics. Good secondary. |
| RQ3 | Does a structured behavior spec reduce behavior variation across simulator models vs a one-line directive? | Extends RealUserSim's Directive Amplification observation. |
| RQ4 | Do grounded personas beat hand-written ones on human-rated realism in customer support? | Most interesting, hardest: needs real-user data and enough human ratings. Stretch. |
| RQ5 | How well do LLM judges agree with humans on realism; is there same-family bias? | Best as a validation component. |
| A | Does a cooperative default simulator inflate agent scores vs less cooperative users? | Replication-flavored; needs proper statistics. |
| B | Front-loaders vs drip-feeders (information-sharing style): effect on agent errors | Clean single-variable causal design; deterministic. |
| C | Does simulator model family bias agent scores (simulator x agent grid)? | Simple to run and analyze. |
| D | Where does outcome variance come from (scenario, persona, simulator model, sampling) and how many runs are needed? | Statistics-forward; close cousin of RQ1. |
| E | Can a classifier distinguish simulated from real user messages? | No judge or human labels; measures surface/style realism only. |
| F | How sensitive are outcomes to paraphrasing the persona prompt? | Cheap robustness study. |
| G | Which SimUser design surfaces the most distinct agent failure modes per 100 conversations? | Needs a fixed taxonomy and ideally a second rater. |

### Research discipline

- `[DECIDED]` Building infrastructure (V0 to V4) before freezing the question is fine and expected.
- `[DECIDED]` Before confirmatory runs (V5), freeze in a dated file in the repo: question, hypotheses (including what result would count against them), agent versions, SimUser variants, scenarios, metrics, statistical plan, run counts, and significance threshold.
- `[DECIDED]` Label pilot runs "exploratory" and frozen-protocol runs "confirmatory." Do not choose the question after seeing which result looked significant.
- `[DECIDED]` Split scenarios into a dev set (for prompt tuning) and a held-out test set untouched until the end.
- `[DECIDED]` Null or negative results are acceptable and reportable.

---

## 6. SimUser variants (conditions to compare)

| Variant | What it contains | Source idea |
|---|---|---|
| V-Prompt (baseline) | One-line role-play, e.g. "act like an angry customer". | Standard practice; Directive Amplification risk. |
| V-Persona | Explicit goal, knowledge, constraints, communication style. | Structured spec. |
| V-Stateful | Persona plus maintained state (facts known, revealed, goal progress, memory) kept separate from chat history; optional reflection. | SimUSER memory and reflection. |
| V-Grounded `[STRETCH]` | Persona built from Command + Example pairs extracted from real user messages (e.g. WildChat), with anti-leakage cleaning. Check license first; state domain mismatch. | RealUserSim Executable Persona Manual. |

**SimUser spec fields:** goal; knowledge (what it knows and does not know); memory; style/personality (behavioral guidance, not just an emotion label); constraints (facts it must not know, details it will not volunteer, actions it will not take); current state; decision maker (rules and/or LLM that picks the next message).

---

## 7. Metrics

| Metric | How | Type |
|---|---|---|
| Task success | Final database-state check in the mock environment (e.g. correct refund issued). | Deterministic |
| Goal adherence | Does the SimUser keep pursuing its goal (via goal-progress state). | Mostly deterministic |
| Knowledge leakage | Does the SimUser reveal facts it should withhold, or use facts it should not know. | Deterministic plus spot checks |
| Fact coverage / information loss | Did the SimUser convey every required fact. | Deterministic |
| Memory consistency | Remembers prior events without inventing them. | Deterministic plus judge |
| Behavior markers | Message length, stage directions (asterisk actions), frustration words, politeness markers. | Deterministic |
| Operational | Turns, latency, tokens, estimated cost, invalid tool calls, repeated information, termination reason. | Deterministic |
| Behavioral realism | Rubric LLM judge on full trajectory (not per turn), different model family, validated on a human-rated subset. | Judge (validated) |

Watch for metrics that can be gamed. Never present illustrative scores as real results.

---

## 8. Architecture

Flow: `React/TypeScript UI -> FastAPI backend -> Simulation Orchestrator -> SimUser Engine <-> Agent Under Test <-> Environment/tools`. The database persists configs, scenarios, runs, transcripts, costs, and scores.

| Component | Responsibility |
|---|---|
| Frontend (React + TypeScript) | Create scenarios and SimUsers, configure and launch runs, live transcript viewer, compare results with confidence intervals, replay demo, (later) blind side-by-side annotation tool. |
| API (FastAPI + Pydantic) | `POST /simulations`, `POST /simulations/{id}/run`, `GET /simulations/{id}`, `GET /simulations/{id}/results`; SSE or WebSocket streaming for live transcripts. |
| Simulation Orchestrator | Turn order, shared state, stopping conditions, retries, errors, lifecycle, budgets, concurrency. |
| SimUser Engine | Goal, knowledge, memory, style, constraints, state, next-message decision. |
| AUT + adapters | Standard send-message / receive-response contract. Adapters: internal hand-written agent, LangChain/LangGraph agent, external HTTP endpoint, local model. Must not depend on one provider. |
| Environment / tools | Mock orders, refund rules, tickets; tools `lookup_order`, `issue_refund`, `create_ticket`; validation and permissions. |
| Evaluation Engine | Deterministic metrics plus rubric judge; stores scores with evidence. |
| Database | SimUser configs, scenarios, runs, messages, token/cost counts, scores, human ratings, experiment protocol versions. |

### Suggested repository layout (adjust freely)

```
simbench/
  core/          # pure Python library: simulation, simusers, agents, env, eval, llm adapters
  api/           # FastAPI app, schemas, routes, SSE
  web/           # React + TypeScript UI
  experiments/   # frozen protocols, configs, analysis scripts/notebooks
  data/          # scenarios (dev/test split), saved transcripts for replay mode
  tests/         # pytest, deterministic fixtures, mock LLM
  docs/          # this context doc, README, report draft
```

### Stack

- Frontend: React + TypeScript. Backend: Python, FastAPI, Pydantic. Tests: pytest with deterministic fixtures and a record/replay or mock LLM client.
- DB: SQLite first, Postgres if useful. LLM access behind a small provider interface; keys server-side only; cheap model for early SimUser iterations.
- Deployment: keep it simple. Avoid queues, distributed workers, or heavy infrastructure until the basic run flow needs them.
- **LangChain relationship:** LangChain/LangGraph *build* agents; SimBench *tests* agents. They are complementary. Hand-write the core loop first for transparency and reproducibility, then wrap a LangGraph agent as an AUT.

---

## 9. Milestones (build while learning)

Each version must be small, working, tested, and understood before moving on. Start with one SimUser, one agent, one scenario, one completed transcript. Do not build every feature up front. V0 to V4 are question-agnostic.

### V0: No AI (question-agnostic)
- **Build:** repo and README with goal and V0 scope. One customer-support scenario with a deterministic fake user and fake agent. A run function returning a structured transcript (Pydantic schema). One FastAPI endpoint. A React page with a "Run simulation" button and transcript viewer. Optional SQLite persistence.
- **Learn:** request/response flow, Python project setup, typing, data models, pytest, persistence basics.
- **Done when:** one command starts the app; a run produces a transcript; tests cover turn order and termination.

### V1: First LLM call (question-agnostic)
- **Build:** `LLMClient` interface with a provider adapter. Replace the fake user with an LLM-driven SimUser. Structured output parsing. Log tokens, latency, and estimated cost per run. Retries and timeouts. Mock or record/replay LLM for tests.
- **Learn:** API keys and server-side secrets, prompts, structured outputs, tokens and context windows, error handling, rate limits.
- **Done when:** a real LLM-vs-fake-agent conversation completes, is logged with cost, and tests pass without network calls.

### V2: Stateful SimUser (question-agnostic)
- **Build:** SimUser spec (goal, knowledge, constraints, style, state). State kept separate from chat history. Fact tracking (known, revealed, withheld). Deterministic leakage and goal-adherence checks. Optional: embedding-based memory retrieval and a post-conversation reflection step (SimUSER-style).
- **Learn:** context management, state vs history, prompt/spec design, unit-testing nondeterministic systems, (optional) embeddings and retrieval.
- **Done when:** a SimUser withholds what it should, reveals on request, and the checks flag deliberate violations in test fixtures.

### V3: Tools and environment (question-agnostic)
- **Build:** mock orders database and refund policy. Agent tools with argument validation and permissions. Agent loop with step limits. Agent versions A, B, C, D, B-prime. Task success via database-state check.
- **Learn:** tool calling, agent loops, validation, stop conditions, designing verifiable environments.
- **Done when:** each agent version runs end to end; success is computed deterministically; the injected flaw in D is detectable.

### V4: Orchestration and adapters (question-agnostic)
- **Build:** run manager with batch runs, manual budget confirmation, async concurrency, retries, termination reasons, run config hashing and seeds, idempotent re-runs. Live transcript streaming (SSE/WebSocket). Generic HTTP agent adapter and a LangChain/LangGraph agent as one AUT.
- **Learn:** async Python, job lifecycle, streaming, reproducibility, interface design, integrating third-party agent frameworks.
- **Done when:** a batch executes within budget, is reproducible from stored configs, and a LangGraph agent is tested through the same interface.

### V5: Experiments and evaluation (REQUIRES a frozen research question)
- **Build:** scenario set with dev/test split. Pilot runs (exploratory). Freeze protocol file. Confirmatory runs. Metrics from Section 7. Bootstrap confidence intervals, power curves, A/A false-positive check. Judge validation against a human-rated subset if included. Analysis scripts and report draft.
- **Learn:** experimental design, baselines, held-out data, repeated runs, bootstrap and power analysis, judge calibration, metric gaming, scientific writing.
- **Done when:** protocol file is dated before confirmatory runs; results (including failures and costs) are reproducible from the repo.

### V6: Polish and deploy
- **Build:** dashboard with confidence intervals, run history, replay-mode demo, optional annotation tool, cost caps and rate limits, auth only if needed, Docker, CI, documentation, public deployment, write-up.
- **Learn:** deployment, CI, security basics, abuse and cost protection, documentation, communicating results.
- **Done when:** public demo works in replay mode; a visitor can reproduce a small experiment from the README; limitations are documented.

---

## 10. Cost controls and scope guardrails

### Cost controls
- Start with 5 to 10 short simulations, not hundreds. Pilot before any large pool.
- Track input/output tokens, latency, and estimated cost per run. Set turn limits, token budgets, and manual confirmation before batches.
- Use deterministic checks wherever possible; judge a full trajectory rather than every turn. Use cheaper models for early SimUser iterations.
- Check current provider prices before experiments. The $20 to $50 and $50 to $100 figures are planning estimates, not quotes. Large benchmark runs can cost much more.

### Scope guardrails (decided)
- No huge multi-agent platform. One user + one agent is enough for the first working version.
- Do not assume an LLM judge is correct; validate and document limits.
- Do not call a SimUser a "digital twin"; call it a simulation and measure fidelity.
- Use public/licensed data or synthetic scenarios. No private conversations or personal data without authorization. Verify dataset licenses.
- Keep provider keys server-side; add rate and cost limits before any public live demo.
- Keep conventional deterministic tests alongside simulations. SimUsers complement scripted tests; they do not replace them.

---

## 11. Skills gained (for resume, startups, and interviews)

Only claim a skill you actually exercised. Items marked **optional** appear only if you build that part. When talking about results, quote real numbers from your own runs; never reuse illustrative figures.

| Skill area | Where learned | Interview talking point |
|---|---|---|
| **LLM application fundamentals:** messages, context windows, sampling, structured outputs, prompt iteration | V1, V2 | How I kept SimUser state separate from chat history and what broke when I did not. |
| **Agent loops and tool calling:** propose, validate, execute, return, stop conditions | V3 | Designed an agent loop with validated tools, step limits, and handling of invalid tool calls. |
| **Agent evaluation design:** baselines, held-out sets, deterministic vs judged metrics, state-based success checks | V3, V5 | Why I used database-state checks for task success and reserved LLM judging for subjective qualities. |
| **LLM-as-judge and calibration:** rubrics, different-family judge, human-rating agreement, self-preference bias | V5 | How I validated (or failed to validate) my judge against human ratings. |
| **Experimental statistics:** repeated runs, bootstrap CIs, power analysis, A/A controls, effect size, ranking stability | V5 | How many runs are needed to detect a known agent difference, and how that changed by SimUser design. |
| **Simulation and memory design:** stateful agents, fact tracking, reflection | V2 | Why stateful SimUsers leak or drift less (or did not). |
| **RAG-adjacent retrieval** `[STRETCH]`: embeddings, vector search, chunking, retrieval quality | V2 (optional) | Only if you implement embedding-based SimUser memory or a retrieval-backed support agent. Not in the core plan; add deliberately for a RAG story. |
| **Agent frameworks:** LangChain / LangGraph as an AUT, adapter pattern | V4 | Built a framework-agnostic adapter and tested a LangGraph agent and a hand-written one through the same interface. |
| **Backend engineering:** FastAPI, Pydantic, async Python, REST design, SSE/WebSocket, job lifecycle | V0, V4 | Designed a run manager with budgets, retries, idempotent re-runs, and live streaming. |
| **Data and persistence:** SQL schema design, SQLite/Postgres, experiment tracking, config hashing | V0, V4 | Every run is reproducible from its stored config, prompt versions, and raw transcript. |
| **Frontend engineering:** React + TypeScript dashboards, transcript viewer, charts with CIs, annotation UI | V0, V6 | Built a results dashboard that shows uncertainty, not just averages. |
| **Testing:** pytest, deterministic fixtures, mock/record-replay LLM, CI | V0 to V6 | Tested nondeterministic systems by isolating deterministic checks and recording LLM calls. |
| **Cost and reliability engineering:** token/cost tracking, budgets, rate limits, timeouts, retries | V1, V4, V6 | Capped cost per run and per day; demo defaults to replay mode for safety and reliability. |
| **Security and deployment basics:** server-side secrets, abuse protection, Docker, CI/CD, hosting | V6 | How I protected a public LLM-backed demo from cost abuse. |
| **Research and communication:** pre-registration-style protocol, honest reporting of failures and limitations, reproducibility, technical writing | V5, V6 | My protocol was frozen before confirmatory runs; here is what failed and why. |
| **Responsible data use:** dataset licensing, privacy, anti-leakage cleaning | V5 | Why I cleaned profile examples to prevent test-conversation leakage. |
| **Basic ML concepts** (conceptual): train/validation/test, overfitting, precision/recall, embeddings, transformers | Throughout | Connect the dev/test split in scenarios to overfitting and metric gaming. |

**Resume line pattern (fill with real facts only):**
"Creator and maintainer, SimBench: open-source framework for simulating users to test AI agents. Built [stack]; evaluated [N] agent versions with [N] runs; found [real result]. Repo: [link]."
Describe it as independent / open-source research. Do not use a founder title unless there are real users, customers, or a pilot.

---

## 12. Open items and next actions

### Open (owner has not decided)
- `[OPEN]` Primary research question (and which are secondary).
- `[OPEN]` Human rating work: include or not.
- `[OPEN]` Reference data for "realistic" and whether to include the Grounded variant.
- `[OPEN]` Timeline and current background (experience, shipped products, target companies).
- `[OPEN]` Model choices and confirmed budget after checking current prices.

### Build log
- 2026-10-23: V0 built. Scripted scenario `late_refund` (order 1042). `POST /runs` returns a transcript and stores the raw JSON in SQLite. No model call, no tools, no scores. Research question remains open. The README describes this slice and does not treat the scripted refund language as a task-success result.
- 2026-10-23: V1 built. The `late_refund` customer is an LLM SimUser (`sim_user.v1` prompt: goal, facts, style, done_when). The support agent is still the V0 script and does not read the customer. Runs log prompt text, model, seed, tokens, latency, and a list-price cost estimate (OpenAI standard uncached rates checked 2026-10-23; unknown models log null). Tests use a mock, record/replay, and a fake HTTP transport. No live provider call has been made in this environment: `OPENAI_API_KEY` was not set. Research question remains open.

- 2026-10-23: Provider switched to the Gemini free tier (default `gemini-3.5-flash-lite`) at the owner's request, to avoid any chance of a bill. The OpenAI adapter stays behind the same interface (`SIMBENCH_PROVIDER`). Runs now also log HTTP request count (free-tier quotas count requests) and a `free_tier` flag. The logged cost is a paid-rate estimate and is labeled as such on free-tier runs. The Gemini adapter is tested only against a fake transport until a real key is used.

- 2026-10-23: V2 built. The `late_refund` SimUser now has a spec (facts with `volunteer` or `on_request` disclosure, required facts, unknown facts, goal and abandon markers). Two prompt variants, `persona` (fixed prompt) and `stateful` (harness-owned state, prompt rebuilt each turn, prompt id `sim_user.v2.<variant>+<script_version>`). `checks.v1` has six deterministic keyword checks and runs after each conversation; tests include deliberate violations for each. Transcripts also store the variant, per-turn prompts, spec, and state log. UI has a variant selector, checks panel, and state log. Scenario `script_version` is `late_refund.v2` and the replay fixture was rewritten (illustrative, not live). Bias flags for any later experiment: checks are keyword-based and miss paraphrase and invented facts; the unlock-phrase heuristic can misfire; stateful-versus-persona leakage is partly true by construction (the hidden fact is never in the stateful prompt); markers and prompts must be tuned on dev scenarios only; only one scenario exists. No live V2 run has been recorded yet. Optional V2 items (embedding memory, reflection step) were not built. Research question remains open.

### Free-tier implications to keep in mind
- **Quota caps study size.** Roughly 500 requests a day on Flash-Lite (read the real figure in AI Studio) is plenty for V1 to V4 pilots but not for a large pool. A V5 pool of hundreds of conversations, each several requests, with an LLM agent doing its own calls from V3 on, would take many days or exceed the quota. Size the study to the quota, record that in the protocol, or use a local open model for bulk runs.
- **Reproducibility.** Hosted model versions change and previews retire. Every run stores the model version the provider reports and the raw transcript, and replay fixtures keep old runs usable. Do not treat a rerun a month later as the same simulator without checking the model version.
- **Same-family bias.** The doc asks for a judge from a different model family than the simulator. With a Gemini simulator, the judge cannot be Gemini or Gemma. A free judge from another family (for example a local Llama or Qwen) is an open item for V5.
- **Data use.** Free-tier prompts may be used by Google. Scenarios stay synthetic. Revisit before using any real user text.
- **Demo.** The V6 public demo stays in replay mode. A shared free-tier key is not suitable for public live runs.

### Immediate next actions
V0, V1, and V2 are in the repo. A live conversation needs `GEMINI_API_KEY` in `.env` and a server restart. Before V3, run each variant live a few times (3 to 4 requests per run) and read the transcripts and checks by hand to see where the keyword checks are wrong. Next milestone is V3 (mock orders database, agent tools, agent versions), only when asked.

---

## 13. Working rules for the coding assistant

- Treat the research question as **OPEN**. Make scenarios, SimUser variants, agent versions, and metrics configuration-driven, not hard-coded to one hypothesis.
- Build only the current milestone. Do not scaffold later versions, multi-agent orchestration, queues, or auth unless asked.
- Write tests alongside code. Keep LLM calls behind an interface so tests run offline with a mock or recorded client.
- Log every run's config, prompt versions, seeds, model names, tokens, cost, and raw transcript. Never discard raw transcripts.
- Never put API keys in the frontend or in the repository. Use environment variables and server-side calls.
- Do not fabricate results, benchmarks, or numbers. Illustrative values must be labeled as such and never reported as findings.
- Prefer simple, readable code the owner can fully explain in an interview. Ask before adding heavy dependencies or frameworks.
- Flag any place where a design choice would bias a future experiment (for example, tuning on held-out scenarios).
