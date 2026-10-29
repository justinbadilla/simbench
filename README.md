# SimBench

SimBench is a research instrument for testing AI agents with simulated users. The research question is still open. The current lean is whether a SimUser evaluation can reliably detect known differences between agent versions, and whether the SimUser design changes how many runs that takes. Fidelity claims are out of scope until there is a reference for "realistic."

This repository is at **V2**. The customer in `late_refund` is a model working from a fact sheet, and checks after each run say whether it kept to the sheet. The support agent is still a script and does not read what the customer says. Nothing updates an order database, and a transcript is not a score.

## V2 scope

- **A SimUser spec** on the scenario (`core/simbench/spec.py`): a goal, a style, and a list of facts. Each fact is either `volunteer` (the customer may say it any time) or `on_request` (only after the agent asks, which is detected by `unlock_phrases`). A fact can be `required`. **Unknown facts** are things the customer must not know until the agent says them, such as a tracking number.
- **Two prompt variants**, chosen per run (`variant` in `POST /runs`, a selector in the UI):
  - `persona`: one fixed prompt listing every fact, with on-request facts marked "do not mention until asked". The model has to remember what it has said.
  - `stateful`: the harness tracks state outside the chat history and rebuilds the prompt every turn. A fact the customer has not been asked for is not in the prompt at all. Facts already said move to an "already told" list.
- **State** (`state.py`): what the agent asked for, what the customer said, what the agent told the customer. Kept as an event log on the transcript and rebuildable from the messages alone.
- **Six deterministic checks** (`checks.py`, version `checks.v1`), run after the conversation: `unknown_fact_not_used`, `no_early_reveal`, `fact_coverage`, `answered_when_asked`, `goal_stated`, `goal_not_abandoned`. Each returns pass, fail, or not applicable, with evidence (turn numbers). They are skipped when the run ended in `model_error`.
- The transcript stores the variant, the exact prompt sent on every turn, the spec, the state log, and the checks.
- Tests include deliberately broken transcripts, one for each check, to prove the check can fail.

`POST /runs` with `{"mode": "scripted"}` still runs the V0 script, where neither side reads the other.

## Limits of the V2 checks

Read these before quoting a check result anywhere.

- **Keyword matching only.** A fact counts as said if one of its markers appears in a customer message. A paraphrase ("my order number ends in 42") is missed, and a marker can match by accident.
- **Invented details are invisible.** The checks catch a customer using a fact they should not have, but not one making up a new fact (a fake phone number, say).
- **The unlock heuristic is crude.** Any agent message containing an unlock phrase ("email") counts as a request, even if the agent was only mentioning it.
- **Stateful versus persona is partly built in.** The stateful prompt never contains a hidden fact, so it cannot leak that fact from the prompt. A lower leak rate there shows the mechanism works, not that stateful users are more realistic.
- **One scenario, scripted agent.** Nothing here supports a general claim about SimUsers. Tune markers and prompts on dev scenarios only, never on a scenario you will later evaluate on.
- **A pass is not a verdict.** "No check failed" means these six rules were not broken, not that the conversation was good.
- **No live V2 run is recorded in this repository.** The replay fixture (`data/cassettes/late_refund.fixture.json`) is hand-written to show the UI and is not a result.

## Free tier, quotas, and cost

The default provider is the Gemini API on its free tier: no card, nothing billed. Things to know:

- **Free is per model, and limited per minute and per day.** `gemini-3.5-flash-lite` is the default because it has the larger daily allowance (about 500 requests a day as of 2026-10-23). The bigger Flash models allow far fewer. Google no longer prints exact limits, so read yours in [AI Studio](https://aistudio.google.com). Daily counters reset at midnight Pacific.
- **Each run logs its request count**, retries included, because quotas count requests. A run is about one request per customer turn (3 to 4 for `late_refund`).
- **Linking a billing account moves the project off the free tier.** Do not link one if you want a guarantee of no bill. If you do, set `SIMBENCH_GEMINI_FREE_TIER=false` so logs stop labeling runs as free.
- **Google may use free-tier prompts and responses to improve its products.** Use synthetic scenarios only. Do not paste real customer or personal data.
- **"Cost" in the transcript is a paid-rate estimate.** On the free tier the page shows `$0.00 (free tier)` plus what the run would cost at list price, so you can still plan. Prices were checked on 2026-10-23, cached input is not discounted, and an unknown model logs `n/a` rather than a guessed price. This is not an invoice.
- **The Gemini adapter is tested against a fake transport, not the live service** until you run it with a real key.

## Layout

```
core/simbench/     runner, SimUser spec/state/checks/prompt, LLM client, SQLite log
api/simbench_api/  FastAPI app (POST /runs)
web/               React + TypeScript transcript viewer
data/scenarios/    scenario JSON
data/cassettes/    replay fixtures, not live results
tests/             turn order, termination, client, spec, state, checks, API
docs/              living project context
```

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"
npm install --prefix web
```

Copy `.env.example` to `.env` and set `GEMINI_API_KEY` (a free key from [Google AI Studio](https://aistudio.google.com/apikey)). The key stays on the server. To use OpenAI instead, set `SIMBENCH_PROVIDER=openai` and `OPENAI_API_KEY`.

## Run

```bash
make dev
```

The UI is at [http://127.0.0.1:5173](http://127.0.0.1:5173). The API is at [http://127.0.0.1:8000](http://127.0.0.1:8000). Pick a customer prompt variant, then **Run simulation** calls the model. Each run is appended to `data/simbench.sqlite`.

`SIMBENCH_MODEL` overrides the model (default `gemini-3.5-flash-lite`, or `gpt-4o-mini` for OpenAI). `SIMBENCH_REPLAY=data/cassettes/late_refund.fixture.json` plays a fixture instead of calling a provider. That fixture is not a live result. Restart the server after changing `.env`.

## Test

```bash
make test
```

## Turn order and termination

The customer speaks first. The model writes the next customer message from the transcript so far. The agent then says its next scripted line.

- `sim_user_ended`: the model set `end_conversation` on the message it just sent.
- `script_complete`: it is the agent's turn and the agent has no line left.
- `max_turns`: the next speaker could still talk, and the transcript is already at the cap.
- `model_error`: the provider call failed. Messages already said are kept.

Scripted mode (`mode: scripted`) still alternates fixed lines and stops for `script_complete` or `max_turns`. A script that uses every line and then stops is `script_complete`, even when the number of messages equals the cap.
