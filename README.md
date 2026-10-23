# SimBench

SimBench is a research instrument for testing AI agents with simulated users. The research question is still open. The current lean is whether a SimUser evaluation can reliably detect known differences between agent versions, and whether the SimUser design changes how many runs that takes. Fidelity claims are out of scope until there is a reference for "realistic."

This repository is at **V1**. The customer in `late_refund` is a model. The support agent is still a script and does not read what the customer says. Nothing updates an order database, and a transcript is not a score.

## V1 scope

- `LLMClient` with a Gemini adapter (free tier by default) and an OpenAI adapter. Both share one request loop with a timeout, retries, and structured JSON.
- The SimUser brief (goal, facts, style, when to stop) lives on the scenario. Facts are repeated in the prompt on every turn. That is not a memory store.
- Each run logs the prompt text, prompt version, model, provider, seed, tokens, requests, latency, and a list-price cost estimate.
- `POST /runs` writes the raw transcript to SQLite, including a run that dies mid-conversation.
- Tests use a mock client, a record/replay cassette, and a fake HTTP transport. They do not call a provider.

`POST /runs` with `{"mode": "scripted"}` still runs the V0 script, where neither side reads the other.

## Not in V1

SimUser variants, agent versions, tools, scores, batches, streaming, and auth. The research question is still open.

## Free tier, quotas, and cost

The default provider is the Gemini API on its free tier: no card, nothing billed. Things to know:

- **Free is per model, and limited per minute and per day.** `gemini-3.5-flash-lite` is the default because it has the larger daily allowance (about 500 requests a day as of 2026-10-23). The bigger Flash models allow far fewer. Google no longer prints exact limits, so read yours in [AI Studio](https://aistudio.google.com). Daily counters reset at midnight Pacific.
- **Each run logs its request count**, retries included, because quotas count requests. A V1 run is about one request per customer turn.
- **Linking a billing account moves the project off the free tier.** Do not link one if you want a guarantee of no bill. If you do, set `SIMBENCH_GEMINI_FREE_TIER=false` so logs stop labeling runs as free.
- **Google may use free-tier prompts and responses to improve its products.** Use synthetic scenarios only. Do not paste real customer or personal data.
- **"Cost" in the transcript is a paid-rate estimate.** On the free tier the page shows `$0.00 (free tier)` plus what the run would cost at list price, so you can still plan. Prices were checked on 2026-10-23, cached input is not discounted, and an unknown model logs `n/a` rather than a guessed price. This is not an invoice.
- **The Gemini adapter is tested against a fake transport, not the live service** until you run it with a real key.

## Layout

```
core/simbench/     runner, SimUser prompt, LLM client, SQLite log
api/simbench_api/  FastAPI app (POST /runs)
web/               React + TypeScript transcript viewer
data/scenarios/    scenario JSON
data/cassettes/    replay fixtures, not live results
tests/             turn order, termination, client, API
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

The UI is at [http://127.0.0.1:5173](http://127.0.0.1:5173). The API is at [http://127.0.0.1:8000](http://127.0.0.1:8000). **Run simulation** calls the model. Each run is appended to `data/simbench.sqlite`.

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
