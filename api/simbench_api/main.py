"""HTTP API: one endpoint, ``POST /runs``. The SimUser is a model unless the request asks for the script.

The API is a thin layer. It validates the request, picks the runner, saves the
transcript, and returns it. All simulation logic lives in ``core/simbench``.
"""

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from simbench.agent import (
    AgentConfig,
    AgentNotFoundError,
    list_agent_configs,
    load_agent_config,
)
from simbench.env import load_local_env
from simbench.llm import (
    DEFAULT_GEMINI_MODEL,
    LLMClient,
    client_from_environment,
    missing_key_message,
    model_from_environment,
)
from simbench.llm_agent import LLMAgent
from simbench.models import RunRequest, Transcript
from simbench.paths import default_db_path
from simbench.run import run_llm_simulation, run_simulation
from simbench.scenario import ScenarioNotFoundError, list_scenarios, load_scenario
from simbench.store import save_transcript


def create_app(
    db_path: Path | None = None,
    llm_client: LLMClient | None = None,
    *,
    model: str = DEFAULT_GEMINI_MODEL,
    missing_key_detail: str | None = None,
) -> FastAPI:
    """Build the FastAPI app with its dependencies injected.

    Taking the database path and LLM client as arguments lets tests pass a temp
    file and a mock client, so no test touches the network or the real log.

    Args:
        db_path: SQLite file for stored runs; defaults to ``data/simbench.sqlite``.
        llm_client: Client for LLM runs. ``None`` means no key is configured and
            LLM runs answer with HTTP 503.
        model: Model name passed to every LLM run.
        missing_key_detail: Message for the 503; defaults to the provider-specific
            text from ``missing_key_message``.

    Returns:
        The configured app, with CORS limited to the local Vite dev server.

    Called by:
        Module import (below) for the real server, and the API tests.
    """
    app = FastAPI(
        title="SimBench",
        version="0.0.1",
        description="Runs a simulated customer against a scripted support agent, checks it, and stores the transcript.",
    )
    app.state.db_path = default_db_path() if db_path is None else db_path
    app.state.llm_client = llm_client
    app.state.model = model
    app.state.missing_key_detail = missing_key_detail or missing_key_message()
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://127.0.0.1:5173", "http://localhost:5173"],
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
    )

    @app.get("/scenarios")
    def get_scenarios() -> list[dict[str, object]]:
        """List the available scenarios for the UI's picker.

        Returns:
            One entry per scenario file: its id, title, description, and whether it
            has a mock environment (agents with tools need one).

        Called by:
            The web UI on page load.
        """
        return [
            {
                "id": scenario.id,
                "title": scenario.title,
                "description": scenario.description,
                "has_environment": scenario.environment is not None,
            }
            for scenario in list_scenarios()
        ]

    @app.get("/agents", response_model=list[AgentConfig])
    def get_agents() -> list[AgentConfig]:
        """List the agent versions (A, B, B-prime, C, D) for the UI's picker.

        The scripted agent is not listed because it is not a config; ``agent`` set to
        ``"scripted"`` always means it.

        Called by:
            The web UI on page load.
        """
        return list_agent_configs()

    @app.post("/runs", response_model=Transcript)
    def create_run(body: RunRequest | None = None) -> Transcript:
        """Run one simulation, store it, and return the transcript.

        An empty body uses the defaults in ``RunRequest`` (LLM SimUser, stateful
        variant, scripted agent).

        Returns:
            The full ``Transcript``, which is also saved to SQLite.

        Raises:
            HTTPException: 404 for an unknown scenario or agent, 503 when no API key is
                configured, 400 when the scenario has no SimUser spec, when an agent is
                requested in scripted mode, when the scenario has no environment for an
                agent that uses tools, or when the configured client cannot drive an
                agent (replay mode). Pydantic returns 422 for an invalid body.

        Called by:
            The web UI's "Run simulation" button.
        """
        request = body or RunRequest()
        try:
            scenario = load_scenario(request.scenario_id)
        except ScenarioNotFoundError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        if request.mode == "scripted":
            if request.agent != "scripted":
                raise HTTPException(status_code=400, detail="Scripted mode can only use the scripted agent.")
            transcript = run_simulation(scenario, max_turns=request.max_turns)
        else:
            client = app.state.llm_client
            if client is None:
                raise HTTPException(status_code=503, detail=app.state.missing_key_detail)
            if scenario.sim_user is None:
                raise HTTPException(status_code=400, detail="This scenario has no SimUser brief.")
            agent = None
            if request.agent != "scripted":
                try:
                    config = load_agent_config(request.agent)
                except AgentNotFoundError as error:
                    raise HTTPException(status_code=404, detail=str(error)) from error
                if not hasattr(client, "complete_agent"):
                    raise HTTPException(
                        status_code=400,
                        detail=(
                            "This server is in replay mode, which can only play the scripted agent. "
                            "Unset SIMBENCH_REPLAY and set an API key to run a model agent."
                        ),
                    )
                try:
                    agent = LLMAgent(config, client, scenario.environment, model=app.state.model)
                except ValueError as error:
                    raise HTTPException(status_code=400, detail=str(error)) from error
            transcript = run_llm_simulation(
                scenario,
                client,
                agent=agent,
                max_turns=request.max_turns,
                model=app.state.model,
                variant=request.variant,
                seed=request.seed,
            )
        save_transcript(app.state.db_path, transcript)
        return transcript

    return app


load_local_env()
app = create_app(llm_client=client_from_environment(), model=model_from_environment())
