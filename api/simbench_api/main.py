"""V1 API: one endpoint. The SimUser is a model unless the request asks for the script."""

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from simbench.env import load_local_env
from simbench.llm import (
    DEFAULT_GEMINI_MODEL,
    LLMClient,
    client_from_environment,
    missing_key_message,
    model_from_environment,
)
from simbench.models import RunRequest, Transcript
from simbench.paths import default_db_path
from simbench.run import run_llm_simulation, run_simulation
from simbench.scenario import ScenarioNotFoundError, load_scenario
from simbench.store import save_transcript


def create_app(
    db_path: Path | None = None,
    llm_client: LLMClient | None = None,
    *,
    model: str = DEFAULT_GEMINI_MODEL,
    missing_key_detail: str | None = None,
) -> FastAPI:
    app = FastAPI(
        title="SimBench",
        version="0.0.1",
        description="V1 runs an LLM SimUser against a scripted support agent and stores the transcript.",
    )
    app.state.db_path = default_db_path() if db_path is None else db_path
    app.state.llm_client = llm_client
    app.state.model = model
    app.state.missing_key_detail = missing_key_detail or missing_key_message()
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://127.0.0.1:5173", "http://localhost:5173"],
        allow_methods=["POST"],
        allow_headers=["Content-Type"],
    )

    @app.post("/runs", response_model=Transcript)
    def create_run(body: RunRequest | None = None) -> Transcript:
        request = body or RunRequest()
        try:
            scenario = load_scenario(request.scenario_id)
        except ScenarioNotFoundError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        if request.mode == "scripted":
            transcript = run_simulation(scenario, max_turns=request.max_turns)
        else:
            client = app.state.llm_client
            if client is None:
                raise HTTPException(status_code=503, detail=app.state.missing_key_detail)
            if scenario.sim_user is None:
                raise HTTPException(status_code=400, detail="This scenario has no SimUser brief.")
            transcript = run_llm_simulation(
                scenario,
                client,
                max_turns=request.max_turns,
                model=app.state.model,
                seed=request.seed,
            )
        save_transcript(app.state.db_path, transcript)
        return transcript

    return app


load_local_env()
app = create_app(llm_client=client_from_environment(), model=model_from_environment())
