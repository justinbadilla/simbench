"""Usage accounting shared by the SimUser side and the agent side of a run."""

from simbench.llm import CallUsage, LLMError
from simbench.models import Usage


class CallTotals:
    """Running totals of model usage across all calls made by one side of a run.

    One instance tracks the SimUser, and a separate one tracks the agent, so the
    two are never mixed. Both are also usable by any future caller of a model.

    Collects tokens, latency, request count, per-call costs, and free-tier
    flags from successful calls and failed ones alike, then folds them into one
    ``Usage`` record. Failed attempts are included because they still consume
    quota and time.
    """

    def __init__(self) -> None:
        """Start with all counters at zero and no model or provider known yet."""
        self.input_tokens = 0
        self.output_tokens = 0
        self.latency_ms = 0.0
        self.requests = 0
        self.costs: list[float | None] = []
        self.free_flags: list[bool] = []
        self.model_name: str | None = None
        self.provider: str | None = None

    def add_call(self, result: CallUsage) -> None:
        """Add the usage of one successful call (``LLMCall`` or ``AgentCall``) and remember its model and provider."""
        self.input_tokens += result.input_tokens
        self.output_tokens += result.output_tokens
        self.latency_ms += result.latency_ms
        self.requests += result.requests
        self.costs.append(result.estimated_cost_usd)
        self.free_flags.append(result.free_tier)
        self.model_name = result.model_name
        self.provider = result.provider

    def add_error(self, error: LLMError) -> None:
        """Add the usage a failed call still incurred (retries, partial tokens).

        A cost or free-tier flag is only recorded if the failure actually spent
        tokens or sent a request, so an error before any request does not skew
        the totals.
        """
        self.input_tokens += error.input_tokens
        self.output_tokens += error.output_tokens
        self.latency_ms += error.latency_ms
        self.requests += error.requests
        if error.input_tokens or error.output_tokens:
            self.costs.append(error.estimated_cost_usd)
        if error.requests:
            self.free_flags.append(error.free_tier)
        if error.model_name:
            self.model_name = error.model_name

    def usage(self, *, model: str, prompt: str, seed: int) -> Usage:
        """Fold the totals into the ``Usage`` stored on the transcript.

        Cost rules: no priced calls gives 0.0; if any call had an unknown price
        the total is ``None`` rather than a partial sum. ``free_tier`` is true
        only if at least one call happened and every call was free.

        Args:
            model: Fallback name when no call reported one.
            prompt: The prompt version string (see ``prompt_version``).
            seed: The seed used for the run.
        """
        if not self.costs:
            cost: float | None = 0.0
        elif any(item is None for item in self.costs):
            cost = None
        else:
            cost = round(sum(item for item in self.costs if item is not None), 8)
        return Usage(
            model_name=self.model_name or model,
            provider=self.provider,
            prompt_version=prompt,
            seed=seed,
            input_tokens=self.input_tokens,
            output_tokens=self.output_tokens,
            latency_ms=round(self.latency_ms, 1),
            requests=self.requests,
            free_tier=bool(self.free_flags) and all(self.free_flags),
            estimated_cost_usd=cost,
        )
