import { useState } from "react";

// These types mirror the API response (see `Transcript` in core/simbench/models.py).
// Keep them in sync by hand when the backend schema changes.

/** Who sent a message: the simulated customer or the scripted support agent. */
type Role = "sim_user" | "agent";

type Message = {
  turn_index: number;
  role: Role;
  content: string;
};

/** How the SimUser's knowledge is presented to the model. Sent as `variant` in the request. */
type Variant = "persona" | "stateful";

/** One deterministic check. `passed` is null when the check did not apply to this scenario. */
type CheckResult = {
  name: string;
  passed: boolean | null;
  value: number | null;
  detail: string;
  evidence: string[];
};

/** One change in the SimUser's tracked state, for example "agent asked for the email". */
type StateEvent = {
  turn_index: number;
  kind: "unlocked" | "revealed" | "early_reveal" | "learned" | "unknown_fact_used";
  fact_id: string;
};

/** The full run record returned by `POST /runs`. */
type Transcript = {
  run_id: string;
  created_at: string;
  scenario_title: string;
  notes: string;
  script_version: string;
  sim_user_id: string;
  agent_id: string;
  max_turns: number;
  variant: string;
  turn_prompts: string[];
  state_log: StateEvent[];
  checks: { checks_version: string; all_passed: boolean; checks: CheckResult[] } | null;
  messages: Message[];
  termination_reason: "script_complete" | "max_turns" | "sim_user_ended" | "model_error";
  error: string;
  usage: {
    model_name: string | null;
    provider: string | null;
    prompt_version: string | null;
    seed: number | null;
    input_tokens: number;
    output_tokens: number;
    latency_ms: number;
    requests: number;
    free_tier: boolean;
    estimated_cost_usd: number | null;
  };
};

/** Display text for each reason a run can stop. */
const terminationLabel: Record<Transcript["termination_reason"], string> = {
  script_complete: "Script complete",
  max_turns: "Max turns",
  sim_user_ended: "SimUser ended",
  model_error: "Model error",
};

/** Display name for each speaker. */
const roleLabel: Record<Role, string> = {
  sim_user: "SimUser",
  agent: "Agent",
};

/** Dropdown text for each prompt variant. Keys must match the backend's `Variant`. */
const variantLabel: Record<Variant, string> = {
  stateful: "Stateful (the harness tracks what the customer may say)",
  persona: "Persona (one fixed prompt, the model tracks it itself)",
};

/** Plain-language description of each state event kind, shown in the state log. */
const eventLabel: Record<StateEvent["kind"], string> = {
  unlocked: "Agent asked for",
  revealed: "Customer said",
  early_reveal: "Customer said before being asked",
  learned: "Agent told the customer",
  unknown_fact_used: "Customer used something they could not know",
};

/** Event kinds that indicate a rule was broken. The state log highlights these. */
const badEvents: ReadonlySet<StateEvent["kind"]> = new Set(["early_reveal", "unknown_fact_used"]);

/**
 * Page root. Holds the selected variant, the latest transcript, and any error.
 * Rendered once by `main.tsx`.
 */
export function App() {
  const [variant, setVariant] = useState<Variant>("stateful");
  const [transcript, setTranscript] = useState<Transcript | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [running, setRunning] = useState(false);

  /**
   * Start a run: POST the chosen variant to `/runs` (proxied to the API in dev),
   * then store the returned transcript. On failure it clears the old transcript
   * and shows the server's `detail` message, for example the missing-key text.
   * Triggered by the "Run simulation" button.
   */
  async function runSimulation() {
    setRunning(true);
    setError(null);
    try {
      const response = await fetch("/runs", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ scenario_id: "late_refund", variant }),
      });
      if (!response.ok) {
        const body = (await response.json().catch(() => null)) as { detail?: string } | null;
        throw new Error(body?.detail ?? `Run failed (${response.status})`);
      }
      setTranscript((await response.json()) as Transcript);
    } catch (caught) {
      setTranscript(null);
      setError(caught instanceof Error ? caught.message : "Run failed");
    } finally {
      setRunning(false);
    }
  }

  return (
    <main className="page">
      <header className="header">
        <p className="eyebrow">SimBench · V2</p>
        <h1>Late delivery refund</h1>
        <p className="lede">
          The customer is a model with a fact sheet: some details they volunteer, one they only give when asked,
          and one they cannot know. The support agent still follows a script. Checks after the run say whether
          the customer kept to the sheet. No order is changed.
        </p>
        <div className="controls">
          <label>
            <span className="control-label">Customer prompt</span>
            <select value={variant} onChange={(event) => setVariant(event.target.value as Variant)} disabled={running}>
              {(Object.keys(variantLabel) as Variant[]).map((key) => (
                <option key={key} value={key}>
                  {variantLabel[key]}
                </option>
              ))}
            </select>
          </label>
          <button type="button" onClick={runSimulation} disabled={running}>
            {running ? "Running…" : "Run simulation"}
          </button>
        </div>
      </header>

      {error ? (
        <p className="error" role="alert">
          {error}
        </p>
      ) : null}

      {transcript ? <TranscriptView transcript={transcript} /> : null}
    </main>
  );
}

/**
 * Format a dollar estimate for display. Returns "n/a" for null (unknown model
 * price), "$0.00" for zero, six decimals for sub-cent values, otherwise four.
 */
function formatCost(value: number | null): string {
  if (value === null) {
    return "n/a";
  }
  if (value === 0) {
    return "$0.00";
  }
  if (value < 0.01) {
    return `$${value.toFixed(6)}`;
  }
  return `$${value.toFixed(4)}`;
}

/** Format a duration as milliseconds below one second, otherwise as seconds with one decimal. */
function formatLatency(milliseconds: number): string {
  if (milliseconds < 1000) {
    return `${Math.round(milliseconds)} ms`;
  }
  return `${(milliseconds / 1000).toFixed(1)} s`;
}

/**
 * Render one run: the summary card (termination, model, usage, cost), the
 * message list, then the checks, state log, and prompts panels.
 * Used by `App` whenever a transcript is available.
 */
function TranscriptView({ transcript }: { transcript: Transcript }) {
  return (
    <section className="transcript" aria-label="Transcript">
      <div className="summary">
        <h2>{transcript.scenario_title}</h2>
        <p className="notes">{transcript.notes}</p>
        {transcript.error ? (
          <p className="error" role="alert">
            {transcript.error}
          </p>
        ) : null}
        <dl>
          <div>
            <dt>Termination</dt>
            <dd>{terminationLabel[transcript.termination_reason]}</dd>
          </div>
          <div>
            <dt>Messages</dt>
            <dd>
              {transcript.messages.length} / {transcript.max_turns}
            </dd>
          </div>
          <div>
            <dt>Model</dt>
            <dd>{transcript.usage.model_name ?? "none"}</dd>
          </div>
          <div>
            <dt>Provider</dt>
            <dd>{transcript.usage.provider ?? "none"}</dd>
          </div>
          <div>
            <dt>Variant</dt>
            <dd>{transcript.variant}</dd>
          </div>
          <div>
            <dt>Prompt</dt>
            <dd>{transcript.usage.prompt_version ?? transcript.script_version}</dd>
          </div>
          <div>
            <dt>Seed</dt>
            <dd>{transcript.usage.seed ?? "none"}</dd>
          </div>
          <div>
            <dt>Latency</dt>
            <dd>{formatLatency(transcript.usage.latency_ms)}</dd>
          </div>
          <div>
            <dt>Requests</dt>
            <dd>{transcript.usage.requests}</dd>
          </div>
          <div>
            <dt>Tokens</dt>
            <dd>
              {transcript.usage.input_tokens} in / {transcript.usage.output_tokens} out
            </dd>
          </div>
          <div>
            <dt>{transcript.usage.free_tier ? "Billed" : "Cost"}</dt>
            <dd>
              {transcript.usage.free_tier
                ? `$0.00 (free tier) · about ${formatCost(transcript.usage.estimated_cost_usd)} at paid rates`
                : formatCost(transcript.usage.estimated_cost_usd)}
            </dd>
          </div>
        </dl>
        <p className="run-id">Run {transcript.run_id}</p>
      </div>
      <ol className="messages">
        {transcript.messages.map((message) => (
          <li key={message.turn_index} className={`message ${message.role}`}>
            <div className="message-meta">
              <span>{roleLabel[message.role]}</span>
              <span>Turn {message.turn_index + 1}</span>
            </div>
            <p>{message.content}</p>
          </li>
        ))}
      </ol>
      <ChecksPanel report={transcript.checks} />
      <StateLog events={transcript.state_log} />
      <PromptsSent prompts={transcript.turn_prompts} />
    </section>
  );
}

/**
 * List each check with a pass, fail, or n/a badge, its explanation, and any
 * evidence lines. `report` is null when checks were skipped after a model
 * error, which is shown as a note rather than as a pass. Includes a reminder
 * that these are keyword checks and a pass does not prove good behavior.
 */
function ChecksPanel({ report }: { report: Transcript["checks"] }) {
  if (!report) {
    return (
      <section className="panel" aria-label="Checks">
        <h3>Checks</h3>
        <p className="muted">Skipped because the model failed before the run finished.</p>
      </section>
    );
  }
  return (
    <section className="panel" aria-label="Checks">
      <h3>
        Checks <span className="muted">· {report.checks_version}</span>
      </h3>
      <p className={report.all_passed ? "verdict pass" : "verdict fail"}>
        {report.all_passed ? "No check failed." : "At least one check failed."}
      </p>
      <ul className="checks">
        {report.checks.map((check) => (
          <li key={check.name}>
            <div className="check-head">
              <span className={`badge ${check.passed === null ? "na" : check.passed ? "pass" : "fail"}`}>
                {check.passed === null ? "n/a" : check.passed ? "pass" : "fail"}
              </span>
              <code>{check.name}</code>
              {check.value !== null ? <span className="muted">{Math.round(check.value * 100)}%</span> : null}
            </div>
            <p>{check.detail}</p>
            {check.evidence.length > 0 ? (
              <ul className="evidence">
                {check.evidence.map((line) => (
                  <li key={line}>{line}</li>
                ))}
              </ul>
            ) : null}
          </li>
        ))}
      </ul>
      <p className="muted small">
        Keyword matching only. It misses paraphrases and cannot see invented details, so a pass does not prove
        the customer behaved.
      </p>
    </section>
  );
}

/** Show the ordered state events (what the agent asked, what the customer said), highlighting rule-breaking ones. */
function StateLog({ events }: { events: StateEvent[] }) {
  return (
    <section className="panel" aria-label="State log">
      <h3>State log</h3>
      {events.length === 0 ? (
        <p className="muted">Nothing was tracked.</p>
      ) : (
        <ol className="events">
          {events.map((event, index) => (
            <li key={`${event.turn_index}-${event.kind}-${event.fact_id}-${index}`}>
              <span className="muted">Turn {event.turn_index + 1}</span>
              <span className={badEvents.has(event.kind) ? "event-bad" : undefined}>
                {eventLabel[event.kind]}
              </span>
              <code>{event.fact_id}</code>
            </li>
          ))}
        </ol>
      )}
    </section>
  );
}

/**
 * Show the exact prompt sent to the model on each customer turn, collapsed by
 * default. Useful for seeing how the stateful prompt changes between turns.
 * Renders nothing when no model call was made.
 */
function PromptsSent({ prompts }: { prompts: string[] }) {
  if (prompts.length === 0) {
    return null;
  }
  return (
    <section className="panel" aria-label="Prompts sent">
      <h3>Prompts sent to the model</h3>
      {prompts.map((prompt, index) => (
        <details key={index}>
          <summary>Customer turn {index + 1}</summary>
          <pre>{prompt}</pre>
        </details>
      ))}
    </section>
  );
}
