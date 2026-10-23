import { useState } from "react";

type Role = "sim_user" | "agent";

type Message = {
  turn_index: number;
  role: Role;
  content: string;
};

type Transcript = {
  run_id: string;
  created_at: string;
  scenario_title: string;
  notes: string;
  script_version: string;
  sim_user_id: string;
  agent_id: string;
  max_turns: number;
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

const terminationLabel: Record<Transcript["termination_reason"], string> = {
  script_complete: "Script complete",
  max_turns: "Max turns",
  sim_user_ended: "SimUser ended",
  model_error: "Model error",
};

const roleLabel: Record<Role, string> = {
  sim_user: "SimUser",
  agent: "Agent",
};

export function App() {
  const [transcript, setTranscript] = useState<Transcript | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [running, setRunning] = useState(false);

  async function runSimulation() {
    setRunning(true);
    setError(null);
    try {
      const response = await fetch("/runs", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ scenario_id: "late_refund" }),
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
        <p className="eyebrow">SimBench · V1</p>
        <h1>Late delivery refund</h1>
        <p className="lede">
          The customer is a model. The support agent still follows a script and does not read them. No order is
          changed.
        </p>
        <button type="button" onClick={runSimulation} disabled={running}>
          {running ? "Running…" : "Run simulation"}
        </button>
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

function formatLatency(milliseconds: number): string {
  if (milliseconds < 1000) {
    return `${Math.round(milliseconds)} ms`;
  }
  return `${(milliseconds / 1000).toFixed(1)} s`;
}

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
    </section>
  );
}
