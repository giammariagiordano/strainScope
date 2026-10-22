import { ArrowLeft, Download, ExternalLink, Pause, Play, Trash2 } from "lucide-react";
import { useState } from "react";
import { Link, NavLink, useNavigate, useParams } from "react-router-dom";
import { api, exportUrl, usePoll } from "../api";
import { Button, StatusBadge, cx } from "../components/ui";
import DevelopersTab from "./DevelopersTab";
import OverviewTab from "./OverviewTab";
import PipelineTab from "./PipelineTab";

const TABS = [
  { id: "pipeline", label: "Pipeline" },
  { id: "overview", label: "Overview" },
  { id: "developers", label: "Developers" },
];

export default function ProjectPage() {
  const { id: idParam, tab = "pipeline" } = useParams();
  const id = Number(idParam);
  const navigate = useNavigate();
  const { data: p, error, refresh } = usePoll(() => api.project(id), 2000, [id]);
  const [busy, setBusy] = useState(false);
  const [actionError, setActionError] = useState<string | null>(null);

  async function act(fn: () => Promise<unknown>) {
    setBusy(true);
    setActionError(null);
    try {
      await fn();
      await refresh();
    } catch (e) {
      setActionError((e as Error).message);
    } finally {
      setBusy(false);
    }
  }

  if (error && !p) return <p className="text-sm text-bad">{error}</p>;
  if (!p) return <div className="h-40 animate-pulse rounded-2xl bg-raised" />;

  const canPause = p.status === "running" || p.status === "queued";
  const canResume = p.status === "paused" || p.status === "failed";
  const [owner, name] = p.full_name.split("/");

  return (
    <div className="space-y-6">
      <div>
        <Link to="/" className="inline-flex items-center gap-1 text-sm text-muted hover:text-ink">
          <ArrowLeft size={14} /> Projects
        </Link>
        <div className="mt-3 flex flex-wrap items-start justify-between gap-4">
          <div className="min-w-0">
            <div className="flex flex-wrap items-center gap-3">
              <h1 className="truncate text-2xl font-semibold tracking-tight text-ink">
                <span className="font-normal text-muted">{owner} / </span>
                {name}
              </h1>
              <StatusBadge status={p.status} pausing={p.pause_requested && p.status === "running"} />
            </div>
            <a
              href={p.url}
              target="_blank"
              rel="noreferrer"
              className="mt-1 inline-flex items-center gap-1 text-sm text-muted hover:text-accent"
            >
              {p.url.replace("https://", "")} <ExternalLink size={12} />
            </a>
          </div>
          <div className="flex gap-2">
            {canPause && (
              <Button onClick={() => act(() => api.pause(id))} disabled={busy || p.pause_requested}>
                <Pause size={15} /> {p.pause_requested ? "Pausing…" : "Pause"}
              </Button>
            )}
            {canResume && (
              <Button variant="primary" onClick={() => act(() => api.resume(id))} disabled={busy}>
                <Play size={15} /> Resume
              </Button>
            )}
            <a
              href={exportUrl([id])}
              download
              title="Download this repository's pseudonymized dataset (CSV + codebook)"
              className="inline-flex items-center justify-center gap-1.5 rounded-lg border border-line bg-surface px-3 py-2 text-sm font-medium text-ink transition hover:bg-raised"
            >
              <Download size={15} /> <span className="sr-only sm:not-sr-only">Export</span>
            </a>
            <Button
              variant="danger"
              disabled={busy || p.status === "deleting"}
              title="Delete project and all mined data"
              onClick={() => {
                if (confirm(`Delete ${p.full_name} and all its mined data? This cannot be undone.`))
                  act(async () => {
                    await api.remove(id);
                    navigate("/");
                  });
              }}
            >
              <Trash2 size={15} />
              <span className="sr-only sm:not-sr-only">Delete</span>
            </Button>
          </div>
        </div>
        {actionError && <p className="mt-2 text-sm text-bad">{actionError}</p>}
        {p.status === "failed" && p.error && (
          <div className="mt-4 rounded-xl border border-line bg-surface px-4 py-3 text-sm text-bad">{p.error}</div>
        )}
      </div>

      <nav className="flex gap-1 border-b border-line" aria-label="Project sections">
        {TABS.map((t) => (
          <NavLink
            key={t.id}
            to={`/projects/${id}/${t.id}`}
            className={() =>
              cx(
                "-mb-px border-b-2 px-3.5 py-2.5 text-sm font-medium transition",
                tab === t.id ? "border-accent text-ink" : "border-transparent text-muted hover:text-ink",
              )
            }
          >
            {t.label}
          </NavLink>
        ))}
      </nav>

      {tab === "pipeline" && <PipelineTab project={p} onAction={act} busy={busy} />}
      {tab === "overview" && <OverviewTab projectId={id} ready={p.stages_done === p.stages_total} />}
      {tab === "developers" && <DevelopersTab projectId={id} ready={p.stages_done === p.stages_total} />}
    </div>
  );
}
