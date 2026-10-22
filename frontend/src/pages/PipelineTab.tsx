import { RotateCcw } from "lucide-react";
import { api, usePoll, type ProjectDetail } from "../api";
import { Button, Card, CardHeader, ProgressBar, StageIcon, cx, fmt } from "../components/ui";

function duration(start: string | null, end: string | null) {
  if (!start) return null;
  const sec = ((end ? new Date(end) : new Date()).getTime() - new Date(start).getTime()) / 1000;
  if (sec < 60) return `${Math.round(sec)}s`;
  if (sec < 3600) return `${Math.floor(sec / 60)}m ${Math.round(sec % 60)}s`;
  return `${Math.floor(sec / 3600)}h ${Math.round((sec % 3600) / 60)}m`;
}

export default function PipelineTab({
  project,
  onAction,
  busy,
}: {
  project: ProjectDetail;
  onAction: (fn: () => Promise<unknown>) => void;
  busy: boolean;
}) {
  const { data: logs } = usePoll(() => api.logs(project.id), 3000, [project.id]);
  const running = project.status === "running";

  return (
    <div className="grid gap-6 lg:grid-cols-5">
      <Card className="lg:col-span-3">
        <CardHeader
          title="Stages"
          subtitle="Each stage saves its cursor at every checkpoint, so you can pause, stop the worker and resume later."
        />
        <ol className="mt-4 px-5 pb-3">
          {project.stages.map((s, i) => {
            const last = i === project.stages.length - 1;
            return (
              <li key={s.name} className="relative flex gap-4 pb-5">
                {!last && (
                  <span
                    className={cx("absolute left-4 top-9 -ml-px h-[calc(100%-2.25rem)] w-0.5", s.status === "done" ? "bg-good/40" : "bg-line")}
                    aria-hidden
                  />
                )}
                <StageIcon status={s.status} />
                <div className="min-w-0 flex-1 pt-0.5">
                  <div className="flex items-start justify-between gap-3">
                    <div className="min-w-0">
                      <div className="text-sm font-medium text-ink">{s.label}</div>
                      <div className="text-xs text-muted">{s.description}</div>
                    </div>
                    <div className="flex shrink-0 items-center gap-2">
                      {duration(s.started_at, s.finished_at) && (
                        <span className="tabular text-xs text-muted">{duration(s.started_at, s.finished_at)}</span>
                      )}
                      <Button
                        variant="ghost"
                        className="px-2 py-1"
                        title={running ? "Pause the project to re-run a stage" : `Re-run from ${s.label}`}
                        aria-label={`Re-run from ${s.label}`}
                        disabled={busy || running || s.status === "pending"}
                        onClick={() => {
                          if (confirm(`Re-run "${s.label}" and every later stage?`))
                            onAction(() => api.rerun(project.id, s.name));
                        }}
                      >
                        <RotateCcw size={14} />
                      </Button>
                    </div>
                  </div>
                  {(s.status === "running" || s.status === "paused" || (s.total && s.status !== "done")) && (
                    <div className="mt-2.5">
                      <ProgressBar value={s.done} max={s.total} active={s.status === "running"} />
                    </div>
                  )}
                  {s.message && (
                    <div
                      className={cx("mt-1.5 text-xs", s.status === "failed" ? "text-bad" : "text-ink2")}
                    >
                      {s.message}
                    </div>
                  )}
                </div>
              </li>
            );
          })}
        </ol>
      </Card>

      <Card className="flex flex-col lg:col-span-2">
        <CardHeader title="Activity log" subtitle="Most recent first" />
        <div className="mt-3 max-h-[640px] flex-1 overflow-y-auto px-2 pb-3">
          {!logs?.length && <p className="px-3 py-6 text-center text-sm text-muted">No log entries yet.</p>}
          <ul>
            {logs?.map((l) => (
              <li key={l.id} className="rounded-lg px-3 py-2 hover:bg-raised">
                <div className="flex items-center gap-2 text-[11px] text-muted">
                  <span className="tabular">{new Date(l.ts).toLocaleTimeString()}</span>
                  {l.stage && <span className="rounded bg-raised px-1.5 py-px">{l.stage}</span>}
                  {l.level !== "info" && (
                    <span className={l.level === "error" ? "font-medium text-bad" : "font-medium text-warn"}>
                      {l.level.toUpperCase()}
                    </span>
                  )}
                  <span className="ml-auto">{fmt.ago(l.ts)}</span>
                </div>
                <pre
                  className={cx(
                    "mt-0.5 whitespace-pre-wrap break-words font-sans text-[13px]",
                    l.level === "error" ? "max-h-40 overflow-y-auto font-mono text-xs text-bad" : "text-ink2",
                  )}
                >
                  {l.message}
                </pre>
              </li>
            ))}
          </ul>
        </div>
      </Card>
    </div>
  );
}
