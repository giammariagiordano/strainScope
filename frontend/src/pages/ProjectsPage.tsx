import { FolderGit2, GitCommitHorizontal, MessageSquare, Pause, Play, Plus, Trash2, Users } from "lucide-react";
import { useState, type MouseEvent } from "react";
import { Link } from "react-router-dom";
import { api, usePoll, type ProjectSummary } from "../api";
import { Button, Card, Empty, ProgressBar, StatusBadge, fmt } from "../components/ui";

function AddProjects({ onAdded }: { onAdded: () => void }) {
  const [text, setText] = useState("");
  const [since, setSince] = useState("");
  const [busy, setBusy] = useState(false);
  const [msg, setMsg] = useState<{ kind: "ok" | "err"; text: string } | null>(null);
  const urls = text.split(/[\s,]+/).filter(Boolean);

  async function submit() {
    setBusy(true);
    setMsg(null);
    try {
      const r = await api.addProjects(urls, since || null);
      const parts = [];
      if (r.created.length) parts.push(`Queued ${r.created.length} project${r.created.length > 1 ? "s" : ""}`);
      if (r.skipped.length) parts.push(`${r.skipped.length} already present`);
      setMsg({ kind: "ok", text: parts.join(" · ") });
      setText("");
      onAdded();
    } catch (e) {
      setMsg({ kind: "err", text: (e as Error).message });
    } finally {
      setBusy(false);
    }
  }

  return (
    <Card className="p-5">
      <label htmlFor="urls" className="text-sm font-semibold text-ink">
        Add repositories
      </label>
      <p className="mt-0.5 text-xs text-muted">One GitHub URL or owner/name per line. Mining starts automatically.</p>
      <textarea
        id="urls"
        value={text}
        onChange={(e) => setText(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === "Enter" && (e.metaKey || e.ctrlKey) && urls.length) submit();
        }}
        rows={3}
        placeholder={"https://github.com/pallets/flask\npsf/requests"}
        className="mt-3 w-full resize-y rounded-xl border border-line bg-page px-3.5 py-2.5 font-mono text-sm text-ink placeholder:text-muted focus:border-accent focus:outline-none focus:ring-2 focus:ring-accent-soft"
      />
      <div className="mt-3 flex flex-wrap items-center gap-x-3 gap-y-1">
        <label htmlFor="since" className="text-sm text-ink2">
          Mine GitHub data from
        </label>
        <input
          id="since"
          type="date"
          value={since}
          onChange={(e) => setSince(e.target.value)}
          className="rounded-lg border border-line bg-page px-2.5 py-1.5 text-sm text-ink focus:border-accent focus:outline-none focus:ring-2 focus:ring-accent-soft"
        />
        {since && (
          <button type="button" onClick={() => setSince("")} className="text-xs text-muted hover:text-ink">
            Clear
          </button>
        )}
        <span className="basis-full text-xs text-muted">
          Optional. Issues, comments, reviews and PR outcomes from this date only (much faster on large projects);
          the git history is always complete. Empty = everything.
        </span>
      </div>
      <div className="mt-3 flex flex-wrap items-center justify-between gap-3">
        <span className={msg?.kind === "err" ? "text-sm text-bad" : "text-sm text-ink2"}>{msg?.text}</span>
        <Button variant="primary" onClick={submit} disabled={!urls.length || busy}>
          <Plus size={16} /> {urls.length > 1 ? `Add ${urls.length} repositories` : "Add repository"}
        </Button>
      </div>
    </Card>
  );
}

function ProjectCard({ p, onChange }: { p: ProjectSummary; onChange: () => void }) {
  const active = p.status === "running";
  const [busy, setBusy] = useState(false);
  const canPause = (p.status === "running" && !p.pause_requested) || p.status === "queued";
  const canResume = p.status === "paused" || p.status === "failed";

  // The card is a link: actions must not navigate.
  const act = (fn: () => Promise<unknown>, confirmText?: string) => async (e: MouseEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (confirmText && !confirm(confirmText)) return;
    setBusy(true);
    try {
      await fn();
    } catch (err) {
      alert((err as Error).message);
    } finally {
      setBusy(false);
      onChange();
    }
  };

  return (
    <Link to={`/projects/${p.id}`} className="group block">
      <Card className="flex h-full flex-col p-5 transition group-hover:border-accent/40 group-hover:shadow-lg">
        <div className="flex items-start justify-between gap-3">
          <div className="min-w-0">
            <div className="truncate text-xs text-muted">
              {p.full_name.split("/")[0]}
              {p.since && <span title="GitHub API data mined from this date"> · from {p.since}</span>}
            </div>
            <div className="truncate text-base font-semibold text-ink">{p.full_name.split("/")[1]}</div>
          </div>
          <StatusBadge status={p.status} pausing={p.pause_requested && active} />
        </div>

        <div className="mt-5">
          <div className="mb-1.5 flex items-baseline justify-between text-xs">
            <span className="truncate text-ink2">
              {p.status === "done"
                ? `Completed ${p.finished_at ? fmt.ago(p.finished_at) : ""}`
                : p.status === "failed"
                  ? p.error
                  : p.status === "deleting"
                    ? "Stopping at the next checkpoint, then deleting"
                    : (p.current_stage_label ?? "Waiting for worker")}
            </span>
            <span className="tabular shrink-0 pl-2 text-muted">
              {p.stages_done}/{p.stages_total}
            </span>
          </div>
          <ProgressBar value={p.stages_done} max={p.stages_total} />
          {active && p.current_message && <div className="mt-1.5 truncate text-xs text-muted">{p.current_message}</div>}
        </div>

        <div className="mt-auto flex items-center gap-2 border-t border-line pt-4 text-xs text-ink2">
          <span className="flex items-center gap-1.5" title="Commits">
            <GitCommitHorizontal size={14} className="text-muted" /> <span className="tabular">{fmt.int(p.counts.commits)}</span>
          </span>
          <span className="ml-2 flex items-center gap-1.5" title="Issues and PRs">
            <MessageSquare size={14} className="text-muted" /> <span className="tabular">{fmt.int(p.counts.issues)}</span>
          </span>
          <span className="ml-2 flex items-center gap-1.5" title="Developers (excluding bots)">
            <Users size={14} className="text-muted" /> <span className="tabular">{fmt.int(p.counts.developers)}</span>
          </span>
          {p.status !== "deleting" && (
            <span className="ml-auto flex gap-0.5">
              {canPause && (
                <Button variant="ghost" className="px-2 py-1" title="Pause" aria-label={`Pause ${p.full_name}`} disabled={busy} onClick={act(() => api.pause(p.id))}>
                  <Pause size={14} />
                </Button>
              )}
              {canResume && (
                <Button variant="ghost" className="px-2 py-1" title="Resume" aria-label={`Resume ${p.full_name}`} disabled={busy} onClick={act(() => api.resume(p.id))}>
                  <Play size={14} />
                </Button>
              )}
              <Button
                variant="ghost"
                className="px-2 py-1 hover:text-bad"
                title="Delete"
                aria-label={`Delete ${p.full_name}`}
                disabled={busy}
                onClick={act(() => api.remove(p.id), `Delete ${p.full_name} and all its mined data? This cannot be undone.`)}
              >
                <Trash2 size={14} />
              </Button>
            </span>
          )}
        </div>
      </Card>
    </Link>
  );
}

function DeleteAllDialog({ count, onClose, onDone }: { count: number; onClose: () => void; onDone: (msg: string) => void }) {
  const [text, setText] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  async function run() {
    setBusy(true);
    try {
      const r = await api.removeAll();
      onDone(
        `Deleted ${r.deleted} project${r.deleted === 1 ? "" : "s"}` +
          (r.deleting ? `; ${r.deleting} running will be deleted at the next checkpoint` : ""),
      );
    } catch (e) {
      setError((e as Error).message);
      setBusy(false);
    }
  }
  return (
    <div className="fixed inset-0 z-50 grid place-items-center p-4" role="dialog" aria-modal="true" aria-labelledby="delete-all-title">
      <button className="absolute inset-0 bg-black/40" onClick={onClose} aria-label="Cancel" />
      <Card className="relative w-full max-w-md p-6">
        <div className="mb-3 grid h-10 w-10 place-items-center rounded-xl bg-raised text-bad">
          <Trash2 size={18} />
        </div>
        <h2 id="delete-all-title" className="text-base font-semibold text-ink">
          Delete all {count} projects?
        </h2>
        <p className="mt-1 text-sm text-ink2">
          This removes every project — queued, running, paused and completed — with all mined data and repository
          clones. Running projects stop at their next checkpoint. This cannot be undone.
        </p>
        <label htmlFor="confirm-delete" className="mt-4 block text-xs text-muted">
          Type <span className="font-mono font-medium text-ink">delete</span> to confirm
        </label>
        <input
          id="confirm-delete"
          autoFocus
          value={text}
          onChange={(e) => setText(e.target.value)}
          onKeyDown={(e) => e.key === "Escape" && onClose()}
          className="mt-1 w-full rounded-lg border border-line bg-page px-3 py-2 text-sm text-ink focus:border-accent focus:outline-none"
        />
        {error && <p className="mt-2 text-sm text-bad">{error}</p>}
        <div className="mt-5 flex justify-end gap-2">
          <Button onClick={onClose} disabled={busy}>
            Cancel
          </Button>
          <Button variant="danger" onClick={run} disabled={text.trim().toLowerCase() !== "delete" || busy}>
            <Trash2 size={15} /> {busy ? "Deleting…" : "Delete all"}
          </Button>
        </div>
      </Card>
    </div>
  );
}

export default function ProjectsPage() {
  const { data, error, refresh } = usePoll(api.projects, 2000);
  const [confirmAll, setConfirmAll] = useState(false);
  const [notice, setNotice] = useState<string | null>(null);
  return (
    <div className="space-y-6">
      <header className="flex flex-wrap items-start justify-between gap-4">
        <div>
          <h1 className="text-2xl font-semibold tracking-tight text-ink">Projects</h1>
          <p className="mt-1 text-sm text-ink2">
            Mine repositories and explore developer roles, workload and activity over time.
          </p>
        </div>
        {!!data?.length && (
          <Button variant="danger" onClick={() => setConfirmAll(true)}>
            <Trash2 size={15} /> Delete all
          </Button>
        )}
      </header>
      {notice && <p className="rounded-xl border border-line bg-surface px-4 py-3 text-sm text-ink2">{notice}</p>}
      {confirmAll && data && (
        <DeleteAllDialog
          count={data.length}
          onClose={() => setConfirmAll(false)}
          onDone={(msg) => {
            setConfirmAll(false);
            setNotice(msg);
            refresh();
          }}
        />
      )}

      <AddProjects onAdded={refresh} />

      {error && <p className="text-sm text-bad">Cannot reach the API: {error}</p>}

      {data && data.length === 0 && (
        <Card>
          <Empty icon={<FolderGit2 size={22} />} title="No projects yet">
            Add a GitHub repository above to start the mining pipeline.
          </Empty>
        </Card>
      )}

      {data && data.length > 0 && (
        <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-3">
          {data.map((p) => (
            <ProjectCard key={p.id} p={p} onChange={refresh} />
          ))}
        </div>
      )}
    </div>
  );
}
