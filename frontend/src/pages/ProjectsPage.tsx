import { FolderGit2, GitCommitHorizontal, MessageSquare, Plus, Users } from "lucide-react";
import { useState } from "react";
import { Link } from "react-router-dom";
import { api, usePoll, type ProjectSummary } from "../api";
import { Button, Card, Empty, ProgressBar, StatusBadge, fmt } from "../components/ui";

function AddProjects({ onAdded }: { onAdded: () => void }) {
  const [text, setText] = useState("");
  const [busy, setBusy] = useState(false);
  const [msg, setMsg] = useState<{ kind: "ok" | "err"; text: string } | null>(null);
  const urls = text.split(/[\s,]+/).filter(Boolean);

  async function submit() {
    setBusy(true);
    setMsg(null);
    try {
      const r = await api.addProjects(urls);
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
      <div className="mt-3 flex flex-wrap items-center justify-between gap-3">
        <span className={msg?.kind === "err" ? "text-sm text-bad" : "text-sm text-ink2"}>{msg?.text}</span>
        <Button variant="primary" onClick={submit} disabled={!urls.length || busy}>
          <Plus size={16} /> {urls.length > 1 ? `Add ${urls.length} repositories` : "Add repository"}
        </Button>
      </div>
    </Card>
  );
}

function ProjectCard({ p }: { p: ProjectSummary }) {
  const active = p.status === "running";
  return (
    <Link to={`/projects/${p.id}`} className="group block">
      <Card className="h-full p-5 transition group-hover:border-accent/40 group-hover:shadow-lg">
        <div className="flex items-start justify-between gap-3">
          <div className="min-w-0">
            <div className="truncate text-xs text-muted">{p.full_name.split("/")[0]}</div>
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
                  : (p.current_stage_label ?? "Waiting for worker")}
            </span>
            <span className="tabular shrink-0 pl-2 text-muted">
              {p.stages_done}/{p.stages_total}
            </span>
          </div>
          <ProgressBar value={p.stages_done} max={p.stages_total} />
          {active && p.current_message && <div className="mt-1.5 truncate text-xs text-muted">{p.current_message}</div>}
        </div>

        <div className="mt-5 grid grid-cols-3 gap-2 border-t border-line pt-4 text-xs text-ink2">
          <span className="flex items-center gap-1.5" title="Commits">
            <GitCommitHorizontal size={14} className="text-muted" /> <span className="tabular">{fmt.int(p.counts.commits)}</span>
          </span>
          <span className="flex items-center gap-1.5" title="Issues and PRs">
            <MessageSquare size={14} className="text-muted" /> <span className="tabular">{fmt.int(p.counts.issues)}</span>
          </span>
          <span className="flex items-center gap-1.5" title="Developers (excluding bots)">
            <Users size={14} className="text-muted" /> <span className="tabular">{fmt.int(p.counts.developers)}</span>
          </span>
        </div>
      </Card>
    </Link>
  );
}

export default function ProjectsPage() {
  const { data, error, refresh } = usePoll(api.projects, 2000);
  return (
    <div className="space-y-6">
      <header>
        <h1 className="text-2xl font-semibold tracking-tight text-ink">Projects</h1>
        <p className="mt-1 text-sm text-ink2">
          Mine repositories and explore developer roles, workload and activity over time.
        </p>
      </header>

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
            <ProjectCard key={p.id} p={p} />
          ))}
        </div>
      )}
    </div>
  );
}
