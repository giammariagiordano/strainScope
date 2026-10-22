import { Download, FileJson, FileSpreadsheet, FileText, ShieldCheck } from "lucide-react";
import { useState } from "react";
import { api, exportUrl, usePoll } from "../api";
import { Card, CardHeader, StatusBadge, cx, fmt } from "../components/ui";

const FILES = [
  { name: "projects.csv", icon: FileSpreadsheet, text: "One row per repository: size, pipeline status, truck factor." },
  { name: "developers.csv", icon: FileSpreadsheet, text: "One row per developer × project: role, activity, off-hours share, centrality." },
  { name: "developer_months.csv", icon: FileSpreadsheet, text: "Panel: one row per active developer × project × month." },
  { name: "codebook.md", icon: FileText, text: "Definition of every column and the parameters used." },
  { name: "metadata.json", icon: FileJson, text: "Export time, projects, row counts, parameters." },
];

const linkCls =
  "inline-flex items-center justify-center gap-1.5 rounded-lg px-3 py-2 text-sm font-medium transition";

export default function ExportPage() {
  const { data: projects } = usePoll(api.projects, 5000);
  const [selected, setSelected] = useState<number[]>([]);
  const toggle = (id: number) => setSelected((s) => (s.includes(id) ? s.filter((x) => x !== id) : [...s, id]));

  return (
    <div className="space-y-6">
      <header>
        <h1 className="text-2xl font-semibold tracking-tight text-ink">Export</h1>
        <p className="mt-1 text-sm text-ink2">
          Download analysis-ready, pseudonymized datasets for one repository, a selection, or everything.
        </p>
      </header>

      <div className="grid gap-6 lg:grid-cols-5">
        <Card className="lg:col-span-3">
          <CardHeader
            title="Repositories"
            subtitle="Each download is a ZIP with the same structure, whatever the selection."
            right={
              <div className="flex gap-2">
                <a
                  href={selected.length ? exportUrl(selected) : undefined}
                  download
                  aria-disabled={!selected.length}
                  className={cx(
                    linkCls,
                    "border border-line bg-surface text-ink hover:bg-raised",
                    !selected.length && "pointer-events-none opacity-50",
                  )}
                >
                  <Download size={15} /> Selected ({selected.length})
                </a>
                <a href={exportUrl()} download className={cx(linkCls, "bg-accent text-white hover:opacity-90")}>
                  <Download size={15} /> All
                </a>
              </div>
            }
          />
          <ul className="mt-4 divide-y divide-line border-t border-line">
            {projects?.map((p) => (
              <li key={p.id} className="flex items-center gap-3 px-5 py-3">
                <input
                  type="checkbox"
                  aria-label={`Select ${p.full_name}`}
                  checked={selected.includes(p.id)}
                  onChange={() => toggle(p.id)}
                  className="accent-[var(--accent)]"
                />
                <div className="min-w-0 flex-1">
                  <div className="truncate text-sm font-medium text-ink">{p.full_name}</div>
                  <div className="tabular text-xs text-muted">
                    {fmt.int(p.counts.commits)} commits · {fmt.int(p.counts.developers)} developers
                  </div>
                </div>
                <StatusBadge status={p.status} />
                <a
                  href={exportUrl([p.id])}
                  download
                  title={`Download ${p.full_name}`}
                  className={cx(linkCls, "px-2 text-ink2 hover:bg-raised hover:text-ink")}
                >
                  <Download size={15} />
                </a>
              </li>
            ))}
            {projects?.length === 0 && <li className="px-5 py-8 text-center text-sm text-muted">No projects yet.</li>}
          </ul>
          <p className="px-5 py-3 text-xs text-muted">
            Projects that are still running export what has been computed so far.
          </p>
        </Card>

        <div className="space-y-6 lg:col-span-2">
          <Card>
            <CardHeader title="What's inside" />
            <ul className="space-y-3 px-5 pb-5 pt-4">
              {FILES.map(({ name, icon: Icon, text }) => (
                <li key={name} className="flex gap-3">
                  <Icon size={16} className="mt-0.5 shrink-0 text-accent" />
                  <div>
                    <div className="font-mono text-xs font-medium text-ink">{name}</div>
                    <div className="text-xs text-ink2">{text}</div>
                  </div>
                </li>
              ))}
            </ul>
          </Card>
          <Card className="px-5 py-4">
            <div className="flex gap-3 text-xs text-ink2">
              <ShieldCheck size={16} className="mt-0.5 shrink-0 text-good" />
              <p>
                Exports never contain names, logins, emails or message text — only pseudonyms and numbers, even when
                real identities are shown in the UI. Pseudonymized data is still personal data: store it accordingly.
              </p>
            </div>
          </Card>
        </div>
      </div>
    </div>
  );
}
