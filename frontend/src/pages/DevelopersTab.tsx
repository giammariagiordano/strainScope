import { ArrowDown, ArrowUp, Search, Users, X } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { api, usePoll, type DeveloperRow } from "../api";
import { Legend, MonthlyBars, MonthlyLines, type SeriesDef } from "../components/charts";
import { Card, Empty, cx, fmt } from "../components/ui";

type SortKey = keyof Pick<
  DeveloperRow,
  "name" | "core_months" | "active_months" | "commits" | "off_hours_share" | "comments" | "reviews" | "files_authored" | "last_activity"
>;

const COLUMNS: { key: SortKey; label: string; numeric?: boolean; title?: string }[] = [
  { key: "name", label: "Developer" },
  { key: "core_months", label: "Core months", numeric: true, title: "Months classified as core" },
  { key: "active_months", label: "Active months", numeric: true },
  { key: "commits", label: "Commits", numeric: true },
  { key: "off_hours_share", label: "Off-hours", numeric: true, title: "Share of commits on weekends or outside working hours" },
  { key: "comments", label: "Comments", numeric: true },
  { key: "reviews", label: "Reviews", numeric: true },
  { key: "files_authored", label: "Files authored", numeric: true, title: "Files where the developer is an author (DOA)" },
  { key: "last_activity", label: "Last active" },
];

const COMMS: SeriesDef[] = [
  { key: "comments", label: "Comments" },
  { key: "reviews", label: "Reviews" },
];

function DeveloperPanel({ projectId, devId, onClose }: { projectId: number; devId: number; onClose: () => void }) {
  const { data } = usePoll(() => api.developer(projectId, devId), 0, [projectId, devId]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  const months = data?.months ?? [];
  const activeMonths = months.filter((m) => m.active);
  const totals = months.reduce(
    (a, m) => ({ commits: a.commits + m.commits, off: a.off + m.off_hours_commits, core: a.core + Number(m.is_core) }),
    { commits: 0, off: 0, core: 0 },
  );

  return (
    <div className="fixed inset-0 z-40 flex justify-end" role="dialog" aria-modal="true" aria-label="Developer details">
      <button className="absolute inset-0 bg-black/30 backdrop-blur-[1px]" onClick={onClose} aria-label="Close" />
      <div className="relative h-full w-full max-w-2xl overflow-y-auto border-l border-line bg-page shadow-2xl">
        <div className="sticky top-0 z-10 flex items-center justify-between border-b border-line bg-page/90 px-6 py-4 backdrop-blur">
          <div>
            <div className="text-xs text-muted">Developer</div>
            <div className="text-lg font-semibold text-ink">{data?.name ?? "…"}</div>
          </div>
          <button onClick={onClose} className="rounded-lg p-2 text-muted hover:bg-raised hover:text-ink" aria-label="Close">
            <X size={18} />
          </button>
        </div>
        {data && (
          <div className="space-y-5 p-6">
            <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
              {[
                ["Commits", fmt.int(totals.commits)],
                ["Off-hours", fmt.pct(totals.commits ? totals.off / totals.commits : null)],
                ["Core months", `${totals.core} / ${activeMonths.length}`],
                ["Truck factor", data.truck_factor_member ? `Member · ${data.files_authored} files` : "Not a member"],
              ].map(([label, value]) => (
                <div key={label} className="rounded-xl border border-line bg-surface px-3.5 py-3">
                  <div className="text-[11px] font-medium uppercase tracking-wide text-muted">{label}</div>
                  <div className="mt-1 text-sm font-semibold text-ink">{value}</div>
                </div>
              ))}
            </div>

            <Card>
              <div className="px-4 pt-4 text-sm font-semibold text-ink">Role over time</div>
              <div className="px-4 pt-1 text-xs text-muted">Each cell is an active month; filled = core</div>
              <div className="flex flex-wrap gap-[2px] px-4 pb-4 pt-3">
                {activeMonths.map((m) => (
                  <span
                    key={m.month}
                    title={`${fmt.month(m.month)} — ${m.is_core ? "core" : "peripheral"}, centrality ${(m.centrality ?? 0).toFixed(3)}`}
                    className={cx("h-4 w-3 rounded-[3px]", m.is_core ? "bg-accent" : "bg-raised ring-1 ring-inset ring-line")}
                  />
                ))}
              </div>
            </Card>

            <Card>
              <div className="px-4 pt-4 text-sm font-semibold text-ink">Commits per month</div>
              <div className="px-2 pb-3 pt-2">
                <MonthlyBars data={months} series={{ key: "commits", label: "Commits" }} height={180} />
              </div>
            </Card>

            <Card>
              <div className="flex items-center justify-between px-4 pt-4">
                <span className="text-sm font-semibold text-ink">Communication per month</span>
                <Legend series={COMMS} />
              </div>
              <div className="px-2 pb-3 pt-2">
                <MonthlyLines data={months} series={COMMS} height={180} />
              </div>
            </Card>

            <Card>
              <div className="px-4 pt-4 text-sm font-semibold text-ink">Network centrality</div>
              <div className="px-4 text-xs text-muted">Eigenvector centrality in the developer network (3-month window)</div>
              <div className="px-2 pb-3 pt-2">
                <MonthlyLines
                  data={months}
                  series={[{ key: "centrality", label: "Centrality", format: (v) => v.toFixed(3) }]}
                  height={180}
                  unitInterval
                />
              </div>
            </Card>
          </div>
        )}
      </div>
    </div>
  );
}

export default function DevelopersTab({ projectId, ready }: { projectId: number; ready: boolean }) {
  const { data, error } = usePoll(() => api.developers(projectId), ready ? 0 : 10000, [projectId, ready]);
  const [query, setQuery] = useState("");
  const [sort, setSort] = useState<{ key: SortKey; desc: boolean }>({ key: "core_months", desc: true });
  const [selected, setSelected] = useState<number | null>(null);
  const [onlyTf, setOnlyTf] = useState(false);

  const rows = useMemo(() => {
    const list = (data?.developers ?? []).filter(
      (d) => d.name.toLowerCase().includes(query.toLowerCase()) && (!onlyTf || d.truck_factor_member),
    );
    const dir = sort.desc ? -1 : 1;
    return [...list].sort((a, b) => {
      const va = a[sort.key] ?? -Infinity;
      const vb = b[sort.key] ?? -Infinity;
      return (va < vb ? -1 : va > vb ? 1 : 0) * dir;
    });
  }, [data, query, sort, onlyTf]);

  if (error) return <p className="text-sm text-bad">{error}</p>;
  if (!data) return <div className="h-64 animate-pulse rounded-2xl bg-raised" />;
  if (!data.developers.length)
    return (
      <Card>
        <Empty icon={<Users size={22} />} title="No developers yet">
          Developers appear after the identity-merge and aggregation stages.
        </Empty>
      </Card>
    );

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-center gap-3">
        <div className="relative min-w-[220px] flex-1 sm:max-w-xs">
          <Search size={15} className="pointer-events-none absolute left-3 top-1/2 -translate-y-1/2 text-muted" />
          <input
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Search developers"
            aria-label="Search developers"
            className="w-full rounded-lg border border-line bg-surface py-2 pl-9 pr-3 text-sm text-ink placeholder:text-muted focus:border-accent focus:outline-none focus:ring-2 focus:ring-accent-soft"
          />
        </div>
        <label className="flex cursor-pointer items-center gap-2 text-sm text-ink2">
          <input type="checkbox" checked={onlyTf} onChange={(e) => setOnlyTf(e.target.checked)} className="accent-[var(--accent)]" />
          Truck-factor members only
        </label>
        <span className="ml-auto text-xs text-muted">
          {rows.length} of {data.developers.length} developers · bots excluded
        </span>
      </div>

      <Card className="overflow-hidden">
        <div className="overflow-x-auto">
          <table className="w-full min-w-[900px] text-sm">
            <thead>
              <tr className="border-b border-line bg-raised/60 text-left text-xs text-muted">
                {COLUMNS.map((c) => (
                  <th key={c.key} className={cx("px-4 py-2.5 font-medium", c.numeric && "text-right")} title={c.title}>
                    <button
                      className={cx("inline-flex items-center gap-1 hover:text-ink", sort.key === c.key && "text-ink")}
                      onClick={() =>
                        setSort((s) => ({ key: c.key, desc: s.key === c.key ? !s.desc : c.key !== "name" }))
                      }
                    >
                      {c.label}
                      {sort.key === c.key && (sort.desc ? <ArrowDown size={12} /> : <ArrowUp size={12} />)}
                    </button>
                  </th>
                ))}
              </tr>
            </thead>
            <tbody className="tabular">
              {rows.map((d) => {
                const corePct = d.active_months ? d.core_months / d.active_months : 0;
                return (
                  <tr
                    key={d.id}
                    onClick={() => setSelected(d.id)}
                    className="cursor-pointer border-b border-line last:border-0 hover:bg-raised"
                  >
                    <td className="px-4 py-2.5">
                      <div className="flex items-center gap-2">
                        <span className="font-medium text-ink">{d.name}</span>
                        {d.truck_factor_member && (
                          <span className="rounded-full bg-accent-soft px-2 py-0.5 text-[10px] font-semibold uppercase tracking-wide text-accent">
                            Truck factor
                          </span>
                        )}
                      </div>
                    </td>
                    <td className="px-4 py-2.5 text-right">
                      <div className="flex items-center justify-end gap-2">
                        <span className="h-1.5 w-12 overflow-hidden rounded-full bg-raised">
                          <span className="block h-full rounded-full bg-accent" style={{ width: `${corePct * 100}%` }} />
                        </span>
                        <span className="w-6 text-ink">{d.core_months}</span>
                      </div>
                    </td>
                    <td className="px-4 py-2.5 text-right text-ink2">{d.active_months}</td>
                    <td className="px-4 py-2.5 text-right text-ink">{fmt.int(d.commits)}</td>
                    <td className="px-4 py-2.5 text-right text-ink2">{fmt.pct(d.off_hours_share)}</td>
                    <td className="px-4 py-2.5 text-right text-ink2">{fmt.int(d.comments)}</td>
                    <td className="px-4 py-2.5 text-right text-ink2">{fmt.int(d.reviews)}</td>
                    <td className="px-4 py-2.5 text-right text-ink2">{fmt.int(d.files_authored)}</td>
                    <td className="whitespace-nowrap px-4 py-2.5 text-ink2">{fmt.date(d.last_activity)}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </Card>

      {selected !== null && <DeveloperPanel projectId={projectId} devId={selected} onClose={() => setSelected(null)} />}
    </div>
  );
}
