import { BarChart3, ShieldAlert } from "lucide-react";
import { useMemo, useState } from "react";
import { api, usePoll } from "../api";
import { HourHeatmap, Legend, MonthlyBars, MonthlyLines, type SeriesDef } from "../components/charts";
import { Card, CardHeader, Empty, StatTile, cx, fmt } from "../components/ui";

const RANGES = [
  { id: "all", label: "All time", months: Infinity },
  { id: "5y", label: "5 years", months: 60 },
  { id: "2y", label: "2 years", months: 24 },
  { id: "1y", label: "12 months", months: 12 },
];

const COMMUNICATION: SeriesDef[] = [
  { key: "comments", label: "Comments" },
  { key: "reviews", label: "Reviews" },
];
const PEOPLE: SeriesDef[] = [
  { key: "active", label: "Active developers" },
  { key: "core", label: "Core developers" },
];
const OFF_HOURS: SeriesDef[] = [{ key: "off_hours_share", label: "Off-hours commits", format: fmt.pct }];
const CLIMATE: SeriesDef[] = [
  { key: "we_share", label: "Belonging (we-talk)", format: fmt.pct },
  { key: "thanks_received_share", label: "Gratitude in replies", format: fmt.pct },
  { key: "negative_share", label: "Negative messages", format: fmt.pct },
  { key: "anger_share", label: "Anger", format: fmt.pct },
  { key: "joy_share", label: "Joy", format: fmt.pct },
];

export default function OverviewTab({ projectId, ready }: { projectId: number; ready: boolean }) {
  const { data, error } = usePoll(() => api.overview(projectId), ready ? 0 : 10000, [projectId, ready]);
  const [range, setRange] = useState("all");

  const monthly = useMemo(() => {
    if (!data) return [];
    const months = RANGES.find((r) => r.id === range)!.months;
    return Number.isFinite(months) ? data.monthly.slice(-months) : data.monthly;
  }, [data, range]);

  if (error) return <p className="text-sm text-bad">{error}</p>;
  if (!data) return <div className="h-64 animate-pulse rounded-2xl bg-raised" />;
  if (!data.monthly.length)
    return (
      <Card>
        <Empty icon={<BarChart3 size={22} />} title="No metrics yet">
          Metrics appear once the pipeline reaches the aggregation stage.
        </Empty>
      </Card>
    );

  const t = data.totals;
  const tf = data.truck_factor;

  return (
    <div className="space-y-6">
      {!ready && (
        <p className="rounded-xl border border-line bg-surface px-4 py-3 text-sm text-ink2">
          The pipeline is not finished: figures reflect the last completed computation.
        </p>
      )}

      <div className="grid grid-cols-2 gap-4 md:grid-cols-4">
        <StatTile label="Commits" value={fmt.int(t.commits)} />
        <StatTile label="Issues · PRs" value={`${fmt.int(t.issues)} · ${fmt.int(t.prs)}`} />
        <StatTile label="Developers" value={fmt.int(t.developers)} hint={`${t.bots} bot accounts excluded`} />
        <StatTile label="Off-hours commits" value={fmt.pct(t.off_hours_share)} hint="weekends or outside working hours" />
      </div>

      <div className="flex flex-wrap items-center justify-between gap-3">
        <h2 className="text-sm font-semibold text-ink">Monthly activity</h2>
        <div className="inline-flex rounded-lg border border-line bg-surface p-0.5" role="group" aria-label="Time range">
          {RANGES.map((r) => (
            <button
              key={r.id}
              onClick={() => setRange(r.id)}
              aria-pressed={range === r.id}
              className={cx(
                "rounded-md px-3 py-1.5 text-xs font-medium transition",
                range === r.id ? "bg-accent-soft text-accent" : "text-ink2 hover:text-ink",
              )}
            >
              {r.label}
            </button>
          ))}
        </div>
      </div>

      <div className="grid gap-6 lg:grid-cols-2">
        <Card>
          <CardHeader title="Commits per month" subtitle="Human authors only" />
          <div className="px-3 pb-4 pt-3">
            <MonthlyBars data={monthly} series={{ key: "commits", label: "Commits" }} />
          </div>
        </Card>
        <Card>
          <CardHeader title="Communication per month" subtitle="Issue/PR comments and submitted reviews" right={<Legend series={COMMUNICATION} />} />
          <div className="px-3 pb-4 pt-3">
            <MonthlyLines data={monthly} series={COMMUNICATION} />
          </div>
        </Card>
        <Card>
          <CardHeader
            title="Active and core developers"
            subtitle="Core: top 80% of cumulative network centrality (3-month window)"
            right={<Legend series={PEOPLE} />}
          />
          <div className="px-3 pb-4 pt-3">
            <MonthlyLines data={monthly} series={PEOPLE} />
          </div>
        </Card>
        <Card>
          <CardHeader title="Off-hours commit share" subtitle="Share of each month's commits made on weekends or outside working hours" />
          <div className="px-3 pb-4 pt-3">
            <MonthlyLines data={monthly} series={OFF_HOURS} percent />
          </div>
        </Card>
        <Card className="lg:col-span-2">
          <CardHeader
            title="Psychological climate"
            subtitle="We-words among first-person pronouns, replies by others that thank or appreciate, negative messages (Senti4SD), anger and joy (EmoTxt); pooled over each month's human messages"
            right={<Legend series={CLIMATE} />}
          />
          <div className="px-3 pb-4 pt-3">
            <MonthlyLines data={monthly} series={CLIMATE} percent />
          </div>
        </Card>
      </div>

      <div className="grid gap-6 lg:grid-cols-3">
        <Card className="lg:col-span-2">
          <CardHeader title="When commits happen" subtitle="All commits by weekday and hour" />
          <div className="px-5 pb-5 pt-4">
            <HourHeatmap data={data.heatmap} workHours={data.work_hours} />
          </div>
        </Card>

        <Card>
          <CardHeader title="Truck factor" subtitle="Avelino et al. (DOA-based)" />
          {tf ? (
            <div className="px-5 pb-5 pt-3">
              <div className="flex items-baseline gap-3">
                <span className="text-5xl font-semibold tracking-tight text-ink">{tf.value}</span>
                {tf.value <= 1 && (
                  <span className="inline-flex items-center gap-1 text-xs font-medium text-warn">
                    <ShieldAlert size={14} /> Single point of failure
                  </span>
                )}
              </div>
              <p className="mt-1 text-xs text-muted">
                Developers whose departure would orphan over half of the {fmt.int(tf.files)} files
                ({fmt.int(tf.files_with_author)} have an identifiable author).
              </p>
              <ul className="mt-4 space-y-2">
                {tf.members.map((m) => (
                  <li key={m.id} className="flex items-center justify-between rounded-lg bg-raised px-3 py-2 text-sm">
                    <span className="truncate font-medium text-ink">{m.name}</span>
                    <span className="tabular shrink-0 text-xs text-ink2">{fmt.int(m.files)} files</span>
                  </li>
                ))}
              </ul>
            </div>
          ) : (
            <p className="px-5 pb-5 pt-3 text-sm text-muted">Computed in the structural-metrics stage.</p>
          )}
        </Card>
      </div>
    </div>
  );
}
