import { AlertTriangle, HeartPulse, Info } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { CartesianGrid, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { api, usePoll, type CoxCoefficient, type KmGroup, type SurvivalResult } from "../api";
import { Legend, SERIES } from "../components/charts";
import { Check, NumberField, Section, Segmented } from "../components/controls";
import ProjectPicker from "../components/ProjectPicker";
import { Card, CardHeader, Empty, StatTile, cx, fmt } from "../components/ui";

const fmtP = (p: number | null | undefined) => (p == null ? "—" : p < 0.001 ? "<0.001" : p.toFixed(3));
const fmtHr = (v: number | null) => (v == null ? "∞" : v >= 100 || v < 0.01 ? v.toExponential(1) : v.toFixed(2));
const fmtNum = (v: number | null, digits = 3) => (v == null ? "—" : v.toFixed(digits));

/** Hazard ratios with 95% CI on a log axis. Filled dot = p < 0.05. */
function ForestPlot({ rows, labels }: { rows: CoxCoefficient[]; labels: Record<string, string> }) {
  // Diverging estimates (null) are drawn at the axis edge.
  const clamp = (v: number | null) => (v == null ? 20 : Math.min(20, Math.max(0.05, v)));
  const lo = Math.min(0.5, ...rows.map((r) => clamp(r.hr_lower)));
  const hi = Math.max(2, ...rows.map((r) => clamp(r.hr_upper)));
  const W = 640, L = 150, R = 20, rowH = 34, top = 10;
  const H = top + rows.length * rowH + 34;
  const x = (v: number | null) => L + ((Math.log(clamp(v)) - Math.log(lo)) / (Math.log(hi) - Math.log(lo))) * (W - L - R);
  const ticks = [0.05, 0.1, 0.25, 0.5, 1, 2, 4, 10, 20].filter((t) => t >= lo && t <= hi);
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="w-full" role="img" aria-label="Forest plot of hazard ratios">
      {ticks.map((t) => (
        <g key={t}>
          <line x1={x(t)} x2={x(t)} y1={top} y2={H - 30} stroke={t === 1 ? "var(--axis)" : "var(--grid)"} strokeDasharray={t === 1 ? "4 3" : undefined} />
          <text x={x(t)} y={H - 14} textAnchor="middle" fontSize={11} fill="var(--text-muted)">
            {t}
          </text>
        </g>
      ))}
      {rows.map((r, i) => {
        const y = top + i * rowH + rowH / 2;
        const sig = r.p != null && r.p < 0.05;
        return (
          <g key={r.covariate}>
            <title>{`${labels[r.covariate] ?? r.covariate}: HR ${fmtHr(r.hr)} [${fmtHr(r.hr_lower)}, ${fmtHr(r.hr_upper)}], p ${fmtP(r.p)}`}</title>
            <rect x={0} y={y - rowH / 2} width={W} height={rowH} fill="transparent" />
            <text x={L - 12} y={y + 4} textAnchor="end" fontSize={12} fill="var(--text-secondary)">
              {labels[r.covariate] ?? r.covariate}
            </text>
            <line x1={x(r.hr_lower)} x2={x(r.hr_upper)} y1={y} y2={y} stroke="var(--series-1)" strokeWidth={2} strokeLinecap="round" />
            <circle cx={x(r.hr)} cy={y} r={5} fill={sig ? "var(--series-1)" : "var(--surface-1)"} stroke="var(--series-1)" strokeWidth={2} />
          </g>
        );
      })}
      <text x={L} y={H - 1} fontSize={10} fill="var(--text-muted)">
        ← lower risk of abandoning
      </text>
      <text x={W - R} y={H - 1} fontSize={10} textAnchor="end" fill="var(--text-muted)">
        higher risk →
      </text>
    </svg>
  );
}

function KmChart({ groups }: { groups: KmGroup[] }) {
  const maxT = Math.max(1, ...groups.flatMap((g) => g.curve.map((p) => p.t)));
  return (
    <ResponsiveContainer width="100%" height={300}>
      <LineChart margin={{ top: 8, right: 16, bottom: 20, left: -8 }}>
        <CartesianGrid vertical={false} />
        <XAxis
          type="number"
          dataKey="t"
          domain={[0, maxT]}
          tickLine={false}
          axisLine={false}
          fontSize={11}
          allowDuplicatedCategory={false}
          label={{ value: "Months since entry", position: "insideBottom", offset: -10, fontSize: 11, fill: "var(--text-secondary)" }}
        />
        <YAxis domain={[0, 1]} tickFormatter={(v: number) => `${Math.round(v * 100)}%`} tickLine={false} axisLine={false} fontSize={11} width={48} />
        <Tooltip
          cursor={{ stroke: "var(--axis)" }}
          content={({ active, payload, label }) => {
            if (!active || !payload?.length) return null;
            return (
              <div className="rounded-lg border border-line bg-surface px-3 py-2 text-xs shadow-lg">
                <div className="mb-1 font-medium text-ink">Month {label}</div>
                {payload.map((p) => {
                  const pt = p.payload as KmGroup["curve"][number];
                  return (
                    <div key={String(p.name)} className="flex items-center justify-between gap-4 py-0.5">
                      <span className="flex items-center gap-1.5 text-ink2">
                        <span className="h-2 w-2 rounded-full" style={{ background: p.color }} />
                        {p.name}
                      </span>
                      <span className="tabular text-ink">
                        {fmt.pct(pt.s)} <span className="text-muted">[{fmt.pct(pt.lo)}–{fmt.pct(pt.hi)}]</span>
                      </span>
                    </div>
                  );
                })}
              </div>
            );
          }}
        />
        {groups.map((g, i) => (
          <Line
            key={g.label}
            data={g.curve}
            dataKey="s"
            name={g.label}
            type="stepAfter"
            stroke={SERIES[i]}
            strokeWidth={2}
            dot={false}
            activeDot={{ r: 4, strokeWidth: 2, stroke: "var(--surface-1)" }}
            isAnimationActive={false}
          />
        ))}
      </LineChart>
    </ResponsiveContainer>
  );
}

export default function SurvivalPage() {
  const { data: projects } = usePoll(api.projects, 0);
  const { data: options } = usePoll(api.survivalOptions, 0);
  const [projectIds, setProjectIds] = useState<number[]>([]);
  const [abandon, setAbandon] = useState(12);
  const [window, setWindow] = useState(3);
  const [entryAfter, setEntryAfter] = useState(1);
  const [covariates, setCovariates] = useState<string[] | null>(null);
  const [logCounts, setLogCounts] = useState(true);
  const [standardize, setStandardize] = useState(true);
  const [stratify, setStratify] = useState(true);
  const [penalizer, setPenalizer] = useState(0);
  const [group, setGroup] = useState("core");
  const [cluster, setCluster] = useState("person");
  const [phTransform, setPhTransform] = useState("rank");
  const [result, setResult] = useState<SurvivalResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (!options) return;
    setAbandon(options.defaults.abandon_months);
    setCovariates((c) => c ?? options.covariates.filter((x) => x.default).map((x) => x.key));
  }, [options]);

  const labels = useMemo(
    () => Object.fromEntries((options?.covariates ?? []).map((c) => [c.key, c.label])),
    [options],
  );

  useEffect(() => {
    if (!options || !covariates) return;
    const t = setTimeout(async () => {
      setLoading(true);
      try {
        setResult(
          await api.survival({
            project_ids: projectIds.length ? projectIds : null,
            abandon_months: abandon,
            window,
            min_active_months: entryAfter,
            covariates,
            log_counts: logCounts,
            standardize,
            stratify,
            penalizer,
            cluster,
            ph_transform: phTransform,
            group,
          }),
        );
        setError(null);
      } catch (e) {
        setError((e as Error).message);
      } finally {
        setLoading(false);
      }
    }, 400);
    return () => clearTimeout(t);
  }, [options, projectIds, abandon, window, entryAfter, covariates, logCounts, standardize, stratify, penalizer, cluster, phTransform, group]);

  const cox = result && !("error" in result.cox) ? result.cox : null;
  const coxError = result && "error" in result.cox ? result.cox.error : null;
  const ds = result?.dataset;

  return (
    <div className="space-y-6">
      <header>
        <h1 className="text-2xl font-semibold tracking-tight text-ink">Survival analysis</h1>
        <p className="mt-1 text-sm text-ink2">
          Time to abandonment with a Cox model on time-varying covariates, and Kaplan–Meier curves by group.
        </p>
      </header>

      <div className="grid gap-6 lg:grid-cols-[300px_minmax(0,1fr)]">
        <Card className="h-fit space-y-5 p-5 lg:sticky lg:top-6">
          <Section title="Projects">
            {projects && <ProjectPicker projects={projects} selected={projectIds} onChange={setProjectIds} />}
          </Section>
          <Section title="Abandonment threshold" hint="Months of inactivity that define abandonment">
            <NumberField label="Abandonment threshold" value={abandon} onChange={setAbandon} />
          </Section>
          <Section title="Covariate window" hint="Covariates summarize the previous N months (lagged)">
            <NumberField label="Covariate window" value={window} onChange={setWindow} />
          </Section>
          <Section title="Enter risk set after" hint="Active months before a developer is followed (left truncation)">
            <NumberField label="Enter risk set after" value={entryAfter} onChange={setEntryAfter} />
          </Section>
          <Section title="Covariates" hint="Hover for definitions">
            <div className="space-y-0.5">
              {options?.covariates.map((c) => (
                <Check
                  key={c.key}
                  title={c.description}
                  checked={covariates?.includes(c.key) ?? false}
                  onChange={() =>
                    setCovariates((cur) =>
                      cur?.includes(c.key) ? cur.filter((k) => k !== c.key) : [...(cur ?? []), c.key],
                    )
                  }
                >
                  {c.label}
                  {c.key === "truck_factor_member" && <AlertTriangle size={13} className="text-warn" aria-label="Look-ahead bias" />}
                </Check>
              ))}
            </div>
          </Section>
          <Section title="Model options">
            <div className="space-y-0.5">
              <Check checked={logCounts} onChange={() => setLogCounts((v) => !v)} title="log(1 + x) for count covariates">
                Log-transform counts
              </Check>
              <Check checked={standardize} onChange={() => setStandardize((v) => !v)} title="Hazard ratios per standard deviation">
                Standardize (HR per SD)
              </Check>
              <Check checked={stratify} onChange={() => setStratify((v) => !v)} title="Separate baseline hazard per project">
                Stratify by project
              </Check>
            </div>
          </Section>
          <Section title="Robust standard errors" hint="Sandwich variance clustered by">
            <select
              value={cluster}
              onChange={(e) => setCluster(e.target.value)}
              aria-label="Cluster robust standard errors by"
              className="w-full rounded-lg border border-line bg-page px-2.5 py-1.5 text-sm text-ink focus:border-accent focus:outline-none"
            >
              {options?.clusters.map((c) => (
                <option key={c.key} value={c.key}>
                  {c.label}
                </option>
              ))}
            </select>
          </Section>
          <Section title="PH test time scale" hint="g(t) in the Grambsch–Therneau test">
            <Segmented
              label="PH test time transform"
              value={phTransform}
              onChange={setPhTransform}
              options={(options?.ph_transforms ?? ["rank"]).map((t) => ({ id: t, label: t }))}
            />
          </Section>
          <Section title="Penalizer" hint="L2 shrinkage; use 0.1 if the model does not converge">
            <NumberField label="Penalizer" value={penalizer} onChange={setPenalizer} min={0} step={0.05} />
          </Section>
        </Card>

        <div className="min-w-0 space-y-6">
          {error && <p className="rounded-xl border border-line bg-surface px-4 py-3 text-sm text-bad">{error}</p>}
          {!result && !error && <div className="h-80 animate-pulse rounded-2xl bg-raised" />}

          {ds && (
            <div className="grid grid-cols-2 gap-4 md:grid-cols-4">
              <StatTile label="Developers at risk" value={fmt.int(ds.at_risk_subjects)} hint={`of ${fmt.int(ds.subjects)} developers`} />
              <StatTile label="Abandonments" value={fmt.int(ds.events)} hint={ds.at_risk_subjects ? `${fmt.pct(ds.events / ds.at_risk_subjects)} of developers at risk` : undefined} />
              <StatTile label="Person-months" value={fmt.int(ds.intervals)} hint="intervals at risk" />
              <StatTile
                label="Excluded"
                value={fmt.int(ds.no_risk_time + ds.late_entry)}
                hint={`${ds.no_risk_time} no time at risk · ${ds.late_entry} joined too late`}
              />
            </div>
          )}
          {loading && result && <p className="text-xs text-muted">Updating…</p>}

          {coxError && (
            <Card>
              <Empty icon={<HeartPulse size={22} />} title="Cox model not available">
                {coxError}
              </Empty>
            </Card>
          )}

          {cox?.warnings.map((w) => (
            <div key={w} className="flex gap-2.5 rounded-xl border border-line bg-surface px-4 py-3 text-sm text-ink2">
              <AlertTriangle size={16} className="mt-0.5 shrink-0 text-warn" aria-label="Warning" />
              {w}
            </div>
          ))}

          {cox && (
            <Card>
              <CardHeader
                title="Cox model — hazard ratios"
                subtitle={`HR > 1: higher monthly risk of abandoning. ${standardize ? "Continuous covariates per SD. " : ""}${cox.n_clusters ? "CIs and p-values use cluster-robust SEs. " : ""}Filled dots: p < 0.05.`}
              />
              <dl className="tabular mx-5 mt-4 grid grid-cols-2 gap-px overflow-hidden rounded-xl border border-line bg-line text-xs sm:grid-cols-5">
                {[
                      ["LR χ²", `${cox.lr_test.statistic.toFixed(1)} (df ${cox.lr_test.df})`],
                      ["p", fmtP(cox.lr_test.p)],
                      ["Global PH p", fmtP(cox.ph_global?.p)],
                      ["SEs", cox.n_clusters ? `${cox.n_clusters} clusters` : "model-based"],
                      ["Strata", cox.stratified ? "project" : "none"],
                    ].map(([k, v]) => (
                      <div key={k} className="bg-surface px-3 py-2">
                        <dt className="text-muted">{k}</dt>
                        <dd className="text-sm font-semibold text-ink">{v}</dd>
                      </div>
                    ))}
              </dl>
              <div className="px-5 pt-4">
                <ForestPlot rows={cox.coefficients} labels={labels} />
              </div>
              <div className="mt-2 overflow-x-auto">
                <table className="w-full min-w-[780px] text-sm">
                  <thead>
                    <tr className="border-y border-line bg-raised/60 text-left text-xs text-muted">
                      <th className="px-5 py-2 font-medium">Covariate</th>
                      <th className="px-3 py-2 text-right font-medium">HR</th>
                      <th className="px-3 py-2 text-right font-medium">95% CI</th>
                      <th className="px-3 py-2 text-right font-medium">coef</th>
                      <th className="px-3 py-2 text-right font-medium" title="Robust SE (model-based SE in brackets)">
                        SE {cox.n_clusters ? "robust (model)" : ""}
                      </th>
                      <th className="px-3 py-2 text-right font-medium">p</th>
                      <th className="px-3 py-2 text-right font-medium" title={`Grambsch–Therneau test, g(t) = ${cox.ph_transform}`}>
                        PH test p
                      </th>
                      <th className="px-5 py-2 font-medium">Scale</th>
                    </tr>
                  </thead>
                  <tbody className="tabular">
                    {cox.coefficients.map((c) => (
                      <tr key={c.covariate} className="border-b border-line last:border-0">
                        <td className="px-5 py-2 text-ink">{labels[c.covariate]}</td>
                        <td className="px-3 py-2 text-right font-medium text-ink">{fmtHr(c.hr)}</td>
                        <td className="px-3 py-2 text-right text-ink2">
                          {fmtHr(c.hr_lower)} – {fmtHr(c.hr_upper)}
                        </td>
                        <td className="px-3 py-2 text-right text-ink2">{fmtNum(c.coef)}</td>
                        <td className="px-3 py-2 text-right text-ink2">
                          {fmtNum(c.se)}
                          {cox.n_clusters ? <span className="text-muted"> ({fmtNum(c.naive_se)})</span> : null}
                        </td>
                        <td className={cx("px-3 py-2 text-right", c.p != null && c.p < 0.05 ? "font-medium text-ink" : "text-ink2")}>
                          {fmtP(c.p)}
                        </td>
                        <td className="px-3 py-2 text-right">
                          <span
                            className={cx(
                              "inline-flex items-center justify-end gap-1",
                              c.ph && c.ph.p < 0.05 ? "font-medium text-warn" : "text-ink2",
                            )}
                          >
                            {c.ph && c.ph.p < 0.05 && <AlertTriangle size={12} aria-label="PH assumption rejected" />}
                            {fmtP(c.ph?.p)}
                          </span>
                        </td>
                        <td className="px-5 py-2 text-xs text-muted">{c.transform || "raw"}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </Card>
          )}

          {result && result.km.groups.length > 0 && (
            <Card>
              <CardHeader
                title="Kaplan–Meier: share of developers still contributing"
                subtitle={
                  result.km.logrank_p != null ? `Log-rank test p = ${fmtP(result.km.logrank_p)}` : "Grouping measured at entry"
                }
                right={
                  <select
                    value={group}
                    onChange={(e) => setGroup(e.target.value)}
                    aria-label="Group by"
                    className="rounded-lg border border-line bg-page px-2.5 py-1.5 text-xs text-ink focus:border-accent focus:outline-none"
                  >
                    {options?.groupings.map((g) => (
                      <option key={g.key} value={g.key}>
                        {g.label}
                      </option>
                    ))}
                  </select>
                }
              />
              <div className="px-5 pt-3">
                <Legend series={result.km.groups.map((g) => ({ key: g.label, label: g.label }))} />
              </div>
              <div className="px-3 pb-2 pt-2">
                <KmChart groups={result.km.groups} />
              </div>
              <div className="grid gap-px border-t border-line bg-line sm:grid-cols-2">
                {result.km.groups.map((g, i) => (
                  <div key={g.label} className="flex items-center gap-3 bg-surface px-5 py-3 text-xs">
                    <span className="h-2.5 w-2.5 rounded-full" style={{ background: SERIES[i] }} />
                    <span className="font-medium text-ink">{g.label}</span>
                    <span className="tabular ml-auto text-ink2">
                      n {g.n} · events {g.events} · median {g.median == null ? "not reached" : `${g.median} mo`}
                    </span>
                  </div>
                ))}
              </div>
            </Card>
          )}

          <Card className="px-5 py-4">
            <div className="flex gap-3 text-xs leading-relaxed text-ink2">
              <Info size={16} className="mt-0.5 shrink-0 text-accent" />
              <div className="space-y-1.5">
                <p>
                  <span className="font-medium text-ink">Design.</span> One subject per developer × project, followed
                  month by month from entry. Abandonment = last active month followed by at least {abandon} inactive
                  months; events are only confirmed up to {abandon} months before each project's last observed month,
                  then everyone still at risk is censored.
                </p>
                <p>
                  Covariates for month t summarize months t−{window}…t−1, so the activity that marks the exit month
                  never predicts itself. Developers who leave in their entry month (one-off contributors) have no time at
                  risk and are excluded.
                </p>
              </div>
            </div>
          </Card>
        </div>
      </div>
    </div>
  );
}
