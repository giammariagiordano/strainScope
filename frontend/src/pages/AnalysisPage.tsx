import { AlertTriangle, Download, Info, Sigma } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { CartesianGrid, ResponsiveContainer, Scatter, ScatterChart, Tooltip, XAxis, YAxis } from "recharts";
import {
  api,
  usePoll,
  type CorrelationResult,
  type Level,
  type Method,
  type ScatterResult,
  type VariableDef,
} from "../api";
import ProjectPicker from "../components/ProjectPicker";
import { Section, Segmented } from "../components/controls";
import { Card, CardHeader, Empty, cx, fmt } from "../components/ui";

const DEFAULTS: Record<Level, string[]> = {
  developer: [
    "core_share", "truck_factor_member", "mean_centrality", "off_hours_share",
    "commits_per_month", "comments", "reviews", "tenure_months", "abandoned",
  ],
  developer_month: ["is_core", "centrality", "off_hours_share", "commits", "comments", "reviews", "exit"],
}; // prettier-ignore

const fmtP = (p: number | null | undefined) =>
  p == null ? "—" : p < 0.001 ? "<0.001" : p.toFixed(3);

function cellColor(r: number | null) {
  if (r == null) return "var(--surface-2)";
  const pole = r >= 0 ? "var(--div-pos)" : "var(--div-neg)";
  return `color-mix(in oklab, ${pole} ${Math.round(Math.min(1, Math.abs(r)) * 100)}%, var(--div-mid))`;
}

function VariableList({
  vars,
  selected,
  disabled,
  onToggle,
}: {
  vars: VariableDef[];
  selected: string[];
  disabled?: string[];
  onToggle: (key: string) => void;
}) {
  return (
    <div className="space-y-0.5">
      {vars.map((v) => (
        <label
          key={v.key}
          title={v.description}
          className={cx(
            "flex cursor-pointer items-center gap-2.5 rounded-lg px-2.5 py-1.5 text-sm text-ink2 hover:bg-raised",
            disabled?.includes(v.key) && "pointer-events-none opacity-40",
          )}
        >
          <input
            type="checkbox"
            checked={selected.includes(v.key)}
            onChange={() => onToggle(v.key)}
            className="accent-[var(--accent)]"
          />
          {v.label}
        </label>
      ))}
    </div>
  );
}

function Matrix({
  result,
  labels,
  selected,
  onSelect,
}: {
  result: CorrelationResult;
  labels: Record<string, string>;
  selected: [string, string] | null;
  onSelect: (pair: [string, string]) => void;
}) {
  const vars = result.variables;
  return (
    <div className="overflow-x-auto">
      <table className="mx-auto border-separate border-spacing-[2px]">
        <tbody>
          {vars.map((row, i) => i > 0 && (
            <tr key={row}>
              <th scope="row" className="whitespace-nowrap pr-2 text-right text-xs font-medium text-ink2">
                {labels[row]}
              </th>
              {vars.slice(0, i).map((col, j) => {
                const c = result.matrix[i][j];
                const strong = c.r != null && Math.abs(c.r) >= 0.5;
                const isSel = selected && ((selected[0] === row && selected[1] === col) || (selected[0] === col && selected[1] === row));
                return (
                  <td key={col} className="p-0">
                    <button
                      onClick={() => onSelect([col, row])}
                      title={`${labels[row]} × ${labels[col]}\nr = ${c.r?.toFixed(3) ?? "n/a"}\np = ${fmtP(c.p)}  q = ${fmtP(c.q)}\nn = ${c.n}`}
                      className={cx(
                        "tabular grid h-12 w-16 place-items-center rounded-md text-xs font-medium transition hover:ring-2 hover:ring-accent",
                        isSel && "ring-2 ring-ink",
                        strong ? "text-white" : "text-ink",
                      )}
                      style={{ background: cellColor(c.r) }}
                    >
                      <span>
                        {c.r == null ? "n/a" : c.r.toFixed(2)}
                        {c.q != null && c.q < 0.05 && <sup className="ml-px">*</sup>}
                      </span>
                    </button>
                  </td>
                );
              })}
            </tr>
          ))}
          <tr>
            <th />
            {vars.slice(0, -1).map((col) => (
              <th key={col} className="h-28 w-16 align-top">
                <div className="mx-auto mt-1 w-4 whitespace-nowrap text-left text-xs font-medium text-ink2 [writing-mode:vertical-rl] rotate-180">
                  {labels[col]}
                </div>
              </th>
            ))}
          </tr>
        </tbody>
      </table>
      <div className="mt-2 flex flex-wrap items-center justify-center gap-x-6 gap-y-2 text-[11px] text-muted">
        <span className="flex items-center gap-1.5">
          −1
          <span
            className="h-2.5 w-40 rounded-full"
            style={{ background: "linear-gradient(to right, var(--div-neg), var(--div-mid), var(--div-pos))" }}
          />
          +1
        </span>
        <span>* significant after Benjamini–Hochberg correction (q &lt; 0.05)</span>
      </div>
    </div>
  );
}

function PairScatter({
  data,
  xLabel,
  yLabel,
}: {
  data: ScatterResult;
  xLabel: string;
  yLabel: string;
}) {
  return (
    <ResponsiveContainer width="100%" height={300}>
      <ScatterChart margin={{ top: 8, right: 16, bottom: 24, left: 8 }}>
        <CartesianGrid />
        <XAxis
          type="number"
          dataKey="x"
          name={xLabel}
          tickLine={false}
          axisLine={false}
          fontSize={11}
          label={{ value: xLabel, position: "insideBottom", offset: -14, fontSize: 11, fill: "var(--text-secondary)" }}
        />
        <YAxis
          type="number"
          dataKey="y"
          name={yLabel}
          tickLine={false}
          axisLine={false}
          fontSize={11}
          width={52}
          label={{ value: yLabel, angle: -90, position: "insideLeft", fontSize: 11, fill: "var(--text-secondary)" }}
        />
        <Tooltip
          cursor={{ strokeDasharray: "3 3", stroke: "var(--axis)" }}
          content={({ active, payload }) => {
            if (!active || !payload?.length) return null;
            const pt = payload[0].payload as ScatterResult["points"][number];
            return (
              <div className="rounded-lg border border-line bg-surface px-3 py-2 text-xs shadow-lg">
                <div className="mb-1 font-medium text-ink">{pt.label}</div>
                <div className="tabular text-ink2">
                  {xLabel}: <span className="text-ink">{+pt.x.toFixed(3)}</span>
                </div>
                <div className="tabular text-ink2">
                  {yLabel}: <span className="text-ink">{+pt.y.toFixed(3)}</span>
                </div>
              </div>
            );
          }}
        />
        <Scatter data={data.points} fill="var(--series-1)" fillOpacity={0.55} isAnimationActive={false} />
      </ScatterChart>
    </ResponsiveContainer>
  );
}

function downloadCsv(result: CorrelationResult, labels: Record<string, string>) {
  const header = ["variable_x", "variable_y", "r", "p", "q_bh", "n", "method", "controls", "level"];
  const rows = result.pairs.map((p) =>
    [labels[p.x], labels[p.y], p.r, p.p, p.q, p.n, result.method, result.controls.join(";"), result.level].join(","),
  );
  const blob = new Blob([[header.join(","), ...rows].join("\n")], { type: "text/csv" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = `correlations_${result.level}_${result.method}.csv`;
  a.click();
  URL.revokeObjectURL(a.href);
}

export default function AnalysisPage() {
  const { data: projects } = usePoll(api.projects, 0);
  const { data: varDefs } = usePoll(api.variables, 0);
  const [projectIds, setProjectIds] = useState<number[]>([]);
  const [level, setLevel] = useState<Level>("developer");
  const [method, setMethod] = useState<Method>("spearman");
  const [minMonths, setMinMonths] = useState(3);
  const [abandonMonths, setAbandonMonths] = useState<number | null>(null);
  const [variables, setVariables] = useState<string[]>(DEFAULTS.developer);
  const [controls, setControls] = useState<string[]>([]);
  const [result, setResult] = useState<CorrelationResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [pair, setPair] = useState<[string, string] | null>(null);
  const [scatter, setScatter] = useState<ScatterResult | null>(null);

  const vars = varDefs?.levels[level] ?? [];
  const labels = useMemo(() => Object.fromEntries(vars.map((v) => [v.key, v.label])), [vars]);
  const abandon = abandonMonths ?? varDefs?.defaults.abandon_months ?? 12;
  const base = {
    project_ids: projectIds.length ? projectIds : null,
    level,
    min_active_months: minMonths,
    abandon_months: abandon,
  };

  function changeLevel(l: Level) {
    setLevel(l);
    setVariables(DEFAULTS[l]);
    setControls([]);
    setPair(null);
  }

  // Re-run the analysis whenever the query changes (debounced).
  useEffect(() => {
    if (!varDefs) return;
    const t = setTimeout(async () => {
      setLoading(true);
      try {
        const r = await api.correlations({ ...base, method, variables, controls });
        setResult(r);
        setError(null);
        if (!pair || !r.variables.includes(pair[0]) || !r.variables.includes(pair[1]))
          setPair(r.pairs.length ? [r.pairs[0].x, r.pairs[0].y] : null);
      } catch (e) {
        setError((e as Error).message);
      } finally {
        setLoading(false);
      }
    }, 350);
    return () => clearTimeout(t);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [varDefs, projectIds, level, method, minMonths, abandon, variables, controls]);

  useEffect(() => {
    if (!pair) return setScatter(null);
    let cancelled = false;
    api.scatter({ ...base, x: pair[0], y: pair[1] }).then((s) => !cancelled && setScatter(s));
    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [pair, projectIds, level, minMonths, abandon]);

  const pairStats = result && pair ? result.pairs.find((p) => (p.x === pair[0] && p.y === pair[1]) || (p.x === pair[1] && p.y === pair[0])) : null;
  const toggle = (list: string[], set: (v: string[]) => void, key: string) =>
    set(list.includes(key) ? list.filter((k) => k !== key) : [...list, key]);

  return (
    <div className="space-y-6">
      <header>
        <h1 className="text-2xl font-semibold tracking-tight text-ink">Correlation analysis</h1>
        <p className="mt-1 text-sm text-ink2">
          Explore associations between structural role, workload and activity. Bots are always excluded.
        </p>
      </header>

      <div className="grid gap-6 lg:grid-cols-[300px_minmax(0,1fr)]">
        <Card className="h-fit space-y-5 p-5 lg:sticky lg:top-6">
          <Section title="Projects">
            {projects && <ProjectPicker projects={projects} selected={projectIds} onChange={setProjectIds} />}
          </Section>
          <Section title="Unit of analysis">
            <Segmented
              label="Unit of analysis"
              value={level}
              onChange={changeLevel}
              options={[
                { id: "developer", label: "Developer" },
                { id: "developer_month", label: "Developer-month" },
              ]}
            />
          </Section>
          <Section title="Method">
            <Segmented
              label="Method"
              value={method}
              onChange={setMethod}
              options={[
                { id: "spearman", label: "Spearman" },
                { id: "pearson", label: "Pearson" },
              ]}
            />
          </Section>
          <Section title="Minimum active months" hint="Excludes occasional contributors">
            <input
              type="number"
              min={1}
              value={minMonths}
              onChange={(e) => setMinMonths(Math.max(1, Number(e.target.value) || 1))}
              className="tabular w-full rounded-lg border border-line bg-page px-3 py-1.5 text-sm text-ink focus:border-accent focus:outline-none"
            />
          </Section>
          <Section
            title="Abandonment threshold"
            hint="Months of inactivity, before the project's last observed month, that count as abandonment"
          >
            <input
              type="number"
              min={1}
              value={abandon}
              onChange={(e) => setAbandonMonths(Math.max(1, Number(e.target.value) || 1))}
              className="tabular w-full rounded-lg border border-line bg-page px-3 py-1.5 text-sm text-ink focus:border-accent focus:outline-none"
            />
          </Section>
          <Section title="Variables">
            <VariableList vars={vars} selected={variables} disabled={controls} onToggle={(k) => toggle(variables, setVariables, k)} />
          </Section>
          <Section title="Control for" hint="Partial correlation: removes the linear effect of these variables">
            <VariableList vars={vars} selected={controls} onToggle={(k) => toggle(controls, setControls, k)} />
          </Section>
        </Card>

        <div className="min-w-0 space-y-6">
          {error && <p className="rounded-xl border border-line bg-surface px-4 py-3 text-sm text-bad">{error}</p>}

          {result && (
            <div className="flex flex-wrap items-center gap-x-6 gap-y-1 text-sm text-ink2">
              <span>
                <span className="tabular font-semibold text-ink">{fmt.int(result.rows)}</span> rows
              </span>
              <span>
                <span className="tabular font-semibold text-ink">{fmt.int(result.developers)}</span> developers
              </span>
              <span>
                <span className="tabular font-semibold text-ink">{result.projects}</span> projects
              </span>
              <span className="capitalize">{result.method}</span>
              {result.controls.length > 0 && (
                <span>controlling for {result.controls.map((c) => labels[c]).join(", ")}</span>
              )}
              {loading && <span className="text-muted">Updating…</span>}
            </div>
          )}

          {result?.warnings.map((w) => (
            <div key={w} className="flex gap-2.5 rounded-xl border border-line bg-surface px-4 py-3 text-sm text-ink2">
              <AlertTriangle size={16} className="mt-0.5 shrink-0 text-warn" aria-label="Warning" />
              {w}
            </div>
          ))}

          {!result && !error && <div className="h-80 animate-pulse rounded-2xl bg-raised" />}

          {result && result.rows === 0 && (
            <Card>
              <Empty icon={<Sigma size={22} />} title="No data for this selection">
                Mine a project to completion or lower the minimum active months.
              </Empty>
            </Card>
          )}

          {result && result.rows > 0 && (
            <>
              <Card>
                <CardHeader
                  title="Correlation matrix"
                  subtitle="Click a cell to inspect the pair. Pairwise deletion of missing values."
                />
                <div className="px-5 pb-5 pt-4">
                  <Matrix result={result} labels={labels} selected={pair} onSelect={setPair} />
                </div>
              </Card>

              {pair && (
                <Card>
                  <CardHeader
                    title={`${labels[pair[0]]} × ${labels[pair[1]]}`}
                    subtitle={
                      scatter
                        ? `${fmt.int(scatter.total)} observations${scatter.sampled ? `, ${scatter.points.length} shown (random sample)` : ""} · raw values`
                        : undefined
                    }
                    right={
                      pairStats && (
                        <div className="tabular flex gap-4 text-right text-xs">
                          {[
                            ["r", pairStats.r.toFixed(3)],
                            ["p", fmtP(pairStats.p)],
                            ["q (BH)", fmtP(pairStats.q)],
                            ["n", fmt.int(pairStats.n)],
                          ].map(([k, v]) => (
                            <div key={k}>
                              <div className="text-muted">{k}</div>
                              <div className="text-sm font-semibold text-ink">{v}</div>
                            </div>
                          ))}
                        </div>
                      )
                    }
                  />
                  <div className="px-3 pb-4 pt-2">
                    {scatter ? (
                      <PairScatter data={scatter} xLabel={labels[pair[0]]} yLabel={labels[pair[1]]} />
                    ) : (
                      <div className="h-[300px] animate-pulse rounded-xl bg-raised" />
                    )}
                    {result.controls.length > 0 && (
                      <p className="flex items-center gap-1.5 px-2 text-xs text-muted">
                        <Info size={13} /> The scatter shows raw values; the statistics above are partial correlations.
                      </p>
                    )}
                  </div>
                </Card>
              )}

              <Card className="overflow-hidden">
                <CardHeader
                  title="All pairs"
                  subtitle="Sorted by absolute correlation"
                  right={
                    <button
                      onClick={() => downloadCsv(result, labels)}
                      className="inline-flex items-center gap-1.5 rounded-lg border border-line bg-surface px-3 py-1.5 text-xs font-medium text-ink hover:bg-raised"
                    >
                      <Download size={14} /> CSV
                    </button>
                  }
                />
                <div className="mt-4 overflow-x-auto">
                  <table className="w-full min-w-[560px] text-sm">
                    <thead>
                      <tr className="border-y border-line bg-raised/60 text-left text-xs text-muted">
                        <th className="px-5 py-2 font-medium">Pair</th>
                        <th className="px-3 py-2 text-right font-medium">r</th>
                        <th className="px-3 py-2 text-right font-medium">p</th>
                        <th className="px-3 py-2 text-right font-medium">q (BH)</th>
                        <th className="px-5 py-2 text-right font-medium">n</th>
                      </tr>
                    </thead>
                    <tbody className="tabular">
                      {result.pairs.map((p) => (
                        <tr
                          key={`${p.x}-${p.y}`}
                          onClick={() => setPair([p.x, p.y])}
                          className="cursor-pointer border-b border-line last:border-0 hover:bg-raised"
                        >
                          <td className="px-5 py-2 text-ink">
                            {labels[p.x]} <span className="text-muted">×</span> {labels[p.y]}
                          </td>
                          <td className="px-3 py-2 text-right">
                            <span className="inline-flex items-center justify-end gap-2">
                              <span className="h-2.5 w-2.5 rounded-sm" style={{ background: cellColor(p.r) }} />
                              <span className="font-medium text-ink">{p.r.toFixed(3)}</span>
                            </span>
                          </td>
                          <td className="px-3 py-2 text-right text-ink2">{fmtP(p.p)}</td>
                          <td className={cx("px-3 py-2 text-right", p.q < 0.05 ? "font-medium text-ink" : "text-ink2")}>
                            {fmtP(p.q)}
                          </td>
                          <td className="px-5 py-2 text-right text-ink2">{fmt.int(p.n)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </Card>
            </>
          )}
        </div>
      </div>
    </div>
  );
}
