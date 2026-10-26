import {
  Bar,
  BarChart,
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
  type TooltipProps,
} from "recharts";
import { fmt } from "./ui";

export const SERIES = ["var(--series-1)", "var(--series-2)", "var(--series-3)", "var(--series-4)", "var(--series-5)"];

export interface SeriesDef {
  key: string;
  label: string;
  format?: (v: number) => string;
}

function ChartTooltip({ active, payload, label, series }: TooltipProps<number, string> & { series: SeriesDef[] }) {
  if (!active || !payload?.length) return null;
  return (
    <div className="min-w-[150px] rounded-lg border border-line bg-surface px-3 py-2 text-xs shadow-lg">
      <div className="mb-1 font-medium text-ink">{typeof label === "string" ? fmt.month(label) : label}</div>
      {series.map((s, i) => {
        const p = payload.find((x) => x.dataKey === s.key);
        if (!p || p.value == null) return null;
        return (
          <div key={s.key} className="flex items-center justify-between gap-4 py-0.5">
            <span className="flex items-center gap-1.5 text-ink2">
              <span className="h-2 w-2 rounded-full" style={{ background: SERIES[i] }} />
              {s.label}
            </span>
            <span className="tabular font-medium text-ink">{s.format ? s.format(p.value) : fmt.int(p.value)}</span>
          </div>
        );
      })}
    </div>
  );
}

export function Legend({ series }: { series: SeriesDef[] }) {
  if (series.length < 2) return null;
  return (
    <div className="flex flex-wrap gap-x-4 gap-y-1 text-xs text-ink2">
      {series.map((s, i) => (
        <span key={s.key} className="flex items-center gap-1.5">
          <span className="h-0.5 w-3.5 rounded-full" style={{ background: SERIES[i] }} />
          {s.label}
        </span>
      ))}
    </div>
  );
}

const axisProps = {
  tickLine: false,
  axisLine: false,
  fontSize: 11,
} as const;

function xTicks(data: { month: string }[]) {
  // Label roughly every year for long ranges, every quarter for short ones.
  const step = data.length > 36 ? 12 : data.length > 12 ? 3 : 1;
  return data.filter((_, i) => i % step === 0).map((d) => d.month);
}

export function MonthlyBars({
  data,
  series,
  height = 220,
}: {
  data: { month: string }[];
  series: SeriesDef;
  height?: number;
}) {
  return (
    <ResponsiveContainer width="100%" height={height}>
      <BarChart data={data} margin={{ top: 8, right: 8, bottom: 0, left: -12 }} barCategoryGap={1}>
        <CartesianGrid vertical={false} strokeDasharray="0" />
        <XAxis dataKey="month" ticks={xTicks(data)} tickFormatter={fmt.month} {...axisProps} />
        <YAxis allowDecimals={false} {...axisProps} width={44} />
        <Tooltip content={<ChartTooltip series={[series]} />} cursor={{ fill: "var(--accent-soft)" }} />
        <Bar dataKey={series.key} fill={SERIES[0]} radius={[4, 4, 0, 0]} maxBarSize={18} isAnimationActive={false} />
      </BarChart>
    </ResponsiveContainer>
  );
}

export function MonthlyLines({
  data,
  series,
  height = 220,
  percent,
  unitInterval,
}: {
  data: { month: string }[];
  series: SeriesDef[];
  height?: number;
  /** Values are shares in [0, 1], shown as percentages. */
  percent?: boolean;
  /** Values are scores in [0, 1], shown as decimals. */
  unitInterval?: boolean;
}) {
  return (
    <ResponsiveContainer width="100%" height={height}>
      <LineChart data={data} margin={{ top: 8, right: 8, bottom: 0, left: -12 }}>
        <CartesianGrid vertical={false} />
        <XAxis dataKey="month" ticks={xTicks(data)} tickFormatter={fmt.month} {...axisProps} />
        <YAxis
          allowDecimals={percent || unitInterval}
          domain={percent || unitInterval ? [0, 1] : [0, "auto"]}
          tickFormatter={percent ? (v: number) => `${Math.round(v * 100)}%` : undefined}
          {...axisProps}
          width={44}
        />
        <Tooltip
          content={<ChartTooltip series={series} />}
          cursor={{ stroke: "var(--axis)", strokeWidth: 1 }}
        />
        {series.map((s, i) => (
          <Line
            key={s.key}
            type="monotone"
            dataKey={s.key}
            stroke={SERIES[i]}
            strokeWidth={2}
            dot={false}
            activeDot={{ r: 4, strokeWidth: 2, stroke: "var(--surface-1)" }}
            connectNulls={false}
            isAnimationActive={false}
          />
        ))}
      </LineChart>
    </ResponsiveContainer>
  );
}

const DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"];

export function HourHeatmap({ data, workHours }: { data: number[][]; workHours: [number, number] }) {
  const max = Math.max(1, ...data.flat());
  const step = (v: number) => (v === 0 ? 0 : Math.min(7, 1 + Math.floor((v / max) * 7)));
  return (
    <div className="overflow-x-auto">
      <div className="min-w-[560px]">
        <div className="grid grid-cols-[36px_repeat(24,minmax(0,1fr))] gap-[2px]">
          <div />
          {Array.from({ length: 24 }, (_, h) => (
            <div key={h} className="tabular pb-1 text-center text-[10px] text-muted">
              {h % 3 === 0 ? h : ""}
            </div>
          ))}
          {data.map((row, d) => (
            <div key={d} className="contents">
              <div className="pr-2 text-right text-[11px] leading-6 text-muted">{DAYS[d]}</div>
              {row.map((v, h) => {
                const work = d < 5 && h >= workHours[0] && h < workHours[1];
                return (
                  <div
                    key={h}
                    title={`${DAYS[d]} ${String(h).padStart(2, "0")}:00 — ${v} commit${v === 1 ? "" : "s"}${work ? "" : " (off-hours)"}`}
                    className="h-6 rounded-[4px] transition hover:ring-2 hover:ring-accent"
                    style={{ background: `var(--seq-${step(v)})` }}
                  />
                );
              })}
            </div>
          ))}
        </div>
        <div className="mt-3 flex items-center justify-between text-[11px] text-muted">
          <span>
            Author's local time · working hours {workHours[0]}:00–{workHours[1]}:00, Mon–Fri
          </span>
          <span className="flex items-center gap-1.5">
            Fewer
            {[0, 1, 3, 5, 7].map((s) => (
              <span key={s} className="h-3 w-3 rounded-[3px]" style={{ background: `var(--seq-${s})` }} />
            ))}
            More
          </span>
        </div>
      </div>
    </div>
  );
}
