import { AlertTriangle, CheckCircle2, Circle, Clock, Loader2, PauseCircle, Trash2 } from "lucide-react";
import type { ButtonHTMLAttributes, ReactNode } from "react";
import type { ProjectStatus, StageStatus } from "../api";

export function cx(...classes: (string | false | null | undefined)[]) {
  return classes.filter(Boolean).join(" ");
}

const STATUS: Record<ProjectStatus | StageStatus, { label: string; icon: typeof Circle; cls: string }> = {
  queued: { label: "Queued", icon: Clock, cls: "text-ink2 bg-raised" },
  pending: { label: "Pending", icon: Circle, cls: "text-muted bg-raised" },
  running: { label: "Running", icon: Loader2, cls: "text-accent bg-accent-soft" },
  paused: { label: "Paused", icon: PauseCircle, cls: "text-warn bg-raised" },
  done: { label: "Done", icon: CheckCircle2, cls: "text-good bg-raised" },
  failed: { label: "Failed", icon: AlertTriangle, cls: "text-bad bg-raised" },
  deleting: { label: "Deleting…", icon: Trash2, cls: "text-bad bg-raised" },
};

export function StatusBadge({ status, pausing }: { status: ProjectStatus | StageStatus; pausing?: boolean }) {
  const s = STATUS[status];
  const Icon = s.icon;
  return (
    <span className={cx("inline-flex items-center gap-1.5 rounded-full px-2.5 py-1 text-xs font-medium", s.cls)}>
      <Icon size={13} className={status === "running" ? "animate-spin" : undefined} aria-hidden />
      {pausing ? "Pausing…" : s.label}
    </span>
  );
}

export function StageIcon({ status }: { status: StageStatus }) {
  const s = STATUS[status];
  const Icon = s.icon;
  return (
    <span className={cx("grid h-8 w-8 shrink-0 place-items-center rounded-full", s.cls)} title={s.label}>
      <Icon size={16} className={status === "running" ? "animate-spin" : undefined} aria-label={s.label} />
    </span>
  );
}

export function ProgressBar({ value, max, active }: { value: number; max: number | null; active?: boolean }) {
  const pct = max ? Math.min(100, (value / max) * 100) : null;
  return (
    <div className="h-1.5 w-full overflow-hidden rounded-full bg-raised">
      {pct !== null ? (
        <div className="h-full rounded-full bg-accent transition-[width] duration-500" style={{ width: `${pct}%` }} />
      ) : active ? (
        <div className="h-full w-1/3 animate-[indeterminate_1.4s_ease-in-out_infinite] rounded-full bg-accent" />
      ) : null}
    </div>
  );
}

export function Card({ children, className }: { children: ReactNode; className?: string }) {
  return <div className={cx("rounded-2xl border border-line bg-surface shadow-card", className)}>{children}</div>;
}

export function CardHeader({ title, subtitle, right }: { title: string; subtitle?: string; right?: ReactNode }) {
  return (
    <div className="flex items-start justify-between gap-4 px-5 pt-5">
      <div>
        <h3 className="text-sm font-semibold text-ink">{title}</h3>
        {subtitle && <p className="mt-0.5 text-xs text-muted">{subtitle}</p>}
      </div>
      {right}
    </div>
  );
}

export function StatTile({ label, value, hint }: { label: string; value: ReactNode; hint?: string }) {
  return (
    <Card className="px-5 py-4">
      <div className="text-xs font-medium uppercase tracking-wide text-muted">{label}</div>
      <div className="mt-1.5 text-2xl font-semibold text-ink">{value}</div>
      {hint && <div className="mt-0.5 text-xs text-muted">{hint}</div>}
    </Card>
  );
}

type Variant = "primary" | "secondary" | "ghost" | "danger";

export function Button({
  variant = "secondary",
  className,
  children,
  ...props
}: ButtonHTMLAttributes<HTMLButtonElement> & { variant?: Variant }) {
  const styles: Record<Variant, string> = {
    primary: "bg-accent text-white hover:opacity-90",
    secondary: "border border-line bg-surface text-ink hover:bg-raised",
    ghost: "text-ink2 hover:bg-raised hover:text-ink",
    danger: "border border-line bg-surface text-bad hover:bg-raised",
  };
  return (
    <button
      className={cx(
        "inline-flex items-center justify-center gap-1.5 rounded-lg px-3 py-2 text-sm font-medium transition disabled:cursor-not-allowed disabled:opacity-50",
        styles[variant],
        className,
      )}
      {...props}
    >
      {children}
    </button>
  );
}

export function Empty({ icon, title, children }: { icon: ReactNode; title: string; children?: ReactNode }) {
  return (
    <div className="flex flex-col items-center justify-center px-6 py-16 text-center">
      <div className="mb-3 grid h-12 w-12 place-items-center rounded-2xl bg-accent-soft text-accent">{icon}</div>
      <h3 className="text-sm font-semibold text-ink">{title}</h3>
      {children && <div className="mt-1 max-w-sm text-sm text-muted">{children}</div>}
    </div>
  );
}

export const fmt = {
  int: (n: number | null | undefined) => (n == null ? "—" : n.toLocaleString("en-US")),
  pct: (n: number | null | undefined) => (n == null ? "—" : `${Math.round(n * 100)}%`),
  date: (s: string | null | undefined) =>
    s ? new Date(s).toLocaleDateString("en-GB", { year: "numeric", month: "short", day: "numeric" }) : "—",
  month: (m: string) => {
    const [y, mo] = m.split("-");
    return new Date(Number(y), Number(mo) - 1).toLocaleDateString("en-GB", { month: "short", year: "2-digit" });
  },
  ago: (s: string) => {
    const sec = Math.max(0, (Date.now() - new Date(s).getTime()) / 1000);
    if (sec < 60) return `${Math.round(sec)}s ago`;
    if (sec < 3600) return `${Math.round(sec / 60)}m ago`;
    if (sec < 86400) return `${Math.round(sec / 3600)}h ago`;
    return `${Math.round(sec / 86400)}d ago`;
  },
};
