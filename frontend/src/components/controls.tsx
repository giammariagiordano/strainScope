import type { ReactNode } from "react";
import { cx } from "./ui";

export function Segmented<T extends string>({
  value,
  options,
  onChange,
  label,
}: {
  value: T;
  options: { id: T; label: string }[];
  onChange: (v: T) => void;
  label: string;
}) {
  return (
    <div className="inline-flex w-full rounded-lg border border-line bg-page p-0.5" role="group" aria-label={label}>
      {options.map((o) => (
        <button
          key={o.id}
          onClick={() => onChange(o.id)}
          aria-pressed={value === o.id}
          className={cx(
            "flex-1 rounded-md px-2.5 py-1.5 text-xs font-medium transition",
            value === o.id ? "bg-surface text-accent shadow-sm" : "text-ink2 hover:text-ink",
          )}
        >
          {o.label}
        </button>
      ))}
    </div>
  );
}

export function Section({ title, children, hint }: { title: string; children: ReactNode; hint?: string }) {
  return (
    <div className="space-y-2">
      <div>
        <div className="text-xs font-semibold uppercase tracking-wide text-muted">{title}</div>
        {hint && <div className="text-[11px] text-muted">{hint}</div>}
      </div>
      {children}
    </div>
  );
}

export function NumberField({
  value,
  onChange,
  min = 1,
  step = 1,
  label,
}: {
  value: number;
  onChange: (v: number) => void;
  min?: number;
  step?: number;
  label: string;
}) {
  return (
    <input
      type="number"
      aria-label={label}
      min={min}
      step={step}
      value={value}
      onChange={(e) => onChange(Math.max(min, Number(e.target.value) || min))}
      className="tabular w-full rounded-lg border border-line bg-page px-3 py-1.5 text-sm text-ink focus:border-accent focus:outline-none"
    />
  );
}

export function Check({
  checked,
  onChange,
  children,
  title,
  disabled,
}: {
  checked: boolean;
  onChange: () => void;
  children: ReactNode;
  title?: string;
  disabled?: boolean;
}) {
  return (
    <label
      title={title}
      className={cx(
        "flex cursor-pointer items-center gap-2.5 rounded-lg px-2.5 py-1.5 text-sm text-ink2 hover:bg-raised",
        disabled && "pointer-events-none opacity-40",
      )}
    >
      <input type="checkbox" checked={checked} onChange={onChange} className="accent-[var(--accent)]" />
      {children}
    </label>
  );
}
