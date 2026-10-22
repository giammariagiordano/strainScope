import type { ProjectSummary } from "../api";
import { cx } from "./ui";

/** Checkbox list of projects; an empty selection means "all projects". */
export default function ProjectPicker({
  projects,
  selected,
  onChange,
}: {
  projects: ProjectSummary[];
  selected: number[];
  onChange: (ids: number[]) => void;
}) {
  const all = selected.length === 0;
  const toggle = (id: number) =>
    onChange(selected.includes(id) ? selected.filter((x) => x !== id) : [...selected, id]);
  return (
    <div className="space-y-1">
      <button
        onClick={() => onChange([])}
        className={cx(
          "flex w-full items-center justify-between rounded-lg px-2.5 py-1.5 text-left text-sm transition",
          all ? "bg-accent-soft font-medium text-accent" : "text-ink2 hover:bg-raised",
        )}
      >
        All projects <span className="tabular text-xs">{projects.length}</span>
      </button>
      {projects.map((p) => (
        <label
          key={p.id}
          className="flex cursor-pointer items-center gap-2.5 rounded-lg px-2.5 py-1.5 text-sm text-ink2 hover:bg-raised"
        >
          <input
            type="checkbox"
            checked={selected.includes(p.id)}
            onChange={() => toggle(p.id)}
            className="accent-[var(--accent)]"
          />
          <span className="min-w-0 flex-1 truncate">{p.full_name}</span>
          {p.status !== "done" && <span className="text-[11px] text-muted">{p.status}</span>}
        </label>
      ))}
    </div>
  );
}
