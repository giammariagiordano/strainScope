import { Activity, Download, FolderGit2, HeartPulse, ShieldCheck, Sigma } from "lucide-react";
import { NavLink, Outlet } from "react-router-dom";
import { api, usePoll } from "../api";
import { cx } from "./ui";

function Logo() {
  return (
    <div className="flex items-center gap-2.5 px-2">
      <div className="grid h-9 w-9 place-items-center rounded-xl bg-accent text-white">
        <Activity size={18} strokeWidth={2.5} />
      </div>
      <div>
        <div className="text-[15px] font-semibold leading-tight text-ink">StrainScope</div>
        <div className="text-[11px] leading-tight text-muted">Developer strain research</div>
      </div>
    </div>
  );
}

function SystemStatus() {
  const { data } = usePoll(api.status, 5000);
  const gh = data?.github;
  const pct = gh ? gh.remaining / Math.max(1, gh.limit) : null;
  const waiting = gh?.waiting_until && gh.waiting_until * 1000 > Date.now();
  return (
    <div className="space-y-3 rounded-xl border border-line bg-surface p-3 text-xs">
      <div className="flex items-center justify-between">
        <span className="text-muted">Worker</span>
        <span className={cx("inline-flex items-center gap-1.5 font-medium", data?.worker_alive ? "text-good" : "text-bad")}>
          <span className={cx("h-2 w-2 rounded-full", data?.worker_alive ? "bg-good" : "bg-bad")} />
          {data ? (data.worker_alive ? "Online" : "Offline") : "…"}
        </span>
      </div>
      <div>
        <div className="mb-1.5 flex items-center justify-between">
          <span className="text-muted">GitHub quota</span>
          <span className="tabular font-medium text-ink2">
            {gh ? `${gh.remaining.toLocaleString()} / ${gh.limit.toLocaleString()}` : "not used yet"}
          </span>
        </div>
        <div className="h-1.5 overflow-hidden rounded-full bg-raised">
          <div
            className={cx("h-full rounded-full", pct !== null && pct < 0.1 ? "bg-bad" : "bg-accent")}
            style={{ width: `${(pct ?? 0) * 100}%` }}
          />
        </div>
        {waiting && (
          <div className="mt-1.5 text-warn">
            Rate limited — resumes at {new Date(gh!.waiting_until! * 1000).toLocaleTimeString()}
          </div>
        )}
      </div>
      {data && !data.worker_alive && (
        <p className="leading-snug text-muted">
          Start it with <code className="rounded bg-raised px-1 py-0.5 text-ink2">python -m app.worker</code>
        </p>
      )}
    </div>
  );
}

export default function Layout() {
  return (
    <div className="min-h-screen md:flex">
      <aside className="border-b border-line bg-page md:sticky md:top-0 md:flex md:h-screen md:w-64 md:shrink-0 md:flex-col md:border-b-0 md:border-r">
        <div className="flex items-center justify-between gap-4 p-4 md:block md:space-y-6 md:p-5">
          <Logo />
          <nav className="flex gap-1 md:flex-col">
            {[
              { to: "/", label: "Projects", icon: FolderGit2 },
              { to: "/analysis", label: "Correlations", icon: Sigma },
              { to: "/survival", label: "Survival", icon: HeartPulse },
              { to: "/export", label: "Export", icon: Download },
            ].map(({ to, label, icon: Icon }) => (
              <NavLink
                key={to}
                to={to}
                end={to === "/"}
                className={({ isActive }) =>
                  cx(
                    "flex items-center gap-2.5 rounded-lg px-3 py-2 text-sm font-medium transition",
                    isActive ? "bg-accent-soft text-accent" : "text-ink2 hover:bg-raised hover:text-ink",
                  )
                }
              >
                <Icon size={16} /> <span className="hidden sm:inline">{label}</span>
              </NavLink>
            ))}
          </nav>
        </div>
        <div className="mt-auto hidden space-y-3 p-5 md:block">
          <SystemStatus />
          <div className="flex items-start gap-2 px-1 text-[11px] leading-snug text-muted">
            <ShieldCheck size={14} className="mt-px shrink-0" />
            Research use only. Identities are pseudonymized and all data stays on this machine.
          </div>
        </div>
      </aside>
      <main className="min-w-0 flex-1">
        <div className="mx-auto max-w-7xl px-4 py-6 md:px-8 md:py-8">
          <Outlet />
        </div>
      </main>
    </div>
  );
}
