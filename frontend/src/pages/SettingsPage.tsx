import { CheckCircle2, Eye, EyeOff, KeyRound, RefreshCw, ShieldCheck, Trash2, XCircle } from "lucide-react";
import { useState } from "react";
import { api, usePoll, type TokenInfo } from "../api";
import { Button, Card, CardHeader, Empty, cx, fmt } from "../components/ui";

const NEW_TOKEN_URL = "https://github.com/settings/personal-access-tokens/new";

function TokenRow({ t, onChange }: { t: TokenInfo; onChange: () => void }) {
  const [busy, setBusy] = useState(false);
  const run = async (fn: () => Promise<unknown>) => {
    setBusy(true);
    try {
      await fn();
      onChange();
    } finally {
      setBusy(false);
    }
  };
  const pct = t.limit ? (t.remaining ?? 0) / t.limit : null;
  return (
    <li className="flex flex-wrap items-center gap-4 px-5 py-4">
      <div className={cx("grid h-9 w-9 shrink-0 place-items-center rounded-xl", t.valid ? "bg-accent-soft text-accent" : "bg-raised text-bad")}>
        <KeyRound size={16} />
      </div>
      <div className="min-w-0 flex-1">
        <div className="flex flex-wrap items-center gap-2">
          <span className="font-mono text-sm text-ink">•••• {t.last4}</span>
          {t.login && <span className="text-sm text-ink2">@{t.login}</span>}
          {t.source === "env" ? (
            <span className="rounded-full bg-raised px-2 py-0.5 text-[11px] text-ink2">GITHUB_TOKENS</span>
          ) : t.valid ? (
            <span className="inline-flex items-center gap-1 text-xs text-good">
              <CheckCircle2 size={13} /> Valid
            </span>
          ) : (
            <span className="inline-flex items-center gap-1 text-xs text-bad">
              <XCircle size={13} /> Rejected by GitHub
            </span>
          )}
        </div>
        <div className="mt-0.5 text-xs text-muted">
          {t.source === "env"
            ? "Set in the environment — edit backend/.env to change it."
            : `Added ${fmt.date(t.added_at)}${t.checked_at ? ` · checked ${fmt.ago(t.checked_at)}` : ""}`}
        </div>
      </div>
      {pct !== null && t.source === "ui" && (
        <div className="w-40">
          <div className="mb-1 flex justify-between text-[11px] text-muted">
            <span>Quota at last check</span>
            <span className="tabular">{fmt.int(t.remaining)}</span>
          </div>
          <div className="h-1.5 overflow-hidden rounded-full bg-raised">
            <div className="h-full rounded-full bg-accent" style={{ width: `${pct * 100}%` }} />
          </div>
        </div>
      )}
      {t.source === "ui" && t.id != null && (
        <div className="flex gap-1">
          <Button variant="ghost" className="px-2" title="Check again" aria-label="Check token again" disabled={busy} onClick={() => run(() => api.checkToken(t.id!))}>
            <RefreshCw size={15} className={busy ? "animate-spin" : undefined} />
          </Button>
          <Button
            variant="ghost"
            className="px-2 text-bad"
            title="Remove token"
            aria-label="Remove token"
            disabled={busy}
            onClick={() => {
              if (confirm(`Remove the token ending in ${t.last4}? Running jobs stop using it within 30 seconds.`))
                run(() => api.removeToken(t.id!));
            }}
          >
            <Trash2 size={15} />
          </Button>
        </div>
      )}
    </li>
  );
}

export default function SettingsPage() {
  const { data, refresh } = usePoll(api.tokens, 0);
  const [value, setValue] = useState("");
  const [show, setShow] = useState(false);
  const [busy, setBusy] = useState(false);
  const [msg, setMsg] = useState<{ kind: "ok" | "err"; text: string } | null>(null);

  async function add() {
    setBusy(true);
    setMsg(null);
    try {
      const t = await api.addToken(value.trim());
      setValue("");
      setShow(false);
      setMsg({ kind: "ok", text: `Token for @${t.login} added. The worker starts using it within 30 seconds.` });
      refresh();
    } catch (e) {
      setMsg({ kind: "err", text: (e as Error).message });
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="space-y-6">
      <header>
        <h1 className="text-2xl font-semibold tracking-tight text-ink">Settings</h1>
        <p className="mt-1 text-sm text-ink2">Configure access to the GitHub API.</p>
      </header>

      <div className="grid gap-6 lg:grid-cols-5">
        <div className="space-y-6 lg:col-span-3">
          <Card className="p-5">
            <label htmlFor="token" className="text-sm font-semibold text-ink">
              Add a GitHub token
            </label>
            <p className="mt-0.5 text-xs text-muted">
              It is checked with GitHub before being saved. Several tokens are used in rotation when one runs out of quota.
            </p>
            <div className="mt-3 flex gap-2">
              <div className="relative flex-1">
                <input
                  id="token"
                  type={show ? "text" : "password"}
                  value={value}
                  onChange={(e) => setValue(e.target.value)}
                  onKeyDown={(e) => e.key === "Enter" && value.trim() && add()}
                  placeholder="github_pat_… or ghp_…"
                  autoComplete="off"
                  spellCheck={false}
                  className="w-full rounded-lg border border-line bg-page py-2 pl-3 pr-10 font-mono text-sm text-ink placeholder:text-muted focus:border-accent focus:outline-none focus:ring-2 focus:ring-accent-soft"
                />
                <button
                  type="button"
                  onClick={() => setShow((v) => !v)}
                  aria-label={show ? "Hide token" : "Show token"}
                  className="absolute right-2 top-1/2 -translate-y-1/2 rounded p-1 text-muted hover:text-ink"
                >
                  {show ? <EyeOff size={15} /> : <Eye size={15} />}
                </button>
              </div>
              <Button variant="primary" onClick={add} disabled={!value.trim() || busy}>
                {busy ? "Checking…" : "Add"}
              </Button>
            </div>
            {msg && <p className={cx("mt-2 text-sm", msg.kind === "err" ? "text-bad" : "text-good")}>{msg.text}</p>}
          </Card>

          <Card className="overflow-hidden">
            <CardHeader title="Configured tokens" subtitle="Only the last four characters are ever shown." />
            {data && data.tokens.length === 0 ? (
              <Empty icon={<KeyRound size={22} />} title="No token yet">
                Without a token GitHub allows 60 requests per hour, so mining even a small repository takes hours.
              </Empty>
            ) : (
              <ul className="mt-3 divide-y divide-line border-t border-line">
                {data?.tokens.map((t) => <TokenRow key={`${t.source}-${t.id ?? t.last4}`} t={t} onChange={refresh} />)}
              </ul>
            )}
          </Card>
        </div>

        <div className="space-y-6 lg:col-span-2">
          <Card className="p-5">
            <h3 className="text-sm font-semibold text-ink">Which token?</h3>
            <ol className="mt-3 list-decimal space-y-2 pl-4 text-sm text-ink2">
              <li>
                Open{" "}
                <a href={NEW_TOKEN_URL} target="_blank" rel="noreferrer" className="text-accent hover:underline">
                  GitHub → Fine-grained tokens
                </a>
                .
              </li>
              <li>
                Repository access: <span className="font-medium text-ink">Public repositories (read-only)</span>.
              </li>
              <li>No extra permissions are needed. Pick an expiry date.</li>
              <li>Generate, copy and paste it here.</li>
            </ol>
          </Card>
          <Card className="px-5 py-4">
            <div className="flex gap-3 text-xs leading-relaxed text-ink2">
              <ShieldCheck size={16} className="mt-0.5 shrink-0 text-good" />
              <p>
                Tokens are stored only in the local database on this machine (plain text, like a <code>.env</code> file) and
                are never sent back to the browser. The API accepts requests only from this app.
              </p>
            </div>
          </Card>
        </div>
      </div>
    </div>
  );
}
