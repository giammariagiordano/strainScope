import { AlertTriangle, CheckCircle2, Eye, EyeOff, KeyRound, RefreshCw, ShieldCheck, Sparkles, Trash2, XCircle } from "lucide-react";
import { useState } from "react";
import { api, usePoll, type EmotionSettings, type TokenInfo } from "../api";
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

function SecretInput({ id, value, onChange, onEnter, placeholder }: { id: string; value: string; onChange: (v: string) => void; onEnter: () => void; placeholder: string }) {
  const [show, setShow] = useState(false);
  return (
    <div className="relative flex-1">
      <input
        id={id}
        type={show ? "text" : "password"}
        value={value}
        onChange={(e) => onChange(e.target.value)}
        onKeyDown={(e) => e.key === "Enter" && value.trim() && onEnter()}
        placeholder={placeholder}
        autoComplete="off"
        spellCheck={false}
        className="w-full rounded-lg border border-line bg-page py-2 pl-3 pr-10 font-mono text-sm text-ink placeholder:text-muted focus:border-accent focus:outline-none focus:ring-2 focus:ring-accent-soft"
      />
      <button
        type="button"
        onClick={() => setShow((v) => !v)}
        aria-label={show ? "Hide key" : "Show key"}
        className="absolute right-2 top-1/2 -translate-y-1/2 rounded p-1 text-muted hover:text-ink"
      >
        {show ? <EyeOff size={15} /> : <Eye size={15} />}
      </button>
    </div>
  );
}

const BACKENDS: { value: EmotionSettings["backend"]; title: string; text: string }[] = [
  { value: "roberta", title: "Local RoBERTa", text: "Fine-tuned on GitHub and Stack Overflow, micro-F1 0.62. Runs on this machine: no text leaves it." },
  { value: "openai", title: "OpenAI model", text: "Each comment is sent to the OpenAI API. Answers are cached, so a text is never paid twice." },
];

function EmotionCard() {
  const { data, refresh } = usePoll(api.emotionSettings, 0);
  const [key, setKey] = useState("");
  const [model, setModel] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [msg, setMsg] = useState<{ kind: "ok" | "err"; text: string } | null>(null);
  const shownModel = model ?? data?.model ?? "";

  async function run(fn: () => Promise<unknown>, ok: string) {
    setBusy(true);
    setMsg(null);
    try {
      const r = await fn();
      setMsg({ kind: "ok", text: typeof r === "string" ? r : ok }); // fn may return its own message
      refresh();
    } catch (e) {
      setMsg({ kind: "err", text: (e as Error).message });
    } finally {
      setBusy(false);
    }
  }

  function choose(backend: EmotionSettings["backend"]) {
    if (!data || backend === data.backend) return;
    if (
      data.labelled_texts > 0 &&
      !confirm(
        `${fmt.int(data.labelled_texts)} texts already have emotion labels from the current classifier. They are kept, and labels from two classifiers are not comparable: re-run the Emotions stage on those projects after resetting them. Switch anyway?`,
      )
    )
      return;
    run(() => api.setEmotionSettings(backend, shownModel), backend === "openai" ? "OpenAI selected: used by the next Emotions stage." : "Local RoBERTa selected.");
  }

  if (!data) return null;
  const openai = data.backend === "openai";
  return (
    <Card className="p-5">
      <div className="flex items-center gap-2">
        <Sparkles size={16} className="text-accent" />
        <h2 className="text-sm font-semibold text-ink">Emotion classifier</h2>
      </div>
      <p className="mt-0.5 text-xs text-muted">Used by the Emotions stage. Changes apply to the next project the worker classifies.</p>

      <div role="radiogroup" aria-label="Emotion classifier" className="mt-3 grid gap-2 sm:grid-cols-2">
        {BACKENDS.map((b) => {
          const active = data.backend === b.value;
          const locked = b.value === "openai" && !data.key_last4;
          return (
            <button
              key={b.value}
              type="button"
              role="radio"
              aria-checked={active}
              disabled={busy || (locked && !active)}
              title={locked ? "Add an OpenAI API key first" : undefined}
              onClick={() => choose(b.value)}
              className={cx(
                "rounded-xl border p-3 text-left transition disabled:cursor-not-allowed disabled:opacity-60",
                active ? "border-accent bg-accent-soft" : "border-line bg-surface hover:bg-raised",
              )}
            >
              <div className="text-sm font-medium text-ink">{b.title}</div>
              <div className="mt-0.5 text-xs leading-relaxed text-ink2">{b.text}</div>
            </button>
          );
        })}
      </div>

      <div className="mt-4 grid gap-3 sm:grid-cols-[1fr_auto]">
        <div>
          <label htmlFor="openai-model" className="text-xs font-medium text-ink2">
            OpenAI model
          </label>
          <input
            id="openai-model"
            value={shownModel}
            onChange={(e) => setModel(e.target.value)}
            spellCheck={false}
            className="mt-1 w-full rounded-lg border border-line bg-page px-3 py-2 font-mono text-sm text-ink focus:border-accent focus:outline-none focus:ring-2 focus:ring-accent-soft"
          />
        </div>
        <Button
          className="self-end"
          disabled={busy || !shownModel.trim() || shownModel === data.model}
          onClick={() => run(() => api.setEmotionSettings(data.backend, shownModel.trim()), `Model set to ${shownModel.trim()}.`)}
        >
          Save model
        </Button>
      </div>

      <div className="mt-4">
        <label htmlFor="openai-key" className="text-xs font-medium text-ink2">
          OpenAI API key
        </label>
        {data.key_last4 ? (
          <div className="mt-1 flex flex-wrap items-center gap-3 rounded-lg border border-line px-3 py-2">
            <KeyRound size={15} className="text-accent" />
            <span className="font-mono text-sm text-ink">•••• {data.key_last4}</span>
            {data.key_source === "env" ? (
              <span className="rounded-full bg-raised px-2 py-0.5 text-[11px] text-ink2">OPENAI_API_KEY</span>
            ) : (
              <span className="inline-flex items-center gap-1 text-xs text-good">
                <CheckCircle2 size={13} /> Checked with OpenAI
              </span>
            )}
            {data.key_source === "ui" && (
              <Button
                variant="ghost"
                className="ml-auto px-2 text-bad"
                title="Remove key"
                aria-label="Remove OpenAI key"
                disabled={busy}
                onClick={() => {
                  if (confirm("Remove the OpenAI key? The Emotions stage goes back to the local RoBERTa."))
                    run(() => api.removeOpenAIKey(), "Key removed.");
                }}
              >
                <Trash2 size={15} />
              </Button>
            )}
          </div>
        ) : (
          <div className="mt-1 flex gap-2">
            <SecretInput id="openai-key" value={key} onChange={setKey} placeholder="sk-…" onEnter={() => saveKey()} />
            <Button variant="primary" onClick={() => saveKey()} disabled={!key.trim() || busy}>
              {busy ? "Checking…" : "Save"}
            </Button>
          </div>
        )}
      </div>

      <div className="mt-4 flex flex-wrap items-center gap-3 border-t border-line pt-4">
        <Button
          disabled={busy}
          onClick={() => {
            if (
              confirm(
                `Re-run the Emotions stage, and the stages after it, on every project that is not running? Labels from a different classifier are cleared and those texts are classified again${openai ? " (paid OpenAI calls)" : ""}.`,
              )
            )
              run(async () => {
                const r = await api.reclassifyEmotions();
                return `${r.queued} projects queued.${r.running ? ` ${r.running} running: re-run them from their Pipeline tab when they finish.` : ""}`;
              }, "");
          }}
        >
          <RefreshCw size={15} /> Re-classify all projects
        </Button>
        <span className="text-xs text-muted">Needed after switching classifier: existing labels are not changed otherwise.</span>
      </div>

      {msg && <p className={cx("mt-3 text-sm", msg.kind === "err" ? "text-bad" : "text-good")}>{msg.text}</p>}
      {openai && (
        <p className="mt-3 flex gap-2 text-xs leading-relaxed text-warn">
          <AlertTriangle size={14} className="mt-0.5 shrink-0" />
          Comments are sent to OpenAI. Check that this fits your ethics approval and data-management plan.
        </p>
      )}
    </Card>
  );

  function saveKey() {
    run(async () => {
      await api.setOpenAIKey(key.trim());
      setKey("");
    }, "Key checked with OpenAI and saved. Select “OpenAI model” to use it.");
  }
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
        <p className="mt-1 text-sm text-ink2">Configure access to the GitHub API and the emotion classifier.</p>
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

          <EmotionCard />
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
