import { useCallback, useEffect, useRef, useState } from "react";

export type ProjectStatus = "queued" | "running" | "paused" | "done" | "failed";
export type StageStatus = "pending" | "running" | "paused" | "done" | "failed";

export interface ProjectSummary {
  id: number;
  full_name: string;
  url: string;
  status: ProjectStatus;
  pause_requested: boolean;
  current_stage: string | null;
  current_stage_label: string | null;
  current_message: string | null;
  stages_done: number;
  stages_total: number;
  error: string | null;
  created_at: string;
  finished_at: string | null;
  counts: { commits: number; issues: number; developers: number };
}

export interface Stage {
  name: string;
  label: string;
  description: string;
  status: StageStatus;
  done: number;
  total: number | null;
  message: string | null;
  started_at: string | null;
  finished_at: string | null;
}

export interface ProjectDetail extends ProjectSummary {
  stages: Stage[];
}

export interface LogEntry {
  id: number;
  ts: string;
  level: string;
  stage: string | null;
  message: string;
}

export interface MonthPoint {
  month: string;
  commits: number;
  off_hours_commits: number;
  issues: number;
  prs: number;
  comments: number;
  reviews: number;
  active: number;
  core: number;
  off_hours_share: number | null;
}

export interface Overview {
  project: ProjectSummary;
  totals: {
    commits: number;
    issues: number;
    prs: number;
    comments: number;
    reviews: number;
    developers: number;
    bots: number;
    off_hours_share: number | null;
  };
  truck_factor: {
    value: number;
    files: number;
    files_with_author: number;
    members: { id: number; name: string; files: number }[];
  } | null;
  monthly: MonthPoint[];
  heatmap: number[][];
  work_hours: [number, number];
  real_identities: boolean;
}

export interface DeveloperRow {
  id: number;
  name: string;
  pseudonym: string;
  is_bot: boolean;
  truck_factor_member: boolean;
  files_authored: number;
  first_activity: string | null;
  last_activity: string | null;
  active_months: number;
  core_months: number;
  commits: number;
  churn: number;
  off_hours_share: number | null;
  opened: number;
  comments: number;
  reviews: number;
  max_centrality: number;
}

export interface DeveloperMonthPoint {
  month: string;
  active: boolean;
  commits: number;
  churn: number;
  off_hours_commits: number;
  weekend_commits: number;
  issues_opened: number;
  prs_opened: number;
  comments: number;
  reviews: number;
  centrality: number | null;
  degree: number;
  is_core: boolean;
}

export interface DeveloperTimeline {
  id: number;
  name: string;
  truck_factor_member: boolean;
  files_authored: number;
  months: DeveloperMonthPoint[];
}

export interface Status {
  worker_alive: boolean;
  github: {
    remaining: number;
    limit: number;
    reset: number;
    tokens: number;
    waiting_until: number | null;
    checked_at: number;
  } | null;
  tokens_configured: number;
  real_identities: boolean;
}

export type Level = "developer" | "developer_month";
export type Method = "spearman" | "pearson";

export interface VariableDef {
  key: string;
  label: string;
  description: string;
  kind: "count" | "share" | "score" | "binary";
}

export interface Cell {
  r: number | null;
  p: number | null;
  q?: number | null;
  n: number;
}

export interface Pair {
  x: string;
  y: string;
  r: number;
  p: number;
  q: number;
  n: number;
}

export interface CorrelationResult {
  variables: string[];
  matrix: Cell[][];
  pairs: Pair[];
  rows: number;
  developers: number;
  projects: number;
  method: Method;
  controls: string[];
  level: Level;
  warnings: string[];
}

export interface AnalysisQuery {
  project_ids: number[] | null;
  level: Level;
  min_active_months: number;
  abandon_months: number;
}

export interface ScatterResult {
  points: { x: number; y: number; label: string }[];
  total: number;
  sampled: boolean;
}

export interface SurvivalCovariate {
  key: string;
  label: string;
  description: string;
  default: boolean;
}

export interface SurvivalQuery {
  project_ids: number[] | null;
  abandon_months: number;
  window: number;
  min_active_months: number;
  covariates: string[];
  log_counts: boolean;
  standardize: boolean;
  stratify: boolean;
  penalizer: number;
  cluster: string;
  ph_transform: string;
  group: string;
}

export interface TestResult {
  statistic: number;
  df: number;
  p: number;
}

/** Non-finite estimates (e.g. under separation) arrive as null. */
export interface CoxCoefficient {
  covariate: string;
  coef: number | null;
  hr: number | null;
  hr_lower: number | null;
  hr_upper: number | null;
  se: number | null;
  naive_se: number | null;
  z: number | null;
  p: number | null;
  transform: string;
  ph: TestResult | null;
}

export interface KmGroup {
  label: string;
  n: number;
  events: number;
  median: number | null;
  curve: { t: number; s: number; lo: number; hi: number }[];
}

export interface SurvivalResult {
  dataset: {
    subjects: number;
    late_entry: number;
    no_risk_time: number;
    events: number;
    at_risk_subjects: number;
    intervals: number;
  };
  cox:
    | { error: string }
    | {
        coefficients: CoxCoefficient[];
        log_likelihood: number;
        lr_test: TestResult;
        ph_global: TestResult | null;
        ph_transform: string;
        cluster: string;
        n_clusters: number | null;
        events: number;
        intervals: number;
        subjects: number;
        stratified: boolean;
        warnings: string[];
      };
  km: { groups: KmGroup[]; logrank_p: number | null };
}

export function exportUrl(projectIds?: number[]) {
  return projectIds?.length ? `/api/export?project_ids=${projectIds.join(",")}` : "/api/export";
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`/api${path}`, {
    ...init,
    headers: { "Content-Type": "application/json", ...(init?.headers ?? {}) },
  });
  if (!res.ok) {
    let detail = res.statusText;
    try {
      detail = (await res.json()).detail ?? detail;
    } catch {
      /* not JSON */
    }
    throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
  }
  return res.json() as Promise<T>;
}

export const api = {
  projects: () => request<ProjectSummary[]>("/projects"),
  project: (id: number) => request<ProjectDetail>(`/projects/${id}`),
  addProjects: (urls: string[]) =>
    request<{ created: string[]; skipped: string[] }>("/projects", { method: "POST", body: JSON.stringify({ urls }) }),
  pause: (id: number) => request(`/projects/${id}/pause`, { method: "POST" }),
  resume: (id: number) => request(`/projects/${id}/resume`, { method: "POST" }),
  rerun: (id: number, stage: string) => request(`/projects/${id}/rerun/${stage}`, { method: "POST" }),
  remove: (id: number) => request(`/projects/${id}`, { method: "DELETE" }),
  logs: (id: number) => request<LogEntry[]>(`/projects/${id}/logs`),
  overview: (id: number) => request<Overview>(`/projects/${id}/overview`),
  developers: (id: number) =>
    request<{ last_month: string | null; developers: DeveloperRow[] }>(`/projects/${id}/developers`),
  developer: (id: number, devId: number) => request<DeveloperTimeline>(`/projects/${id}/developers/${devId}`),
  status: () => request<Status>("/status"),
  variables: () =>
    request<{ defaults: { abandon_months: number }; levels: Record<Level, VariableDef[]> }>("/analysis/variables"),
  correlations: (q: AnalysisQuery & { method: Method; variables: string[]; controls: string[] }) =>
    request<CorrelationResult>("/analysis/correlations", { method: "POST", body: JSON.stringify(q) }),
  survivalOptions: () =>
    request<{
      covariates: SurvivalCovariate[];
      groupings: { key: string; label: string }[];
      clusters: { key: string; label: string }[];
      ph_transforms: string[];
      defaults: { abandon_months: number; window: number; min_active_months: number };
    }>("/survival/options"),
  survival: (q: SurvivalQuery) =>
    request<SurvivalResult>("/survival", { method: "POST", body: JSON.stringify(q) }),
  scatter: (q: AnalysisQuery & { x: string; y: string }) =>
    request<ScatterResult>("/analysis/scatter", { method: "POST", body: JSON.stringify(q) }),
};

/** Fetch on mount and every `intervalMs` (0 = once). Returns data, error and a manual refresh. */
export function usePoll<T>(fetcher: () => Promise<T>, intervalMs: number, deps: unknown[] = []) {
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState<string | null>(null);
  const fetcherRef = useRef(fetcher);
  fetcherRef.current = fetcher;

  const refresh = useCallback(async () => {
    try {
      setData(await fetcherRef.current());
      setError(null);
    } catch (e) {
      setError((e as Error).message);
    }
  }, []);

  useEffect(() => {
    setData(null);
    refresh();
    if (!intervalMs) return;
    const t = setInterval(refresh, intervalMs);
    return () => clearInterval(t);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [intervalMs, refresh, ...deps]);

  return { data, error, refresh };
}
