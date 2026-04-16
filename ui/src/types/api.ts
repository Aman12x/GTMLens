// ─── Analyze ────────────────────────────────────────────────────────────────

export interface AnalyzeRequest {
  company_size?: string;
  channel?: string;
  date_from?: string;
  date_to?: string;
}

export interface FunnelStage {
  stage: string;
  n: number;
  rate_from_prev: number | null;
  rate_from_impression: number;
}

export interface SrmResult {
  srm_detected: boolean;
  p_value: number;
  observed_split: number;
  recommendation: string;
}

export interface CupedResult {
  ate: number;
  ate_se: number;
  p_value: number;
  ci_lower: number;
  ci_upper: number;
  variance_reduction_pct: number;
  n_treatment: number;
  n_control: number;
}

export interface DailyTrendPoint {
  date: string;
  impressions: number;
  clicks: number;
  signups: number;
  activations: number;
  conversions: number;
  treatment_activation_rate: number;
  control_activation_rate: number;
}

export interface AnalyzeResponse {
  total_users: number;
  funnel: FunnelStage[];
  srm: SrmResult;
  cuped: CupedResult | null;
  daily_trend: DailyTrendPoint[];
  filters_applied: Record<string, string>;
}

// ─── Experiment ──────────────────────────────────────────────────────────────

export interface ExperimentDesignRequest {
  baseline_rate: number;
  mde: number;
  alpha?: number;
  power?: number;
  daily_traffic?: number;
  use_cuped?: boolean;
  variance_reduction?: number;
  treatment_split?: number;
  guardrail_metrics?: string[];
}

export interface ExperimentDesignResponse {
  required_n_per_arm: number;
  required_n_total: number;
  naive_n_per_arm: number;
  duration_days: number | null;
  treatment_split: number;
  alpha: number;
  power: number;
  baseline_rate: number;
  mde: number;
  treatment_rate: number;
  cuped_applied: boolean;
  variance_reduction_pct: number;
  primary_metric: string;
  guardrail_metrics: string[];
  notes: string;
}

// ─── Outreach ────────────────────────────────────────────────────────────────

export interface OutreachGenerateRequest {
  segment: {
    cate_estimate: number;
    company_size?: string;
    industry?: string;
    channel?: string;
    funnel_stage?: string;
    segment_id?: string;
  };
  product_context: string;
  tone?: "warm" | "direct" | "technical";
  cate_threshold?: number;
  user_id?: string;
}

export interface OutreachGenerateResponse {
  subject: string;
  body: string;
  cta: string;
  predicted_uplift_group: "high" | "marginal" | "low";
  holdout_flag: boolean;
  segment_id: string;
  is_fallback?: boolean;
}

export interface OutreachLogEntry {
  sent_at: string;
  segment_id: string;
  company_size: string;
  industry: string;
  channel: string;
  cate_estimate: number;
  tone: string;
  subject_hash: string;
  body_hash: string;
  is_holdout: boolean;
}

export interface OutreachResultsResponse {
  results: OutreachLogEntry[];
  total: number;
}

export interface LiftSegment {
  segment_id: string;
  company_size: string;
  channel: string;
  predicted_cate: number;
  observed_lift: number;
  treatment_rate: number;
  control_rate: number;
  n_sent: number;
  n_holdout: number;
  last_sent_at: string;
}

export interface LiftSummary {
  total_sent: number;
  total_holdout: number;
  avg_predicted_cate: number;
  avg_observed_lift: number;
  n_segments: number;
}

export interface LiftResponse {
  segments: LiftSegment[];
  summary: LiftSummary;
  data_source: "campaign" | "baseline";
}

// ─── Data upload ─────────────────────────────────────────────────────────────

export interface DataStatus {
  tenant_id: string;
  has_real_data: boolean;
  is_demo: boolean;
  message: string;
}

export interface UploadDataResult {
  n_users: number;
  n_treatment: number;
  n_control: number;
  activation_rate: number;
  message: string;
}

// ─── Contacts ────────────────────────────────────────────────────────────────

export interface Contact {
  id: number;
  email: string;
  first_name: string | null;
  company: string | null;
  company_size: string | null;
  channel: string | null;
  industry: string | null;
  uploaded_at: string;
}

export interface ContactsListResponse {
  contacts: Contact[];
  total: number;
}

export interface UploadResult {
  inserted: number;
  updated: number;
  skipped: number;
  errors: string[];
}

export interface SendSegmentRequest {
  segment_id: string;
  company_size: string;
  channel: string;
  cate_estimate: number;
  product_context: string;
  tone: string;
}

export interface SendSegmentResult {
  sent: number;
  held_out: number;
  failed: number;
  errors: string[];
}

// ─── Segment CATE ────────────────────────────────────────────────────────────

export interface CateRequest {
  method?: "t_learner" | "s_learner";
  min_segment_n?: number;
  apply_bh?: boolean;
  bh_alpha?: number;
  date_from?: string;
  date_to?: string;
}

export interface SegmentCateResult {
  company_size: string;
  channel: string;
  mean_cate: number;
  n_treatment: number;
  n_control: number;
  segment_ate: number;
  p_value_raw: number;
  significant_bh: boolean;
  recommended_for_outreach: boolean;
}

export interface CateResponse {
  segments: SegmentCateResult[];
  method: string;
  n_users: number;
  bh_applied: boolean;
  bh_alpha: number;
  cate_threshold: number;
  n_significant: number;
}

// ─── Narrative ───────────────────────────────────────────────────────────────

export interface NarrativeRequest {
  experiment_result: {
    ate: number;
    p_value: number;
    ci_lower?: number;
    ci_upper?: number;
    ate_se?: number;
    variance_reduction_pct?: number;
    n_treatment?: number;
    n_control?: number;
    guardrail_results?: Record<string, string>;
    segment_breakdown?: Array<{ segment: string; ate: number; p_value: number }>;
  };
  metric_hierarchy?: {
    nsm?: string;
    primary_metric?: string;
    secondary_metrics?: string[];
    guardrail_metrics?: string[];
  };
  recommendation?: "ship" | "iterate" | "abort";
}

export interface NarrativeResponse {
  outcome: string;
  driver: string;
  guardrails: Record<string, string>;
  recommendation: "SHIP" | "ITERATE" | "ABORT";
  rationale: string;
  is_fallback?: boolean;
}
