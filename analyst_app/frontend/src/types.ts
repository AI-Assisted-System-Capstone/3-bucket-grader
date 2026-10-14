export type Severity = {
  grade: string;
  grade_probabilities: Record<string, number>;
  bucket: string;
  bucket_probabilities: Record<string, number>;
  triage_score_raw: number;
  harm_probability_calibrated: number;
  priority_flag: boolean;
  model_version: string;
  fixture: boolean;
};
export type EventType = {
  event_type_code: string;
  event_type_label: string;
  confidence: number;
  probabilities: Record<string, number>;
  model_version: string;
  fixture: boolean;
};
export type Match = {
  event_id: string;
  source_event_id?: string | null;
  similarity: number;
  event_date: string;
  location_name: string;
};
export type Membership = {
  theme_id: string;
  label: string;
  membership_score: number;
};
export type Trend = {
  theme_id: string;
  current_count: number;
  baseline_count: number;
  percent_change: number | null;
  trend_score: number;
  status: "stable" | "watch" | "emerging";
  timeline: { week: string; count: number }[];
  period_start: string;
  period_end: string;
  as_of: string;
};
export type AI = {
  event_id: string;
  severity: Severity | null;
  event_type: EventType | null;
  themes: Membership[];
  similar_events: Match[];
  trend_context: Trend[];
  errors: Record<string, string>;
  status: string;
  created_at: string;
  model_versions: Record<string, string>;
};
export type Decision = {
  id: string;
  grade: string;
  event_type_code: string;
  note: string;
  created_at: string;
};
export type Event = {
  review_status?: "reviewed" | "unreviewed";
  id: string;
  event_id: string;
  event_comments: string;
  event_date: string;
  location_name: string;
  encounter_service: string;
  facility: string;
  age_at_encounter: number | null;
  prescribed_medication: string;
  prescribed_dose: string;
  administered_medication: string;
  administered_dose: string;
  suspect_medication: string;
  suspect_dose: string;
  ai?: AI;
  analyst_decisions?: Decision[];
};
export type Pattern = {
  trend_stale?: boolean;
  theme_id: string;
  name: string;
  keywords: string[];
  event_count: number;
  first_seen: string;
  last_seen: string;
  representative_event_ids: string[];
  location_distribution: Record<string, number>;
  facility_distribution: Record<string, number>;
  event_type_distribution: Record<string, number>;
  severity_distribution: Record<string, number>;
  trend: Trend | null;
};
export type Health = {
  fixture_mode: boolean;
  models: Record<string, string>;
  errors: Record<string, string>;
  rebuild: { status: string; error?: string };
};

export type DashboardData = {
  generated_at: string;
  fixture_mode: boolean;
  event_count: number;
  unreviewed_count: number;
  reviewed_count: number;
  priority_unreviewed_count: number;
  analysis_issue_count: number;
  pending_analysis_count: number;
  emerging_pattern_count: number;
  review_queue: DashboardReport[];
  recent_events: DashboardReport[];
  patterns: Pattern[];
};

export type DashboardReport = Pick<
  Event,
  "id" | "event_id" | "event_date" | "location_name" | "review_status"
> & {
  ai?: {
    status: string;
    severity?: Pick<Severity, "priority_flag" | "fixture"> | null;
  } | null;
};
