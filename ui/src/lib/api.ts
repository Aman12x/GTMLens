import axios from "axios";
import { clearToken, getToken } from "@/lib/auth";
import type {
  AnalyzeRequest,
  AnalyzeResponse,
  CateRequest,
  CateResponse,
  ContactsListResponse,
  DataStatus,
  ExperimentDesignRequest,
  ExperimentDesignResponse,
  LiftResponse,
  NarrativeRequest,
  NarrativeResponse,
  OutreachGenerateRequest,
  OutreachGenerateResponse,
  OutreachResultsResponse,
  SendSegmentRequest,
  SendSegmentResult,
  UploadDataResult,
  UploadResult,
} from "@/types/api";

const client = axios.create({
  baseURL: import.meta.env.VITE_API_BASE_URL ?? "",
  headers: { "Content-Type": "application/json" },
  timeout: 30_000,
});

// Attach JWT on every request if present
client.interceptors.request.use((config) => {
  const token = getToken();
  if (token) config.headers.Authorization = `Bearer ${token}`;
  return config;
});

// On 401, clear stale token so the auth page is shown
client.interceptors.response.use(
  (r) => r,
  (err) => {
    if (axios.isAxiosError(err) && err.response?.status === 401) {
      clearToken();
      window.dispatchEvent(new Event("gtmlens:logout"));
    }
    return Promise.reject(err);
  }
);

export const api = {
  // ── Auth ──────────────────────────────────────────────────────────────────
  login(email: string, password: string): Promise<{ access_token: string }> {
    const form = new URLSearchParams({ username: email, password });
    return client
      .post("/api/alpha/auth/login", form, {
        headers: { "Content-Type": "application/x-www-form-urlencoded" },
      })
      .then((r) => r.data);
  },

  register(email: string, password: string): Promise<{ id: number; email: string }> {
    return client.post("/api/alpha/auth/register", { email, password }).then((r) => r.data);
  },

  me(): Promise<{ id: number; email: string }> {
    return client.get("/api/alpha/auth/me").then((r) => r.data);
  },

  // ── Data upload ───────────────────────────────────────────────────────────
  dataStatus(): Promise<DataStatus> {
    return client.get("/api/data/status").then((r) => r.data);
  },

  uploadData(file: File): Promise<UploadDataResult> {
    const form = new FormData();
    form.append("file", file);
    return client
      .post<UploadDataResult>("/api/data/upload", form, {
        headers: { "Content-Type": "multipart/form-data" },
      })
      .then((r) => r.data);
  },

  // ── Analysis ───────────────────────────────────────────────────────────────
  analyze(req: AnalyzeRequest): Promise<AnalyzeResponse> {
    return client.post<AnalyzeResponse>("/api/analyze", req).then((r) => r.data);
  },

  experimentDesign(req: ExperimentDesignRequest): Promise<ExperimentDesignResponse> {
    return client
      .post<ExperimentDesignResponse>("/api/experiment/design", req)
      .then((r) => r.data);
  },

  segmentCate(req: CateRequest = {}): Promise<CateResponse> {
    return client.post<CateResponse>("/api/segment/cate", req).then((r) => r.data);
  },

  outreachGenerate(req: OutreachGenerateRequest): Promise<OutreachGenerateResponse> {
    return client
      .post<OutreachGenerateResponse>("/api/outreach/generate", req)
      .then((r) => r.data);
  },

  outreachResults(limit = 20): Promise<OutreachResultsResponse> {
    return client
      .get<OutreachResultsResponse>(`/api/outreach/results?limit=${limit}`)
      .then((r) => r.data);
  },

  outreachLift(): Promise<LiftResponse> {
    return client.get<LiftResponse>("/api/outreach/lift").then((r) => r.data);
  },

  sendSegment(req: SendSegmentRequest): Promise<SendSegmentResult> {
    return client.post<SendSegmentResult>("/api/outreach/send-segment", req).then((r) => r.data);
  },

  uploadContacts(file: File): Promise<UploadResult> {
    const form = new FormData();
    form.append("file", file);
    return client
      .post<UploadResult>("/api/contacts/upload", form, {
        headers: { "Content-Type": "multipart/form-data" },
      })
      .then((r) => r.data);
  },

  listContacts(company_size?: string, channel?: string): Promise<ContactsListResponse> {
    const params = new URLSearchParams();
    if (company_size) params.set("company_size", company_size);
    if (channel) params.set("channel", channel);
    return client
      .get<ContactsListResponse>(`/api/contacts?${params}`)
      .then((r) => r.data);
  },

  deleteContact(id: number): Promise<void> {
    return client.delete(`/api/contacts/${id}`).then(() => undefined);
  },

  activateContacts(emails: string[]): Promise<{ updated: number; not_found: string[] }> {
    return client.post("/api/contacts/activate", { emails }).then((r) => r.data);
  },

  narrative(req: NarrativeRequest): Promise<NarrativeResponse> {
    return client.post<NarrativeResponse>("/api/narrative", req).then((r) => r.data);
  },

  // POST — not GET. The reset drops and recreates DuckDB tables, which is
  // a destructive state mutation. GET must be safe/idempotent per HTTP spec.
  demoReset(): Promise<{ status: string; message: string }> {
    return client.post("/api/demo/reset").then((r) => r.data);
  },
};

export function extractError(err: unknown): string {
  if (axios.isAxiosError(err)) {
    const detail = err.response?.data?.detail;
    if (typeof detail === "string") return detail;
    if (detail?.detail) return detail.detail;
    if (detail?.error) return detail.error;
    return err.message;
  }
  if (err instanceof Error) return err.message;
  return "An unexpected error occurred.";
}
