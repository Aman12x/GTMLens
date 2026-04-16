import { useRef, useState } from "react";
import { useMutation, useQuery } from "@tanstack/react-query";
import { BarChart3, Clock, Lock, Mail, RefreshCw, Send, Sparkles, Target, Upload, Users } from "lucide-react";
import { api, extractError } from "@/lib/api";
import { pct } from "@/lib/utils";
import { Card, CardHeader, CardTitle } from "@/components/ui/Card";
import { Button } from "@/components/ui/Button";
import { Input, Select } from "@/components/ui/Input";
import { Badge } from "@/components/ui/Badge";
import { Alert } from "@/components/ui/Alert";
import type { OutreachGenerateRequest, OutreachGenerateResponse, SegmentCateResult } from "@/types/api";

const TONES = ["direct", "warm", "technical"] as const;

const UPLIFT_BADGE: Record<string, "success" | "warning" | "muted"> = {
  high:     "success",
  marginal: "warning",
  low:      "muted",
};

const DEFAULT_CONTEXT =
  "GTMLens is a causal targeting engine that helps B2B SaaS GTM teams identify high-uplift customer segments and run statistically rigorous experiments to measure activation lift.";

export function OutreachPage() {
  const [form, setForm] = useState<OutreachGenerateRequest>({
    segment: { cate_estimate: 0.55, company_size: "enterprise", channel: "paid_search", funnel_stage: "activation" },
    product_context: DEFAULT_CONTEXT,
    tone: "direct",
  });
  const [result, setResult] = useState<OutreachGenerateResponse | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // ── Load CATE estimates from the API (the core causal capability)
  const cateQuery = useQuery({
    queryKey: ["segment-cate"],
    queryFn: () => api.segmentCate({ method: "t_learner", apply_bh: true }),
    staleTime: 5 * 60_000,
    retry: 1,
  });

  const generateMutation = useMutation({
    mutationFn: api.outreachGenerate,
    onSuccess: setResult,
  });

  const logsQuery = useQuery({
    queryKey: ["outreach-results"],
    queryFn: () => api.outreachResults(10),
    staleTime: 60_000,
  });

  const contactsQuery = useQuery({
    queryKey: ["contacts", form.segment.company_size, form.segment.channel],
    queryFn: () => api.listContacts(form.segment.company_size ?? undefined, form.segment.channel ?? undefined),
    staleTime: 30_000,
    enabled: !!(form.segment.company_size && form.segment.channel),
  });

  const uploadMutation = useMutation({
    mutationFn: (file: File) => api.uploadContacts(file),
    onSuccess: () => contactsQuery.refetch(),
  });

  const sendMutation = useMutation({
    mutationFn: () => api.sendSegment({
      segment_id: form.segment.segment_id ?? `${form.segment.company_size}_${form.segment.channel}`,
      company_size: form.segment.company_size ?? "",
      channel: form.segment.channel ?? "",
      cate_estimate: form.segment.cate_estimate,
      product_context: form.product_context,
      tone: form.tone ?? "direct",
    }),
    onSuccess: () => {
      logsQuery.refetch();
      contactsQuery.refetch();
    },
  });

  function applySegment(seg: SegmentCateResult) {
    setForm((f) => ({
      ...f,
      segment: {
        ...f.segment,
        cate_estimate: seg.mean_cate,
        company_size: seg.company_size,
        channel: seg.channel,
        segment_id: `${seg.company_size}_${seg.channel}`,
      },
      // Pass the percentile-based threshold so the backend guard uses the
      // same cutoff that was used to compute recommended_for_outreach.
      cate_threshold: cateQuery.data?.cate_threshold,
    }));
    setResult(null);
  }

  function handleGenerate(e: React.FormEvent) {
    e.preventDefault();
    generateMutation.mutate({
      ...form,
      cate_threshold: form.cate_threshold ?? cateQuery.data?.cate_threshold,
    });
    logsQuery.refetch();
  }

  const threshold = cateQuery.data?.cate_threshold ?? 0.0;

  return (
    <div className="space-y-6">
      {/* ── Segment CATE table ───────────────────────────────────────── */}
      <Card>
        <CardHeader>
          <CardTitle>
            <Target className="mr-1.5 inline h-3.5 w-3.5" />
            Who responds to outreach
          </CardTitle>
          <div className="flex items-center gap-2">
            {cateQuery.data && (
              <span className="text-xs text-muted">
                {cateQuery.data.n_significant} / {cateQuery.data.segments.length} segments with reliable lift
              </span>
            )}
            <Button
              variant="ghost"
              size="sm"
              onClick={() => cateQuery.refetch()}
              loading={cateQuery.isFetching}
            >
              <RefreshCw className="h-3.5 w-3.5" />
            </Button>
          </div>
        </CardHeader>

        {cateQuery.isError && (
          <Alert variant="warning">
            {extractError(cateQuery.error)} — segment analysis requires the demo dataset to be seeded.
          </Alert>
        )}

        {cateQuery.isLoading && (
          <p className="text-sm text-faint">Analyzing segments…</p>
        )}

        {cateQuery.data && cateQuery.data.segments.length > 0 && (
          <div className="overflow-x-auto">
            <table className="w-full text-xs">
              <thead>
                <tr className="border-b border-border text-left">
                  <th className="pb-2 pr-4 font-medium text-faint">Segment</th>
                  <th className="pb-2 pr-4 font-medium text-faint">Predicted lift</th>
                  <th className="pb-2 pr-4 font-medium text-faint">Observed lift</th>
                  <th className="pb-2 pr-4 font-medium text-faint">Confidence</th>
                  <th className="pb-2 pr-4 font-medium text-faint">N (T/C)</th>
                  <th className="pb-2 font-medium text-faint">Action</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-border">
                {cateQuery.data.segments.map((seg) => (
                  <tr
                    key={`${seg.company_size}_${seg.channel}`}
                    className="hover:bg-overlay/50 transition-colors"
                  >
                    <td className="py-2 pr-4">
                      <div className="flex items-center gap-1.5">
                        <span className="font-medium text-text">{seg.company_size}</span>
                        <span className="text-faint">·</span>
                        <span className="text-muted">{seg.channel}</span>
                      </div>
                    </td>
                    <td className="py-2 pr-4 font-mono font-medium text-text">
                      {pct(seg.mean_cate)}
                    </td>
                    <td className="py-2 pr-4 font-mono text-muted">
                      {seg.segment_ate >= 0 ? "+" : ""}{pct(seg.segment_ate)}
                    </td>
                    <td className="py-2 pr-4">
                      <span className={seg.significant_bh ? "text-green" : "text-muted"}>
                        {seg.p_value_raw < 0.001 ? "<0.001" : seg.p_value_raw.toFixed(3)}
                        {seg.significant_bh && " ✓"}
                      </span>
                    </td>
                    <td className="py-2 pr-4 font-mono text-muted">
                      {seg.n_treatment}/{seg.n_control}
                    </td>
                    <td className="py-2">
                      {seg.recommended_for_outreach ? (
                        <Button size="sm" variant="primary" onClick={() => applySegment(seg)}>
                          Use
                        </Button>
                      ) : (
                        <Badge variant="muted">
                          {seg.mean_cate < threshold ? "Below threshold" : "Not significant"}
                        </Badge>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
            <p className="mt-3 text-xs text-faint">
              Showing top {Math.round((1 - 0.6) * 100)}% of segments by predicted lift ·{" "}
              N={cateQuery.data.n_users.toLocaleString()} signed-up users
            </p>
          </div>
        )}
      </Card>

      {/* ── Generate + Result ────────────────────────────────────────── */}
      <div className="grid gap-6 lg:grid-cols-[400px_1fr]">
        {/* Config panel */}
        <Card>
          <CardHeader>
            <CardTitle>
              <Sparkles className="mr-1.5 inline h-3.5 w-3.5" />
              Message configuration
            </CardTitle>
          </CardHeader>
          <form onSubmit={handleGenerate} className="space-y-4">
            <div className="grid grid-cols-2 gap-3">
              <Input
                label="Company size"
                value={form.segment.company_size ?? ""}
                readOnly
              />
              <Input
                label="Channel"
                value={form.segment.channel ?? ""}
                readOnly
              />
            </div>
            <Input
              label="Predicted lift"
              type="number"
              min={0}
              max={1}
              step={0.01}
              value={form.segment.cate_estimate}
              onChange={(e) =>
                setForm((f) => ({
                  ...f,
                  segment: { ...f.segment, cate_estimate: parseFloat(e.target.value) },
                }))
              }
              hint={`Threshold: ${pct(threshold)} — click "Use" in the table above`}
            />
            <Select
              label="Tone"
              value={form.tone ?? "direct"}
              onChange={(e) =>
                setForm((f) => ({ ...f, tone: e.target.value as typeof form.tone }))
              }
            >
              {TONES.map((t) => (
                <option key={t} value={t}>{t}</option>
              ))}
            </Select>
            <div className="space-y-1.5">
              <label className="text-xs font-medium text-muted">Product context</label>
              <textarea
                className="w-full resize-none rounded border border-border bg-overlay px-3 py-2 text-sm
                  text-text placeholder:text-faint focus:border-accent focus:outline-none
                  focus:ring-1 focus:ring-accent/30"
                rows={3}
                value={form.product_context}
                onChange={(e) => setForm((f) => ({ ...f, product_context: e.target.value }))}
              />
            </div>
            <Button
              type="submit"
              loading={generateMutation.isPending}
              className="w-full"
              disabled={(form.segment.cate_estimate ?? 0) < threshold}
            >
              <Mail className="h-4 w-4" />
              Generate outreach
            </Button>
            {(form.segment.cate_estimate ?? 0) < threshold && (
              <p className="text-center text-xs text-yellow">
                Select a recommended segment from the table above
              </p>
            )}
          </form>
        </Card>

        {/* Output panel */}
        <div className="space-y-4">
          {generateMutation.isError && (
            <Alert variant="danger" title="Generation failed">
              {extractError(generateMutation.error)}
            </Alert>
          )}

          {result && (
            <>
              {result.holdout_flag && (
                <Alert variant="warning" title="Holdout group — do not send">
                  This user is in the 20% holdout control group. The message was generated for
                  preview only. Sending contaminate the lift measurement.
                </Alert>
              )}
              {result.is_fallback && (
                <Alert variant="warning" title="Claude API unavailable">
                  Showing fallback message. Configure ANTHROPIC_API_KEY to enable generation.
                </Alert>
              )}

              <Card>
                <CardHeader>
                  <CardTitle>
                    <Mail className="mr-1.5 inline h-3.5 w-3.5" />
                    Generated message
                  </CardTitle>
                  <div className="flex items-center gap-2">
                    <Badge variant={UPLIFT_BADGE[result.predicted_uplift_group] ?? "muted"}>
                      {result.predicted_uplift_group} uplift
                    </Badge>
                    {result.holdout_flag ? (
                      <Badge variant="warning">
                        <Lock className="mr-1 h-3 w-3" />
                        Holdout
                      </Badge>
                    ) : (
                      <Badge variant="success">Ready to send</Badge>
                    )}
                  </div>
                </CardHeader>
                <div className="space-y-3">
                  <div className="rounded-lg border border-border bg-overlay p-4">
                    <p className="mb-1 text-xs font-medium text-faint">Subject</p>
                    <p className="text-sm font-medium text-text">{result.subject}</p>
                  </div>
                  <div className="rounded-lg border border-border bg-overlay p-4">
                    <p className="mb-2 text-xs font-medium text-faint">Body</p>
                    <p className="whitespace-pre-wrap text-sm leading-relaxed text-text">
                      {result.body}
                    </p>
                  </div>
                  <div className="flex items-center gap-3 rounded-lg border border-accent/20 bg-accent/5 p-4">
                    <div className="h-2 w-2 shrink-0 rounded-full bg-accent" />
                    <div>
                      <p className="text-xs font-medium text-faint">Call to action</p>
                      <p className="text-sm font-medium text-accent">{result.cta}</p>
                    </div>
                  </div>
                  <p className="text-xs text-faint">
                    Segment ID: <span className="font-mono">{result.segment_id}</span>
                  </p>
                </div>
              </Card>
            </>
          )}

          {/* ── Contacts + Send ──────────────────────────────── */}
          <Card>
            <CardHeader>
              <CardTitle>
                <Users className="mr-1.5 inline h-3.5 w-3.5" />
                Contacts in segment
              </CardTitle>
              <div className="flex items-center gap-2">
                {contactsQuery.data && (
                  <span className="text-xs text-muted">{contactsQuery.data.total} contacts</span>
                )}
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => fileInputRef.current?.click()}
                  loading={uploadMutation.isPending}
                >
                  <Upload className="h-3.5 w-3.5" />
                  Upload CSV
                </Button>
                <input
                  ref={fileInputRef}
                  type="file"
                  accept=".csv"
                  className="hidden"
                  onChange={(e) => {
                    const file = e.target.files?.[0];
                    if (file) uploadMutation.mutate(file);
                    e.target.value = "";
                  }}
                />
              </div>
            </CardHeader>

            {uploadMutation.data && (
              <Alert variant="success" className="mb-3">
                Uploaded — {uploadMutation.data.inserted} new,{" "}
                {uploadMutation.data.updated} updated,{" "}
                {uploadMutation.data.skipped} skipped
                {uploadMutation.data.errors.length > 0 && (
                  <ul className="mt-1 list-disc pl-4 text-xs">
                    {uploadMutation.data.errors.map((e, i) => <li key={i}>{e}</li>)}
                  </ul>
                )}
              </Alert>
            )}
            {uploadMutation.isError && (
              <Alert variant="danger" className="mb-3">{extractError(uploadMutation.error)}</Alert>
            )}

            {contactsQuery.data && contactsQuery.data.contacts.length > 0 ? (
              <>
                <div className="max-h-40 overflow-y-auto">
                  <table className="w-full text-xs">
                    <thead>
                      <tr className="border-b border-border text-left text-faint">
                        <th className="pb-2 pr-4 font-medium">Email</th>
                        <th className="pb-2 pr-4 font-medium">Name</th>
                        <th className="pb-2 font-medium">Company</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-border">
                      {contactsQuery.data.contacts.slice(0, 20).map((c) => (
                        <tr key={c.id} className="text-muted">
                          <td className="py-1.5 pr-4 font-mono">{c.email}</td>
                          <td className="py-1.5 pr-4">{c.first_name ?? "—"}</td>
                          <td className="py-1.5">{c.company ?? "—"}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                  {contactsQuery.data.total > 20 && (
                    <p className="mt-2 text-center text-xs text-faint">
                      +{contactsQuery.data.total - 20} more
                    </p>
                  )}
                </div>

                {sendMutation.data && (
                  <Alert variant="success" className="mt-3">
                    Sent {sendMutation.data.sent}, held out {sendMutation.data.held_out}
                    {sendMutation.data.failed > 0 && `, ${sendMutation.data.failed} failed`}
                  </Alert>
                )}
                {sendMutation.isError && (
                  <Alert variant="danger" className="mt-3">{extractError(sendMutation.error)}</Alert>
                )}

                <Button
                  className="mt-3 w-full"
                  disabled={!result || result.holdout_flag}
                  loading={sendMutation.isPending}
                  onClick={() => sendMutation.mutate()}
                >
                  <Send className="h-4 w-4" />
                  Send to {contactsQuery.data.total} contacts
                </Button>
                {(!result) && (
                  <p className="mt-1 text-center text-xs text-faint">
                    Generate a message above before sending
                  </p>
                )}
              </>
            ) : (
              <div className="space-y-2 py-2 text-center">
                <p className="text-xs text-muted">No contacts for this segment yet.</p>
                <p className="text-xs text-faint">
                  Upload a CSV with columns: email, first_name, company, company_size, channel
                </p>
              </div>
            )}
          </Card>

          {/* Recent sends log */}
          <Card>
            <CardHeader>
              <CardTitle>
                <BarChart3 className="mr-1.5 inline h-3.5 w-3.5" />
                Recent sends
              </CardTitle>
              {logsQuery.data && (
                <span className="text-xs text-muted">{logsQuery.data.total} total</span>
              )}
            </CardHeader>
            {logsQuery.isLoading && <p className="text-sm text-faint">Loading…</p>}
            {logsQuery.data?.results.length === 0 && (
              <p className="text-sm text-faint">No messages sent yet.</p>
            )}
            {logsQuery.data?.results && logsQuery.data.results.length > 0 && (
              <div className="overflow-x-auto">
                <table className="w-full text-xs">
                  <thead>
                    <tr className="border-b border-border text-left text-faint">
                      <th className="pb-2 pr-4 font-medium">Segment</th>
                      <th className="pb-2 pr-4 font-medium">Predicted lift</th>
                      <th className="pb-2 pr-4 font-medium">Tone</th>
                      <th className="pb-2 pr-4 font-medium">Holdout</th>
                      <th className="pb-2 font-medium">Sent</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-border">
                    {logsQuery.data.results.map((row, i) => (
                      <tr key={i} className="text-muted transition-colors hover:text-text">
                        <td className="py-2 pr-4 font-mono">{row.segment_id}</td>
                        <td className="py-2 pr-4">{pct(row.cate_estimate)}</td>
                        <td className="py-2 pr-4 capitalize">{row.tone}</td>
                        <td className="py-2 pr-4">
                          {row.is_holdout ? (
                            <Badge variant="warning">yes</Badge>
                          ) : (
                            <Badge variant="success">no</Badge>
                          )}
                        </td>
                        <td className="flex items-center gap-1 py-2">
                          <Clock className="h-3 w-3 shrink-0" />
                          {new Date(row.sent_at).toLocaleDateString()}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </Card>
        </div>
      </div>
    </div>
  );
}
