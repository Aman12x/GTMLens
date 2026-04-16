import { useQuery } from "@tanstack/react-query";
import { BarChart3, CheckCircle, RefreshCw, XCircle } from "lucide-react";
import { api, extractError } from "@/lib/api";
import { pct } from "@/lib/utils";
import { Card, CardHeader, CardTitle } from "@/components/ui/Card";
import { Button } from "@/components/ui/Button";
import { Badge } from "@/components/ui/Badge";
import { Alert } from "@/components/ui/Alert";
import { Stat } from "@/components/ui/Stat";

/** Difference between predicted CATE and observed lift, in percentage points. */
function accuracy(predicted: number, observed: number): number {
  return Math.abs(predicted - observed) * 100;
}

function AccuracyBadge({ predicted, observed }: { predicted: number; observed: number }) {
  const delta = accuracy(predicted, observed);
  if (delta <= 3)  return <Badge variant="success">±{delta.toFixed(1)}pp</Badge>;
  if (delta <= 8)  return <Badge variant="warning">±{delta.toFixed(1)}pp</Badge>;
  return <Badge variant="danger">±{delta.toFixed(1)}pp</Badge>;
}

function LiftBar({ predicted, observed }: { predicted: number; observed: number }) {
  const max = Math.max(predicted, observed, 0.01);
  const pPct = Math.min((predicted / max) * 100, 100);
  const oPct = Math.min((observed / max) * 100, 100);
  return (
    <div className="flex flex-col gap-1">
      <div className="flex items-center gap-2">
        <span className="w-16 text-right text-xs text-faint">Predicted</span>
        <div className="relative h-2 flex-1 rounded-full bg-surface">
          <div
            className="absolute left-0 h-2 rounded-full bg-accent/60"
            style={{ width: `${pPct}%` }}
          />
        </div>
        <span className="w-10 text-right font-mono text-xs text-muted">{pct(predicted)}</span>
      </div>
      <div className="flex items-center gap-2">
        <span className="w-16 text-right text-xs text-faint">Observed</span>
        <div className="relative h-2 flex-1 rounded-full bg-surface">
          <div
            className={`absolute left-0 h-2 rounded-full ${observed >= 0 ? "bg-green" : "bg-red"}`}
            style={{ width: `${oPct}%` }}
          />
        </div>
        <span className="w-10 text-right font-mono text-xs text-muted">{pct(observed)}</span>
      </div>
    </div>
  );
}

export function LiftPage() {
  const { data, isLoading, isError, error, refetch, isFetching } = useQuery({
    queryKey: ["outreach-lift"],
    queryFn: api.outreachLift,
    staleTime: 60_000,
  });

  const summary = data?.summary;

  return (
    <div className="space-y-6">
      {/* ── Summary stats ─────────────────────────────────────────── */}
      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <Stat
          label="Messages sent"
          value={summary ? summary.total_sent.toLocaleString() : "—"}
        />
        <Stat
          label="Holdout (control)"
          value={summary ? summary.total_holdout.toLocaleString() : "—"}
          sub="Never received outreach"
        />
        <Stat
          label="Avg predicted lift"
          value={summary ? pct(summary.avg_predicted_cate) : "—"}
          sub="Model estimate at send time"
        />
        <Stat
          label="Avg observed lift"
          value={summary ? pct(summary.avg_observed_lift) : "—"}
          sub="Treatment − control activation rate"
        />
      </div>

      {/* ── Segment lift table ────────────────────────────────────── */}
      <Card>
        <CardHeader>
          <CardTitle>
            <BarChart3 className="mr-1.5 inline h-3.5 w-3.5" />
            Predicted vs. observed lift by segment
          </CardTitle>
          <Button
            variant="ghost"
            size="sm"
            onClick={() => refetch()}
            loading={isFetching}
          >
            <RefreshCw className="h-3.5 w-3.5" />
          </Button>
        </CardHeader>

        {isError && (
          <Alert variant="danger">
            {extractError(error)}
          </Alert>
        )}

        {isLoading && (
          <p className="text-sm text-faint">Loading lift data…</p>
        )}

        {data && data.segments.length === 0 && (
          <div className="space-y-2 py-4 text-center">
            <p className="text-sm text-muted">No outreach sent yet.</p>
            <p className="text-xs text-faint">
              Go to the Outreach tab, select a recommended segment, and generate a message.
              Results will appear here once messages have been logged.
            </p>
          </div>
        )}

        {data && data.data_source === "baseline" && (
          <Alert variant="warning" className="mb-3">
            Showing historical baseline — these are estimates from your funnel data, not real campaign results.
            Import activation results on the Data tab to see actual lift from your outreach.
          </Alert>
        )}

        {data && data.segments.length > 0 && (
          <div className="overflow-x-auto">
            <table className="w-full text-xs">
              <thead>
                <tr className="border-b border-border text-left text-faint">
                  <th className="pb-2 pr-4 font-medium">Segment</th>
                  <th className="pb-2 pr-4 font-medium">Lift comparison</th>
                  <th className="pb-2 pr-4 font-medium">Treatment rate</th>
                  <th className="pb-2 pr-4 font-medium">Control rate</th>
                  <th className="pb-2 pr-4 font-medium">Model accuracy</th>
                  <th className="pb-2 pr-4 font-medium">Sent / Holdout</th>
                  <th className="pb-2 font-medium">Signal</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-border">
                {data.segments.map((seg) => (
                  <tr key={seg.segment_id} className="transition-colors hover:bg-overlay/50">
                    <td className="py-3 pr-4">
                      <div className="flex flex-col gap-0.5">
                        <span className="font-medium text-text">{seg.company_size}</span>
                        <span className="text-faint">{seg.channel}</span>
                      </div>
                    </td>
                    <td className="py-3 pr-4 min-w-[200px]">
                      <LiftBar
                        predicted={seg.predicted_cate}
                        observed={seg.observed_lift}
                      />
                    </td>
                    <td className="py-3 pr-4 font-mono text-green">
                      {pct(seg.treatment_rate)}
                    </td>
                    <td className="py-3 pr-4 font-mono text-muted">
                      {pct(seg.control_rate)}
                    </td>
                    <td className="py-3 pr-4">
                      <AccuracyBadge
                        predicted={seg.predicted_cate}
                        observed={seg.observed_lift}
                      />
                    </td>
                    <td className="py-3 pr-4 font-mono text-muted">
                      {seg.n_sent} / {seg.n_holdout}
                    </td>
                    <td className="py-3">
                      {seg.observed_lift > 0 ? (
                        <CheckCircle className="h-4 w-4 text-green" />
                      ) : (
                        <XCircle className="h-4 w-4 text-red" />
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
            <p className="mt-3 text-xs text-faint">
              Observed lift = outreach group activation rate − control group activation rate.
              Model accuracy is how far the prediction was from reality (±3pp is good).
            </p>
          </div>
        )}
      </Card>
    </div>
  );
}
