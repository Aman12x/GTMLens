import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { Filter, RefreshCw, TrendingUp, Users } from "lucide-react";
import { api, extractError } from "@/lib/api";
import { fmtN, fmtPct, pct, fmt } from "@/lib/utils";
import { Card, CardHeader, CardTitle } from "@/components/ui/Card";
import { Button } from "@/components/ui/Button";
import { Select } from "@/components/ui/Input";
import { Badge } from "@/components/ui/Badge";
import { Alert } from "@/components/ui/Alert";
import { Stat } from "@/components/ui/Stat";
import { FunnelChart } from "@/components/charts/FunnelChart";
import { TrendChart } from "@/components/charts/TrendChart";
import type { AnalyzeRequest } from "@/types/api";

const COMPANY_SIZES = ["", "SMB", "mid_market", "enterprise"];
const CHANNELS = ["", "organic", "paid_search", "social", "referral", "email"];

export function FunnelPage() {
  const [filters, setFilters] = useState<AnalyzeRequest>({});

  const { data, isLoading, isError, error, refetch, isFetching } = useQuery({
    queryKey: ["analyze", filters],
    queryFn: () => api.analyze(filters),
    staleTime: 30_000,
  });

  const cuped = data?.cuped;
  const srm = data?.srm;

  return (
    <div className="space-y-6">
      {/* ── Filters ─────────────────────────────────────────── */}
      <Card>
        <CardHeader>
          <CardTitle>
            <Filter className="mr-1.5 inline h-3.5 w-3.5" />
            Segment filters
          </CardTitle>
          <Button
            variant="ghost"
            size="sm"
            onClick={() => refetch()}
            loading={isFetching}
          >
            <RefreshCw className="h-3.5 w-3.5" />
            Refresh
          </Button>
        </CardHeader>
        <div className="flex flex-wrap gap-4">
          <Select
            label="Company size"
            value={filters.company_size ?? ""}
            onChange={(e) =>
              setFilters((f) => ({ ...f, company_size: e.target.value || undefined }))
            }
            className="w-40"
          >
            {COMPANY_SIZES.map((s) => (
              <option key={s} value={s}>
                {s || "All sizes"}
              </option>
            ))}
          </Select>
          <Select
            label="Channel"
            value={filters.channel ?? ""}
            onChange={(e) =>
              setFilters((f) => ({ ...f, channel: e.target.value || undefined }))
            }
            className="w-44"
          >
            {CHANNELS.map((c) => (
              <option key={c} value={c}>
                {c || "All channels"}
              </option>
            ))}
          </Select>
          {Object.keys(filters).length > 0 && (
            <div className="flex items-end">
              <Button
                variant="ghost"
                size="sm"
                onClick={() => setFilters({})}
              >
                Clear filters
              </Button>
            </div>
          )}
        </div>
      </Card>

      {isError && (
        <Alert variant="danger" title="Failed to load analysis">
          {extractError(error)}
        </Alert>
      )}

      {isLoading && (
        <div className="flex h-48 items-center justify-center text-muted">
          <svg className="mr-2 h-5 w-5 animate-spin" fill="none" viewBox="0 0 24 24">
            <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
            <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
          </svg>
          Running analysis…
        </div>
      )}

      {data && (
        <>
          {/* ── SRM Warning ─────────────────────────────────── */}
          {srm?.srm_detected && (
            <Alert variant="warning" title="Uneven treatment split detected">
              {pct(srm.observed_split)} of users received outreach — expected 50%.{" "}
              Activation estimates may be unreliable. Check that your data was split randomly.
            </Alert>
          )}

          {/* ── Summary stats ───────────────────────────────── */}
          <div className="grid grid-cols-2 gap-4 md:grid-cols-4">
            <Card>
              <Stat
                label="Total users"
                value={fmtN(data.total_users)}
              />
            </Card>
            <Card>
              <Stat
                label="Activation lift"
                value={cuped ? fmtPct(cuped.ate) : "—"}
                sub={
                  cuped
                    ? `p=${fmt(cuped.p_value, 4)}`
                    : "< 200 sign-ups"
                }
                valueClassName={
                  cuped
                    ? cuped.ate > 0
                      ? "text-green"
                      : "text-red"
                    : undefined
                }
              />
            </Card>
            <Card>
              <Stat
                label="95% confidence range"
                value={
                  cuped
                    ? `[${fmtPct(cuped.ci_lower)}, ${fmtPct(cuped.ci_upper)}]`
                    : "—"
                }
                valueClassName="text-base"
              />
            </Card>
            <Card>
              <Stat
                label="Estimate precision"
                value={cuped ? `${fmt(cuped.variance_reduction_pct)}% better` : "—"}
                sub={cuped ? "vs. unadjusted baseline" : undefined}
                valueClassName="text-accent"
              />
            </Card>
          </div>

          {/* ── Causal estimate detail ──────────────────────── */}
          {cuped && (
            <Card>
              <CardHeader>
                <CardTitle>
                  <TrendingUp className="mr-1.5 inline h-3.5 w-3.5" />
                  Outreach impact estimate
                </CardTitle>
                <div className="flex items-center gap-2">
                  {cuped.p_value < 0.05 ? (
                    <Badge variant="success">Statistically significant</Badge>
                  ) : cuped.p_value < 0.10 ? (
                    <Badge variant="warning">Borderline</Badge>
                  ) : (
                    <Badge variant="muted">Not significant</Badge>
                  )}
                  {!srm?.srm_detected && (
                    <Badge variant="success">Groups balanced</Badge>
                  )}
                </div>
              </CardHeader>
              <div className="grid grid-cols-3 gap-6 md:grid-cols-6">
                <Stat label="Outreach group" value={fmtN(cuped.n_treatment)} />
                <Stat label="Control group" value={fmtN(cuped.n_control)} />
                <Stat label="Lift" value={fmtPct(cuped.ate)} valueClassName="text-green" />
                <Stat label="Margin of error" value={`±${fmtPct(cuped.ate_se * 1.96)}`} />
                <Stat label="Low end" value={fmtPct(cuped.ci_lower)} />
                <Stat label="High end" value={fmtPct(cuped.ci_upper)} />
              </div>
            </Card>
          )}

          {/* ── Funnel ──────────────────────────────────────── */}
          <Card>
            <CardHeader>
              <CardTitle>
                <Users className="mr-1.5 inline h-3.5 w-3.5" />
                Funnel breakdown
              </CardTitle>
              {Object.keys(data.filters_applied).length > 0 && (
                <div className="flex flex-wrap gap-1.5">
                  {Object.entries(data.filters_applied).map(([k, v]) => (
                    <Badge key={k} variant="muted">
                      {k}: {v}
                    </Badge>
                  ))}
                </div>
              )}
            </CardHeader>
            <FunnelChart stages={data.funnel} />
          </Card>

          {/* ── Daily trend ─────────────────────────────────── */}
          {data.daily_trend.length > 0 && (
            <Card>
              <CardHeader>
                <CardTitle>Daily activation rate — treatment vs control</CardTitle>
              </CardHeader>
              <TrendChart data={data.daily_trend} />
            </Card>
          )}
        </>
      )}
    </div>
  );
}
