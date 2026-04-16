import { fmtN, pct } from "@/lib/utils";
import type { FunnelStage } from "@/types/api";

const STAGE_COLORS: Record<string, string> = {
  impression: "#58a6ff",
  click:      "#79c0ff",
  signup:     "#56d364",
  activation: "#3fb950",
  conversion: "#26a641",
};

interface FunnelChartProps {
  stages: FunnelStage[];
}

export function FunnelChart({ stages }: FunnelChartProps) {
  const maxN = stages[0]?.n ?? 1;

  return (
    <div className="space-y-2">
      {stages.map((stage) => {
        const widthPct = (stage.n / maxN) * 100;
        const color = STAGE_COLORS[stage.stage] ?? "#58a6ff";

        return (
          <div key={stage.stage} className="group">
            <div className="mb-1 flex items-center justify-between text-xs">
              <span className="font-medium capitalize text-text">{stage.stage}</span>
              <div className="flex items-center gap-3 text-muted">
                <span className="font-mono">{fmtN(stage.n)}</span>
                {stage.rate_from_prev !== null && (
                  <span className="text-faint">
                    {pct(stage.rate_from_prev)} from prev
                  </span>
                )}
              </div>
            </div>
            <div className="relative h-8 overflow-hidden rounded bg-overlay">
              <div
                className="h-full rounded transition-all duration-500"
                style={{ width: `${widthPct}%`, backgroundColor: color, opacity: 0.75 }}
              />
              <div
                className="absolute inset-0 flex items-center px-3"
                style={{ width: `${widthPct}%` }}
              >
                <span className="text-xs font-medium text-bg">
                  {pct(stage.rate_from_impression)} of total
                </span>
              </div>
            </div>
          </div>
        );
      })}
    </div>
  );
}
