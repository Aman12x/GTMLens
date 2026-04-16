import {
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
} from "recharts";
import type { DailyTrendPoint } from "@/types/api";

interface TrendChartProps {
  data: DailyTrendPoint[];
}

function formatDate(dateStr: string): string {
  return new Date(dateStr).toLocaleDateString("en-US", { month: "short", day: "numeric" });
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
function CustomTooltip({ active, payload, label }: any) {
  if (!active || !payload?.length) return null;
  return (
    <div className="rounded-lg border border-border bg-surface p-3 text-xs shadow-lg">
      <p className="mb-2 font-medium text-text">{formatDate(label)}</p>
      {/* eslint-disable-next-line @typescript-eslint/no-explicit-any */}
      {payload.map((entry: any) => (
        <div key={entry.name} className="flex items-center justify-between gap-6">
          <span style={{ color: entry.color }}>{entry.name}</span>
          <span className="font-mono text-text">
            {(entry.value * 100).toFixed(1)}%
          </span>
        </div>
      ))}
    </div>
  );
}

export function TrendChart({ data }: TrendChartProps) {
  return (
    <ResponsiveContainer width="100%" height={220}>
      <LineChart data={data} margin={{ top: 4, right: 4, left: -20, bottom: 0 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#30363d" vertical={false} />
        <XAxis
          dataKey="date"
          tickFormatter={formatDate}
          tick={{ fontSize: 11, fill: "#8b949e" }}
          axisLine={false}
          tickLine={false}
          interval="preserveStartEnd"
        />
        <YAxis
          tickFormatter={(v) => `${(v * 100).toFixed(0)}%`}
          tick={{ fontSize: 11, fill: "#8b949e" }}
          axisLine={false}
          tickLine={false}
        />
        <Tooltip content={<CustomTooltip />} />
        <Legend
          iconType="circle"
          iconSize={8}
          wrapperStyle={{ fontSize: "12px", color: "#8b949e", paddingTop: "12px" }}
        />
        <Line
          type="monotone"
          dataKey="treatment_activation_rate"
          name="Treatment"
          stroke="#58a6ff"
          strokeWidth={2}
          dot={false}
          activeDot={{ r: 4, fill: "#58a6ff" }}
        />
        <Line
          type="monotone"
          dataKey="control_activation_rate"
          name="Control"
          stroke="#8b949e"
          strokeWidth={2}
          dot={false}
          strokeDasharray="4 2"
          activeDot={{ r: 4, fill: "#8b949e" }}
        />
      </LineChart>
    </ResponsiveContainer>
  );
}
