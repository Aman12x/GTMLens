import { cn } from "@/lib/utils";
import type { ReactNode } from "react";

interface StatProps {
  label: string;
  value: ReactNode;
  sub?: ReactNode;
  className?: string;
  valueClassName?: string;
}

export function Stat({ label, value, sub, className, valueClassName }: StatProps) {
  return (
    <div className={cn("flex flex-col gap-0.5", className)}>
      <p className="text-xs font-medium uppercase tracking-wider text-faint">{label}</p>
      <p className={cn("text-2xl font-semibold tabular-nums text-text", valueClassName)}>
        {value}
      </p>
      {sub && <p className="text-xs text-muted">{sub}</p>}
    </div>
  );
}
