import { cn } from "@/lib/utils";
import type { ReactNode } from "react";

type Variant = "default" | "success" | "warning" | "danger" | "muted";

const variants: Record<Variant, string> = {
  default: "bg-accent/10 text-accent border-accent/20",
  success: "bg-green/10 text-green border-green/20",
  warning: "bg-yellow/10 text-yellow border-yellow/20",
  danger:  "bg-red/10 text-red border-red/20",
  muted:   "bg-faint/20 text-muted border-faint/30",
};

interface BadgeProps {
  variant?: Variant;
  className?: string;
  children: ReactNode;
}

export function Badge({ variant = "default", className, children }: BadgeProps) {
  return (
    <span
      className={cn(
        "inline-flex items-center rounded-full border px-2.5 py-0.5 text-xs font-medium",
        variants[variant],
        className,
      )}
    >
      {children}
    </span>
  );
}
