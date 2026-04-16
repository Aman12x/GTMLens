import { AlertTriangle, CheckCircle2, Info, XCircle } from "lucide-react";
import { cn } from "@/lib/utils";
import type { ReactNode } from "react";

type AlertVariant = "info" | "success" | "warning" | "danger";

const config: Record<AlertVariant, { icon: typeof Info; styles: string }> = {
  info:    { icon: Info,           styles: "border-accent/30 bg-accent/5 text-accent" },
  success: { icon: CheckCircle2,   styles: "border-green/30 bg-green/5 text-green" },
  warning: { icon: AlertTriangle,  styles: "border-yellow/30 bg-yellow/5 text-yellow" },
  danger:  { icon: XCircle,        styles: "border-red/30 bg-red/5 text-red" },
};

interface AlertProps {
  variant?: AlertVariant;
  title?: string;
  className?: string;
  children: ReactNode;
}

export function Alert({ variant = "info", title, className, children }: AlertProps) {
  const { icon: Icon, styles } = config[variant];
  return (
    <div className={cn("flex gap-3 rounded-lg border p-4", styles, className)}>
      <Icon className="mt-0.5 h-4 w-4 shrink-0" />
      <div className="text-sm">
        {title && <p className="mb-1 font-medium">{title}</p>}
        <div className="opacity-80">{children}</div>
      </div>
    </div>
  );
}
