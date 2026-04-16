import { cn } from "@/lib/utils";
import type { InputHTMLAttributes } from "react";

interface InputProps extends InputHTMLAttributes<HTMLInputElement> {
  label?: string;
  hint?: string;
  error?: string;
}

export function Input({ label, hint, error, className, id, ...props }: InputProps) {
  const inputId = id ?? label?.toLowerCase().replace(/\s+/g, "-");
  return (
    <div className="flex flex-col gap-1.5">
      {label && (
        <label htmlFor={inputId} className="text-xs font-medium text-muted">
          {label}
        </label>
      )}
      <input
        id={inputId}
        className={cn(
          "h-9 w-full rounded border border-border bg-overlay px-3 text-sm text-text",
          "placeholder:text-faint focus:border-accent focus:outline-none focus:ring-1 focus:ring-accent/30",
          "disabled:cursor-not-allowed disabled:opacity-40",
          error && "border-red focus:border-red focus:ring-red/20",
          className,
        )}
        {...props}
      />
      {hint && !error && <p className="text-xs text-faint">{hint}</p>}
      {error && <p className="text-xs text-red">{error}</p>}
    </div>
  );
}

interface SelectProps extends React.SelectHTMLAttributes<HTMLSelectElement> {
  label?: string;
  hint?: string;
  children: React.ReactNode;
}

export function Select({ label, hint, className, id, children, ...props }: SelectProps) {
  const selectId = id ?? label?.toLowerCase().replace(/\s+/g, "-");
  return (
    <div className="flex flex-col gap-1.5">
      {label && (
        <label htmlFor={selectId} className="text-xs font-medium text-muted">
          {label}
        </label>
      )}
      <select
        id={selectId}
        className={cn(
          "h-9 w-full rounded border border-border bg-overlay px-3 text-sm text-text",
          "focus:border-accent focus:outline-none focus:ring-1 focus:ring-accent/30",
          "disabled:cursor-not-allowed disabled:opacity-40",
          className,
        )}
        {...props}
      >
        {children}
      </select>
      {hint && <p className="text-xs text-faint">{hint}</p>}
    </div>
  );
}
