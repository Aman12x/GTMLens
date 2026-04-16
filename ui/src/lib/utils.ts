import { type ClassValue, clsx } from "clsx";
import { twMerge } from "tailwind-merge";

export function cn(...inputs: ClassValue[]): string {
  return twMerge(clsx(inputs));
}

export function fmt(n: number, decimals = 1): string {
  return n.toFixed(decimals);
}

export function pct(n: number, decimals = 1): string {
  return `${(n * 100).toFixed(decimals)}%`;
}

export function fmtPct(n: number, decimals = 1): string {
  const sign = n >= 0 ? "+" : "";
  return `${sign}${(n * 100).toFixed(decimals)}pp`;
}

export function fmtN(n: number): string {
  return n.toLocaleString();
}

export function sigLevel(p: number): "significant" | "marginal" | "ns" {
  if (p < 0.05) return "significant";
  if (p < 0.10) return "marginal";
  return "ns";
}
