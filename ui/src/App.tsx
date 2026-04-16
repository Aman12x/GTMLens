import { useEffect, useState } from "react";
import { useMutation } from "@tanstack/react-query";
import { Activity, Database, FlaskConical, LogIn, LogOut, Mail, RefreshCw, RotateCcw } from "lucide-react";
import { api, extractError } from "@/lib/api";
import { clearToken, isAuthenticated } from "@/lib/auth";
import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/Button";
import { Alert } from "@/components/ui/Alert";
import { ErrorBoundary } from "@/components/ui/ErrorBoundary";
import { AuthPage } from "@/pages/AuthPage";
import { DataPage } from "@/pages/DataPage";
import { FunnelPage } from "@/pages/FunnelPage";
import { OutreachPage } from "@/pages/OutreachPage";
import { LiftPage } from "@/pages/LiftPage";

type Tab = "data" | "funnel" | "outreach" | "results";

const tabs: Array<{ id: Tab; label: string; icon: typeof Activity; authOnly?: boolean }> = [
  { id: "data",     label: "Data",      icon: Database,     authOnly: true },
  { id: "funnel",   label: "Segments",  icon: Activity },
  { id: "outreach", label: "Outreach",  icon: Mail },
  { id: "results",  label: "Results",   icon: FlaskConical },
];

export function App() {
  const [authed, setAuthed] = useState(isAuthenticated());
  // skipAuth=true means the user clicked "Try demo" — show the app unauthenticated
  const [skipAuth, setSkipAuth] = useState(false);
  const [activeTab, setActiveTab] = useState<Tab>("funnel");
  const [resetMsg, setResetMsg] = useState<string | null>(null);

  // Listen for 401 events from the axios interceptor
  useEffect(() => {
    const handle = () => { setAuthed(false); setSkipAuth(false); };
    window.addEventListener("gtmlens:logout", handle);
    return () => window.removeEventListener("gtmlens:logout", handle);
  }, []);

  const resetMutation = useMutation({
    mutationFn: api.demoReset,
    onSuccess: (d) => {
      setResetMsg(d.message);
      setTimeout(() => setResetMsg(null), 4000);
    },
  });

  function handleLogout() {
    clearToken();
    setAuthed(false);
    setSkipAuth(false);
    setActiveTab("funnel");
  }

  if (!authed && !skipAuth) {
    return (
      <AuthPage
        onAuth={() => { setAuthed(true); setActiveTab("data"); }}
        onDemo={() => setSkipAuth(true)}
      />
    );
  }

  return (
    <div className="min-h-screen bg-bg font-sans text-text antialiased">
      {/* ── Top nav ──────────────────────────────────────────── */}
      <header className="sticky top-0 z-50 border-b border-border bg-bg/95 backdrop-blur">
        <div className="mx-auto flex h-14 max-w-7xl items-center justify-between px-6">
          <div className="flex items-center gap-3">
            <div className="flex h-7 w-7 items-center justify-center rounded-md bg-accent">
              <Activity className="h-4 w-4 text-bg" />
            </div>
            <span className="font-semibold tracking-tight text-text">GTMLens</span>
            <span className="hidden text-xs text-faint sm:block">
              Causal targeting for GTM funnels
            </span>
          </div>
          <div className="flex items-center gap-2">
            {!authed && (
              <Button
                variant="secondary"
                size="sm"
                onClick={() => resetMutation.mutate()}
                loading={resetMutation.isPending}
              >
                <RotateCcw className="h-3.5 w-3.5" />
                Reset demo
              </Button>
            )}
            {authed ? (
              <Button variant="ghost" size="sm" onClick={handleLogout}>
                <LogOut className="h-3.5 w-3.5" />
              </Button>
            ) : (
              <Button variant="ghost" size="sm" onClick={() => setSkipAuth(false)}>
                <LogIn className="h-3.5 w-3.5" />
                Sign in
              </Button>
            )}
          </div>
        </div>
      </header>

      {/* ── Tab bar ──────────────────────────────────────────── */}
      <div className="border-b border-border bg-surface">
        <nav className="mx-auto flex max-w-7xl gap-0 px-6">
          {tabs.filter((t) => !t.authOnly || authed).map((tab) => {
            const Icon = tab.icon;
            const active = activeTab === tab.id;
            return (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={cn(
                  "flex h-11 items-center gap-2 border-b-2 px-4 text-sm font-medium transition-colors",
                  active
                    ? "border-accent text-text"
                    : "border-transparent text-muted hover:border-faint hover:text-text",
                )}
              >
                <Icon className="h-4 w-4" />
                <span>{tab.label}</span>
              </button>
            );
          })}
        </nav>
      </div>

      {/* ── Main content ─────────────────────────────────────── */}
      <main className="mx-auto max-w-7xl px-6 py-8">
        {resetMsg && (
          <Alert variant="success" className="mb-6">
            <RefreshCw className="mr-1.5 inline h-3.5 w-3.5" />
            {resetMsg}
          </Alert>
        )}
        {resetMutation.isError && (
          <Alert variant="danger" className="mb-6">
            Reset failed: {extractError(resetMutation.error)}
          </Alert>
        )}

        <ErrorBoundary>
          {activeTab === "data"     && <DataPage />}
          {activeTab === "funnel"   && <FunnelPage />}
          {activeTab === "outreach" && <OutreachPage />}
          {activeTab === "results"  && <LiftPage />}
        </ErrorBoundary>
      </main>

      {/* ── Footer ───────────────────────────────────────────── */}
      <footer className="mt-16 border-t border-border py-6 text-center text-xs text-faint">
        GTMLens · Causal targeting for GTM funnels
      </footer>
    </div>
  );
}
