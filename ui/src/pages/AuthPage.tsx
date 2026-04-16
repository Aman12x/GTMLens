import { useState } from "react";
import { useMutation } from "@tanstack/react-query";
import { Activity } from "lucide-react";
import { api, extractError } from "@/lib/api";
import { setToken } from "@/lib/auth";
import { Button } from "@/components/ui/Button";
import { Input } from "@/components/ui/Input";
import { Alert } from "@/components/ui/Alert";

interface Props {
  onAuth: () => void;
  onDemo: () => void;
}

export function AuthPage({ onAuth, onDemo }: Props) {
  const [mode, setMode] = useState<"login" | "register">("login");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");

  const loginMutation = useMutation({
    mutationFn: () => api.login(email, password),
    onSuccess: (data) => {
      setToken(data.access_token);
      onAuth();
    },
  });

  const registerMutation = useMutation({
    mutationFn: () => api.register(email, password),
    onSuccess: () => {
      // Auto-login after register
      loginMutation.mutate();
    },
  });

  const isPending = loginMutation.isPending || registerMutation.isPending;
  const error = loginMutation.error || registerMutation.error;

  function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (mode === "login") loginMutation.mutate();
    else registerMutation.mutate();
  }

  return (
    <div className="flex min-h-screen flex-col items-center justify-center bg-bg px-4">
      <div className="w-full max-w-sm space-y-6">
        {/* Logo */}
        <div className="flex flex-col items-center gap-3">
          <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-accent">
            <Activity className="h-5 w-5 text-bg" />
          </div>
          <div className="text-center">
            <h1 className="text-xl font-semibold text-text">GTMLens</h1>
            <p className="text-sm text-muted">Causal targeting for GTM funnels</p>
          </div>
        </div>

        {/* Tab switcher */}
        <div className="flex rounded-lg border border-border bg-surface p-1">
          {(["login", "register"] as const).map((m) => (
            <button
              key={m}
              onClick={() => setMode(m)}
              className={`flex-1 rounded-md py-1.5 text-sm font-medium capitalize transition-colors ${
                mode === m
                  ? "bg-overlay text-text shadow-sm"
                  : "text-muted hover:text-text"
              }`}
            >
              {m}
            </button>
          ))}
        </div>

        {/* Form */}
        <form onSubmit={handleSubmit} className="space-y-4">
          {error && (
            <Alert variant="danger">{extractError(error)}</Alert>
          )}
          <Input
            label="Email"
            type="email"
            autoComplete="email"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            required
          />
          <Input
            label="Password"
            type="password"
            autoComplete={mode === "login" ? "current-password" : "new-password"}
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            required
            hint={mode === "register" ? "Minimum 8 characters" : undefined}
          />
          <Button type="submit" className="w-full" loading={isPending}>
            {mode === "login" ? "Sign in" : "Create account"}
          </Button>
        </form>

        {/* Demo note */}
        <p className="text-center text-xs text-faint">
          No account?{" "}
          <button
            className="text-accent underline-offset-2 hover:underline"
            onClick={() => setMode("register")}
          >
            Register free
          </button>
          {" · "}
          <button
            className="text-accent underline-offset-2 hover:underline"
            onClick={onDemo}
          >
            Try demo
          </button>
        </p>
      </div>
    </div>
  );
}
