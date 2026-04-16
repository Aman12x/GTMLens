import { Component, type ErrorInfo, type ReactNode } from "react";

interface Props {
  children: ReactNode;
  fallback?: ReactNode;
}

interface State {
  hasError: boolean;
  error: Error | null;
}

/**
 * React class-based error boundary.
 *
 * React hooks cannot catch render errors — a class component is required.
 * Without this, an exception in Recharts or any child crashes the entire app
 * to a blank screen with no recovery path.
 *
 * Usage:
 *   <ErrorBoundary>
 *     <ComponentThatMightThrow />
 *   </ErrorBoundary>
 */
export class ErrorBoundary extends Component<Props, State> {
  constructor(props: Props) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error: Error): State {
    return { hasError: true, error };
  }

  componentDidCatch(error: Error, info: ErrorInfo): void {
    console.error("[ErrorBoundary]", error, info.componentStack);
  }

  handleReset = (): void => {
    this.setState({ hasError: false, error: null });
  };

  render(): ReactNode {
    if (this.state.hasError) {
      if (this.props.fallback) return this.props.fallback;

      return (
        <div className="flex min-h-[200px] flex-col items-center justify-center gap-4 rounded-lg border border-red/30 bg-red/5 p-8 text-center">
          <p className="text-sm font-medium text-red">Something went wrong</p>
          <p className="max-w-md text-xs text-muted">
            {this.state.error?.message ?? "An unexpected render error occurred."}
          </p>
          <button
            onClick={this.handleReset}
            className="rounded border border-border bg-overlay px-4 py-1.5 text-xs text-text hover:border-faint"
          >
            Try again
          </button>
        </div>
      );
    }

    return this.props.children;
  }
}
