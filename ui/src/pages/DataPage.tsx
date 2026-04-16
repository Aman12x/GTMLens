import { useRef, useState } from "react";
import { useMutation, useQuery } from "@tanstack/react-query";
import { Activity, CheckCircle, Circle, Database, Download, FileUp, RefreshCw, Users } from "lucide-react";
import { api, extractError } from "@/lib/api";
import { Card, CardHeader, CardTitle } from "@/components/ui/Card";
import { Button } from "@/components/ui/Button";
import { Alert } from "@/components/ui/Alert";

// ---------------------------------------------------------------------------
// Sample CSV content — downloaded client-side, no round-trip needed
// ---------------------------------------------------------------------------

const FUNNEL_SAMPLE = `user_id,treatment,activated,company_size,channel,industry
u001,1,1,enterprise,paid_search,SaaS
u002,1,0,enterprise,paid_search,SaaS
u003,0,0,enterprise,paid_search,SaaS
u004,0,1,enterprise,paid_search,SaaS
u005,1,1,mid_market,email,Fintech
u006,1,0,mid_market,email,Fintech
u007,0,0,mid_market,email,Fintech
u008,0,0,mid_market,email,Fintech
u009,1,0,SMB,organic,Ecommerce
u010,0,1,SMB,organic,Ecommerce
`;

const CONTACTS_SAMPLE = `email,first_name,company,company_size,channel
alice@acme.com,Alice,Acme Corp,enterprise,paid_search
bob@beta.io,Bob,Beta Labs,enterprise,paid_search
carol@gamma.co,Carol,Gamma Inc,mid_market,email
dan@delta.com,Dan,Delta Ltd,SMB,organic
`;

function downloadCsv(filename: string, content: string) {
  const blob = new Blob([content], { type: "text/csv" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}

// ---------------------------------------------------------------------------
// Flow step indicator
// ---------------------------------------------------------------------------

function FlowStep({
  n,
  label,
  sub,
  done,
}: {
  n: number;
  label: string;
  sub: string;
  done: boolean;
}) {
  return (
    <div className="flex items-start gap-3">
      <div className="mt-0.5 shrink-0">
        {done ? (
          <CheckCircle className="h-5 w-5 text-green" />
        ) : (
          <Circle className="h-5 w-5 text-border" />
        )}
      </div>
      <div>
        <p className={`text-sm font-medium ${done ? "text-muted line-through" : "text-text"}`}>
          <span className="mr-1.5 font-mono text-xs text-faint">{n}.</span>
          {label}
        </p>
        <p className="text-xs text-faint">{sub}</p>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Page
// ---------------------------------------------------------------------------

export function DataPage() {
  const fileRef = useRef<HTMLInputElement>(null);
  const contactFileRef = useRef<HTMLInputElement>(null);
  const [activationText, setActivationText] = useState("");

  const statusQuery = useQuery({
    queryKey: ["data-status"],
    queryFn: api.dataStatus,
    staleTime: 30_000,
  });

  const contactCountQuery = useQuery({
    queryKey: ["contact-count"],
    queryFn: () => api.listContacts(),
    staleTime: 30_000,
  });

  const uploadMutation = useMutation({
    mutationFn: (file: File) => api.uploadData(file),
    onSuccess: () => statusQuery.refetch(),
  });

  const uploadContactsMutation = useMutation({
    mutationFn: (file: File) => api.uploadContacts(file),
    onSuccess: () => contactCountQuery.refetch(),
  });

  const activateMutation = useMutation({
    mutationFn: (emails: string[]) => api.activateContacts(emails),
    onSuccess: () => setActivationText(""),
  });

  const status = statusQuery.data;
  const hasData = status?.has_real_data ?? false;
  const hasContacts = (contactCountQuery.data?.total ?? 0) > 0;

  return (
    <div className="space-y-6">
      {/* ── Flow overview ───────────────────────────────────────── */}
      <Card>
        <CardHeader>
          <CardTitle>
            <Database className="mr-1.5 inline h-3.5 w-3.5" />
            Setup checklist
          </CardTitle>
        </CardHeader>
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
          <FlowStep
            n={1}
            label="Upload funnel data"
            sub="Your users, treatment flag, and activation outcome"
            done={hasData}
          />
          <FlowStep
            n={2}
            label="Upload contacts"
            sub="The people you'll send outreach to"
            done={hasContacts}
          />
          <FlowStep
            n={3}
            label="Generate & send outreach"
            sub="Go to the Outreach tab once steps 1–2 are done"
            done={false}
          />
          <FlowStep
            n={4}
            label="Import activation results"
            sub="Paste who converted — closes the causal loop"
            done={false}
          />
        </div>
      </Card>

      {/* ── Data source status ──────────────────────────────────── */}
      {status && (
        <Alert variant={status.is_demo ? "warning" : "success"}>
          {status.message}
          {!status.is_demo && status.has_real_data && (
            <span className="ml-2 text-faint">
              · Tenant: {status.tenant_id.split("@")[0]}
            </span>
          )}
        </Alert>
      )}

      {/* ── Funnel data upload ──────────────────────────────────── */}
      <Card>
        <CardHeader>
          <CardTitle>
            <FileUp className="mr-1.5 inline h-3.5 w-3.5" />
            Step 1 — Upload funnel data
          </CardTitle>
          <Button
            variant="ghost"
            size="sm"
            onClick={() => downloadCsv("funnel_sample.csv", FUNNEL_SAMPLE)}
          >
            <Download className="h-3.5 w-3.5" />
            Sample CSV
          </Button>
        </CardHeader>

        <div className="space-y-4">
          <div className="rounded-lg border border-border bg-overlay p-4 text-xs text-muted space-y-1.5">
            <p className="font-medium text-text">Required columns</p>
            <p>
              <span className="font-mono text-accent">user_id</span>
              {" · "}
              <span className="font-mono text-accent">treatment</span>
              <span className="text-faint"> (1 = received outreach, 0 = holdout)</span>
              {" · "}
              <span className="font-mono text-accent">activated</span>
              <span className="text-faint"> (1 = converted, 0 = did not)</span>
            </p>
            <p className="font-medium text-text mt-2">Optional columns</p>
            <p>
              <span className="font-mono text-muted">company_size</span>
              <span className="text-faint"> (SMB | mid_market | enterprise)</span>
              {" · "}
              <span className="font-mono text-muted">channel</span>
              <span className="text-faint"> (organic | paid_search | social | referral | email)</span>
              {" · "}
              <span className="font-mono text-muted">industry</span>
            </p>
            <p className="text-faint pt-1">
              Common aliases are accepted automatically — e.g. <span className="font-mono">converted</span>,{" "}
              <span className="font-mono">variant</span>, <span className="font-mono">customer_id</span>.
              Download the sample CSV to see the exact format.
            </p>
          </div>

          {uploadMutation.isError && (
            <Alert variant="danger">{extractError(uploadMutation.error)}</Alert>
          )}
          {uploadMutation.data && (
            <Alert variant="success">{uploadMutation.data.message}</Alert>
          )}

          <input
            ref={fileRef}
            type="file"
            accept=".csv"
            className="hidden"
            onChange={(e) => {
              const file = e.target.files?.[0];
              if (file) uploadMutation.mutate(file);
              e.target.value = "";
            }}
          />
          <Button
            onClick={() => fileRef.current?.click()}
            loading={uploadMutation.isPending}
            className="w-full"
          >
            <FileUp className="h-4 w-4" />
            {uploadMutation.isPending ? "Uploading and seeding…" : "Upload funnel CSV"}
          </Button>
          <p className="text-center text-xs text-faint">
            Max 500,000 rows · UTF-8 · Replaces previously uploaded data
          </p>
        </div>
      </Card>

      {/* ── Contacts upload ─────────────────────────────────────── */}
      <Card>
        <CardHeader>
          <CardTitle>
            <Users className="mr-1.5 inline h-3.5 w-3.5" />
            Step 2 — Upload contacts
            {hasContacts && (
              <span className="ml-2 text-xs font-normal text-muted">
                {contactCountQuery.data!.total.toLocaleString()} loaded
              </span>
            )}
          </CardTitle>
          <div className="flex items-center gap-2">
            <Button
              variant="ghost"
              size="sm"
              onClick={() => downloadCsv("contacts_sample.csv", CONTACTS_SAMPLE)}
            >
              <Download className="h-3.5 w-3.5" />
              Sample CSV
            </Button>
            <Button
              variant="ghost"
              size="sm"
              onClick={() => contactCountQuery.refetch()}
              loading={contactCountQuery.isFetching}
            >
              <RefreshCw className="h-3.5 w-3.5" />
            </Button>
          </div>
        </CardHeader>

        <div className="space-y-4">
          <div className="rounded-lg border border-border bg-overlay p-4 text-xs text-muted space-y-1.5">
            <p className="font-medium text-text">Required columns</p>
            <p>
              <span className="font-mono text-accent">email</span>
            </p>
            <p className="font-medium text-text mt-2">Optional columns</p>
            <p>
              <span className="font-mono text-muted">first_name</span>
              {" · "}
              <span className="font-mono text-muted">company</span>
              {" · "}
              <span className="font-mono text-muted">company_size</span>
              {" · "}
              <span className="font-mono text-muted">channel</span>
            </p>
            <p className="text-faint pt-1">
              <span className="font-mono">company_size</span> and{" "}
              <span className="font-mono">channel</span> must match your funnel data exactly —
              this is how GTMLens finds the right contacts for each segment when sending outreach.
            </p>
          </div>

          {uploadContactsMutation.isError && (
            <Alert variant="danger">{extractError(uploadContactsMutation.error)}</Alert>
          )}
          {uploadContactsMutation.data && (
            <Alert variant="success">
              {uploadContactsMutation.data.inserted} new
              {uploadContactsMutation.data.updated > 0 && `, ${uploadContactsMutation.data.updated} updated`}
              {uploadContactsMutation.data.skipped > 0 && `, ${uploadContactsMutation.data.skipped} skipped`}
              {" "}contacts loaded.
              {uploadContactsMutation.data.errors.length > 0 && (
                <ul className="mt-1 list-disc pl-4 text-faint">
                  {uploadContactsMutation.data.errors.slice(0, 5).map((e, i) => (
                    <li key={i}>{e}</li>
                  ))}
                </ul>
              )}
            </Alert>
          )}

          <input
            ref={contactFileRef}
            type="file"
            accept=".csv"
            className="hidden"
            onChange={(e) => {
              const file = e.target.files?.[0];
              if (file) uploadContactsMutation.mutate(file);
              e.target.value = "";
            }}
          />
          <Button
            onClick={() => contactFileRef.current?.click()}
            loading={uploadContactsMutation.isPending}
            className="w-full"
          >
            <FileUp className="h-4 w-4" />
            {uploadContactsMutation.isPending ? "Uploading contacts…" : "Upload contacts CSV"}
          </Button>
          <p className="text-center text-xs text-faint">
            Max 10,000 contacts · UTF-8 · Re-upload to update existing contacts
          </p>
        </div>
      </Card>

      {/* ── Activation import ───────────────────────────────────── */}
      <Card>
        <CardHeader>
          <CardTitle>
            <Activity className="mr-1.5 inline h-3.5 w-3.5" />
            Step 4 — Import activation results
          </CardTitle>
        </CardHeader>
        <div className="space-y-4">
          <p className="text-xs text-muted">
            After your campaign runs, paste the emails of contacts who activated. The Results tab
            will switch from the historical baseline to real campaign lift — this is the payoff.
          </p>
          <p className="text-xs text-faint">
            Holdout contacts are excluded automatically. Only contacts who were actually sent
            outreach can be marked as activated, keeping the lift estimate unbiased.
          </p>

          {activateMutation.isError && (
            <Alert variant="danger">{extractError(activateMutation.error)}</Alert>
          )}
          {activateMutation.data && (
            <Alert variant="success">
              Marked {activateMutation.data.updated} contact
              {activateMutation.data.updated !== 1 ? "s" : ""} as activated.
              {activateMutation.data.not_found.length > 0 && (
                <span className="text-faint">
                  {" "}({activateMutation.data.not_found.length} not found — were they sent outreach?)
                </span>
              )}
            </Alert>
          )}

          <textarea
            className="w-full resize-none rounded border border-border bg-overlay px-3 py-2 text-xs
              font-mono text-text placeholder:text-faint focus:border-accent focus:outline-none
              focus:ring-1 focus:ring-accent/30"
            rows={5}
            placeholder={"alice@acme.com\nbob@beta.io\ncarol@gamma.co"}
            value={activationText}
            onChange={(e) => setActivationText(e.target.value)}
          />
          <Button
            onClick={() => {
              const emails = activationText
                .split(/[\n,]+/)
                .map((e) => e.trim().toLowerCase())
                .filter((e) => e.includes("@"));
              if (emails.length > 0) activateMutation.mutate(emails);
            }}
            loading={activateMutation.isPending}
            disabled={!activationText.trim()}
            className="w-full"
          >
            <Activity className="h-4 w-4" />
            {activateMutation.isPending ? "Importing…" : "Mark as activated"}
          </Button>
          <p className="text-center text-xs text-faint">
            One email per line or comma-separated
          </p>
        </div>
      </Card>
    </div>
  );
}
