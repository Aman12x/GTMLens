# GTMLens

Causal targeting engine for GTM funnels. Identifies which customer segments respond to outreach, generates personalised messages, and measures real campaign lift — not just open rates.

**Stack:** Python 3.12 · FastAPI · React · EconML · Claude API · DuckDB · Resend

---

## What it does

Most GTM tools tell you what happened. GTMLens tells you *why* — and who to target next.

1. **Upload your funnel data** — treatment/control split, activation outcome, segment labels
2. **T-Learner CATE estimation** — identifies heterogeneous treatment effects across (company_size × channel) segments with BH-corrected significance
3. **Claude-generated outreach** — segment-conditional emails, only for segments above the uplift threshold
4. **Causal lift measurement** — 20% holdout auto-assigned per segment; Results tab shows treatment rate minus holdout rate once activations are imported

---

## CATE accuracy on demo scenario

The demo dataset has **known ground truth** (heterogeneous effects by design). This table evaluates how close the T-Learner gets.

| Segment | True ATE | T-Learner estimate | Error |
|---|---|---|---|
| enterprise × paid_search | +18.0pp | +15.7pp | −2.3pp |
| enterprise × organic | +14.0pp | +14.1pp | +0.1pp |
| enterprise × referral | +12.0pp | +12.6pp | +0.6pp |
| enterprise × email | +10.0pp | +10.2pp | +0.2pp |
| enterprise × social | +6.0pp | +3.8pp | −2.2pp |
| mid_market × paid_search | +11.0pp | +7.8pp | −3.2pp |
| mid_market × organic | +9.0pp | +6.8pp | −2.2pp |
| mid_market × referral | +8.0pp | +4.2pp | −3.8pp |
| mid_market × email | +7.0pp | +4.3pp | −2.7pp |
| mid_market × social | +4.0pp | +8.9pp | +4.9pp |
| SMB × paid_search | +4.0pp | +1.6pp | −2.4pp |
| SMB × organic | +3.0pp | +2.3pp | −0.7pp |
| SMB × referral | +3.0pp | +1.8pp | −1.2pp |
| SMB × email | +2.0pp | +4.1pp | +2.1pp |
| SMB × social | +1.0pp | +2.4pp | +1.4pp |
| **Weighted aggregate** | **+7.0pp** | **+6.3pp** | **−0.7pp** |

12/15 segments within ±3pp. The three headline segments (enterprise × paid_search, mid_market × organic, SMB × organic) all pass the ±3pp threshold.

---

## Architecture

```
React SPA (Vite)
├── Funnel · Segments · Outreach · Results
│
└── HTTP + Bearer JWT
        │
    FastAPI
    ├── /analyze          CUPED ATE · SRM detection · funnel
    ├── /segment/cate     T-Learner · BH correction
    ├── /outreach/*       generate · send-segment · lift
    ├── /contacts/*       upload · activate
    ├── /data/*           upload funnel CSV · status
    └── /auth             login · register
        │
    core/
    ├── causal.py         CUPED · CATE · SRM · BH correction
    ├── preprocess.py     winsorize · log transform
    ├── outreach.py       CATE threshold guard · holdout hash
    ├── email_sender.py   Resend · CAN-SPAM footer
    └── narrative.py      LLM result explanation
        │
    data/
    ├── gtmlens.duckdb          demo tenant (shared, synthetic)
    ├── tenants/{sha256}.duckdb per-user upload (isolated)
    ├── logs.db                 contacts · sends · outreach log
    └── auth.db                 user accounts
```

**Tenant isolation:** unauthenticated requests route to the shared demo DuckDB. Authenticated requests route to a per-user DuckDB keyed by `sha256(email)[:24]`. SQLite tables carry a `tenant_id` column indexed on it.

---

## Statistical methods

### CUPED
Variance reduction using pre-experiment activation rate as covariate. Applied before ATE estimation.

- Outcome: binary `activated` flag — not winsorized (binary outcomes have no outliers)
- Covariate: `pre_activation_rate` — winsorized at 99th percentile
- Falls back gracefully when pre-period data is unavailable

### T-Learner CATE
Separate response surface models for treatment and control arms (scikit-learn `GradientBoostingRegressor`, hand-rolled T-Learner — EconML is used for the optional CausalForest method, gated at N > 5000 per arm). Per-user estimates aggregated to segment level.

- Log-transform on continuous features before fitting (offset = 1.0)
- Segment-level significance: Welch's t-test on binary activation within each arm
- Multiple testing: Benjamini-Hochberg FDR correction across all segments
- Uncertainty: opt-in bootstrap 95% CIs on segment mean CATE (`include_ci` on
  `/api/segment/cate`) — stratified-by-arm resampling with a full learner refit
  per replicate, so the interval reflects model-fit variability, not just the
  variance of averaging fixed predictions

### Qini curve (targeting validation)
`POST /api/segment/qini` checks whether the CATE ranking actually orders users by
incremental response. Scores are **cross-fitted** (5-fold; every user is scored by a
model that never saw them) so the curve reflects out-of-sample ranking quality.
On the demo dataset: Qini coefficient +49.3, with the top 20% of users by predicted
CATE capturing 34% of total incremental activations (1.7× random targeting) and the
top 30% capturing 43%.

### SRM detection
Chi-square test at α = 0.01. Run before reporting any ATE. Surfaced as a warning in the UI.

### Holdout assignment
`hash(segment_id + email) % 100 < holdout_pct * 100` — deterministic, no state required, same contact always lands in the same bucket for a given segment.

---

## Tests

```bash
pytest tests/ -v --tb=short
```

| Module | Tests |
|---|---|
| `test_preprocess.py` | Winsorize clips correctly; log handles zeros; no input mutation |
| `test_causal.py` | CUPED recovers known ATE; SRM detects 60/40 split at p<0.01; BH rejects fewer than Bonferroni |
| `test_experiment.py` | Power calc matches scipy reference; CUPED adjustment reduces N |
| `test_outreach.py` | Only high-uplift segments get messages; holdout fraction correct; JSON parse succeeds |
| `test_auth.py` | Register · login · JWT validation · 401 on tampered token |

---

## Using GTMLens with Clay

GTMLens plugs into a [Clay](https://clay.com) table as the **causal scoring brain**: Clay
sources and enriches prospects and pushes to your sequencer; GTMLens decides who is worth
contacting (uplift, not fit score), keeps the holdout honest, and measures real lift.

### 1. Mint an API key

```bash
# register + login, then:
curl -X POST https://<your-deploy>/api/alpha/auth/api-key \
  -H "Authorization: Bearer $JWT" \
  -H "Content-Type: application/json" \
  -d '{"label": "clay-outbound-table"}'
# → {"api_key": "gtml_…", …}   shown once — store it in Clay
```

### 2. Add an HTTP API enrichment column in Clay

- **Method** `POST` · **URL** `https://<your-deploy>/api/score`
- **Headers** `X-API-Key: gtml_…` · `Content-Type: application/json`
- **Body** (map Clay columns into the template):

```json
{
  "email": "{{Email}}",
  "employee_count": {{Employee Count}},
  "industry": "{{Industry}}",
  "channel": "{{Source}}"
}
```

Raw values are normalized server-side: employee counts and ranges ("51-200") bucket into
SMB / mid_market / enterprise; source labels ("Google Ads", "LinkedIn", "SEO") map onto
paid_search / social / organic / referral / email.

- **Response fields → Clay columns**: `assignment`, `holdout_flag`, `uplift_tier`,
  `cate_estimate`, `message_angle`, `model_version`.

### 3. Route rows with Clay filters

| Filter | Action |
|---|---|
| `holdout_flag = true` | **Suppress — never send.** These contacts measure lift. |
| `uplift_tier = high` | Push to your primary sequence |
| `uplift_tier = mid` | Standard sequence |
| `uplift_tier = deprioritize` | Low-cost nurture or skip |

Feed `message_angle` into Clay's AI messaging column prompt — GTMLens supplies *why* this
segment responds; Clay writes the copy.

### 4. Close the loop

1. **Wave 0 (cold start)** — no model yet: `/api/score` returns a randomized
   `assignment` (`treat`/`control`). Send to `treat` only. The first wave *is* the experiment.
2. Import outcomes (replies / activations) via `/api/contacts/upload` + `/api/contacts/activate`
   or the Data tab, then `POST /api/score/train`.
3. **Wave 1+** — `/api/score` now returns real CATE estimates and tiers. Keep the holdout
   filter on; the Results tab shows treatment-minus-holdout lift per segment.
4. `GET /api/score/srm` audits the loop: chi-square on the realized assignment split
   (α = 0.01) and a **holdout-violation count** — control-assigned contacts that were sent
   anyway, i.e. a misconfigured Clay filter contaminating your experiment.

Scoring is pure model inference (no LLM in the hot path), deterministic per contact, and
idempotent — Clay retries and column re-runs are safe.

---

## Environment variables

| Variable | Required | Description |
|---|---|---|
| `ANTHROPIC_API_KEY` | yes | Claude API key |
| `JWT_SECRET_KEY` | yes | Token signing secret |
| `RESEND_API_KEY` | for email | Resend API key |
| `RESEND_FROM_EMAIL` | for email | Verified sender address |
| `PHYSICAL_ADDRESS` | for email | CAN-SPAM §7(a)(5)(A) mailing address |
| `DATABASE_URL` | no | DuckDB path (default `./data/gtmlens.duckdb`) |
| `SQLITE_PATH` | no | SQLite path (default `./data/logs.db`) |
| `MODELS_DIR` | no | Persisted scoring models (default `./data/models`) |
| `CATE_UPLIFT_THRESHOLD` | no | Top fraction of segments to target (default `0.40`) |
| `HOLDOUT_FRACTION` | no | Fraction of each segment held out (default `0.20`) |
| `COLD_START_CONTROL_FRACTION` | no | Control share in wave-0 randomization (default `0.50`) |
| `SCORE_DEPRIORITIZE_FRACTION` | no | Bottom CATE fraction tiered `deprioritize` (default `0.30`) |

Copy `.env.example` to `.env` and fill in the required values before running.

---

