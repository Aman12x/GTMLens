# GTMLens

Causal targeting engine for GTM funnels. Identifies which customer segments respond to outreach, generates personalised messages, and measures real campaign lift — not just open rates.

**Stack:** Python 3.12 · FastAPI · React · EconML · Claude API · DuckDB · Resend · Railway

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
| enterprise × paid_search | +18.0pp | *(run demo)* | — |
| enterprise × organic | +14.0pp | *(run demo)* | — |
| enterprise × referral | +12.0pp | *(run demo)* | — |
| enterprise × email | +10.0pp | *(run demo)* | — |
| enterprise × social | +6.0pp | *(run demo)* | — |
| mid_market × paid_search | +11.0pp | *(run demo)* | — |
| mid_market × organic | +9.0pp | *(run demo)* | — |
| mid_market × referral | +8.0pp | *(run demo)* | — |
| mid_market × email | +7.0pp | *(run demo)* | — |
| mid_market × social | +4.0pp | *(run demo)* | — |
| SMB × paid_search | +4.0pp | *(run demo)* | — |
| SMB × organic | +3.0pp | *(run demo)* | — |
| SMB × referral | +3.0pp | *(run demo)* | — |
| SMB × email | +2.0pp | *(run demo)* | — |
| SMB × social | +1.0pp | *(run demo)* | — |
| **Weighted aggregate** | **+7.0pp** | *(run demo)* | — |

To populate the estimate column: spin up the app, hit the Segments tab, and read off the Mean CATE values. Success criterion: estimates within ±3pp of ground truth on the three headline segments (enterprise × paid_search, mid_market × organic, SMB × organic).

Ground truth source: `data/ground_truth.py`. Synthetic data generator: `data/synthetic.py`. These are the only files that should know the true effect sizes.

---

## Architecture

```
React SPA (Vite)
├── Auth · Data · Segments · Outreach · Results
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
        │
    external
    ├── Anthropic Claude        outreach generation · narrative
    └── Resend                  email delivery
```

**Tenant isolation:** unauthenticated requests route to the shared demo DuckDB. Authenticated requests route to a per-user DuckDB keyed by `sha256(email)[:24]`. SQLite tables carry a `tenant_id` column and are indexed on it.

---

## Statistical methods

### CUPED
Variance reduction using pre-experiment activation rate as covariate. Applied before ATE estimation on the Segments tab.

- Outcome: binary `activated` flag — **not winsorized** (binary outcomes have no outliers)
- Covariate: `pre_activation_rate` — **winsorized at 99th percentile**
- Reports variance reduction % alongside the adjusted ATE
- Falls back gracefully when pre-period data is unavailable (constant covariate → 0% variance reduction, which is correct)

### T-Learner CATE
Separate response surface models for treatment and control arms (EconML `BaseTRegressor` with `RandomForestRegressor`). Per-user estimates aggregated to segment level.

- Log-transform on continuous features before fitting (offset = 1.0)
- Segment-level significance: Welch's t-test on binary activation within each arm
- Multiple testing: Benjamini-Hochberg FDR correction across all segments (never Bonferroni — too conservative; BH controls false discovery rate directly)
- `recommended_for_outreach` flag requires both BH significance **and** CATE above threshold

### SRM detection
Chi-square test at α = 0.01 (more conservative than the experiment α). Run before reporting any ATE. Surfaces a warning banner in the UI — never suppressed.

### Holdout assignment
`hash(segment_id + email) % 100 < holdout_pct * 100` — deterministic, no state required, same contact always lands in the same bucket for a given segment.

---

## Local setup

```bash
# 1. Clone and install Python deps
git clone <repo>
cd GTM
pip install -e .

# 2. Install frontend deps
cd ui && npm install && cd ..

# 3. Configure environment
cp .env.example .env
# Fill in: ANTHROPIC_API_KEY, JWT_SECRET_KEY
# Optional: RESEND_API_KEY, RESEND_FROM_EMAIL (email delivery)

# 4. Seed demo data
python data/seed_db.py

# 5. Run tests (must pass before starting API)
pytest tests/ -v --tb=short

# 6. Start backend
uvicorn api.main:app --reload

# 7. Start frontend (separate terminal)
cd ui && npm run dev
# → http://localhost:5173
```

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

## Deploy to Railway

```bash
# 1. Push to GitHub
# 2. Create Railway project → Deploy from GitHub repo
# 3. Set environment variables in Railway dashboard:
#      ANTHROPIC_API_KEY
#      JWT_SECRET_KEY        (python -c "import secrets; print(secrets.token_hex(32))")
#      RESEND_API_KEY
#      RESEND_FROM_EMAIL
#      PHYSICAL_ADDRESS      (required for CAN-SPAM compliance)
# 4. Add a Volume mounted at /app/data
#    (without this, DuckDB and SQLite files are lost on redeploy)
# 5. Railway runs: nixpacks build → npm run build → uvicorn
```

The built React app is served from FastAPI as static files — single process, single port.

---

## Environment variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `ANTHROPIC_API_KEY` | yes | — | Claude API key |
| `JWT_SECRET_KEY` | yes | — | Token signing secret — generate with `secrets.token_hex(32)` |
| `RESEND_API_KEY` | for email | — | Resend API key |
| `RESEND_FROM_EMAIL` | for email | — | Verified sender address |
| `PHYSICAL_ADDRESS` | for email | — | CAN-SPAM §7(a)(5)(A) mailing address |
| `DATABASE_URL` | no | `./data/gtmlens.duckdb` | Demo DuckDB path |
| `SQLITE_PATH` | no | `./data/logs.db` | SQLite path |
| `AUTH_DB_PATH` | no | `./data/auth.db` | Auth SQLite path |
| `CATE_UPLIFT_THRESHOLD` | no | `0.40` | Top fraction of segments to target |
| `HOLDOUT_FRACTION` | no | `0.20` | Fraction of each segment held out |
| `WINSORIZE_UPPER_PCT` | no | `0.99` | Winsorization cap |

---

## Interview talking points

- **CUPED:** "Winsorize at 99th percentile for the experiment estimator — preserves units, reduces variance, same approach used in production at Meta and Uber. Binary outcomes are never winsorized."
- **CATE:** "T-Learner gives separate response surfaces per arm. Segment aggregation identifies which buckets have the highest lift. BH correction controls FDR — we never use Bonferroni because it's too conservative and doesn't match how we think about false discovery in a targeting context."
- **Holdout:** "Deterministic hash of segment + email — no state required, reproducible, same contact always lands in the same bucket."
- **Outreach as treatment:** "Outreach message is modelled as a treatment variant. We measure causal lift on activation rate, not open rate. The holdout group is the counterfactual."
- **Why not send to everyone:** "CATE identifies heterogeneous effects. Sending to low-uplift segments dilutes the signal and wastes budget. Threshold filter enforces precision over volume."
