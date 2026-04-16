# CLAUDE.md — GTMLens

> Causal targeting engine for GTM funnels: identifies high-uplift segments, generates personalized outreach, designs and measures experiments with statistical rigor.

---

## Project Overview

**Name:** GTMLens  
**Pitch:** Most GTM tools tell you what happened. GTMLens tells you why — and who to target next.  
**Stack:** Python · FastAPI · Streamlit · Claude API · Railway  
**Timeline:** 1–2 weeks MVP  
**Portfolio signal:** Product DS (Meta/Uber IC3-4) + Applied Scientist  

---

## Claude Code Instructions

> These instructions are for Claude Code specifically. Read this section before writing any code.

### Persona
You are a senior ML engineer building a portfolio project for a Product DS / Applied Scientist job search. Code must be production-quality, well-tested, and defensible in a technical interview. Prioritize correctness over cleverness.

### Build Order
Always build in this sequence. Do not skip ahead.

```
1. data/synthetic.py          ← everything depends on this
2. data/ground_truth.py
3. data/seed_db.py
4. core/preprocess.py
5. core/causal.py
6. core/experiment.py
7. tests/ (for steps 4-6)
8. core/outreach.py
9. core/narrative.py
10. api/main.py + routes/
11. ui/app.py
12. Railway deploy config
```

Do not start a new module until the current one has passing tests.

### Code Style
- Python 3.11+
- Type hints on every function signature — no exceptions
- Docstrings on every public function: one-line summary + Args + Returns
- No `print()` for logging — use Python `logging` module with named loggers
- No hardcoded values — all thresholds/params read from environment or passed as arguments
- Max function length: 50 lines. Extract helpers if longer.
- No bare `except:` — always catch specific exceptions

### Imports
Preferred libraries — use these, do not substitute without flagging:
```
pandas, numpy          → data manipulation
duckdb                 → funnel queries
econml                 → CATE estimation (S/T-Learner, CausalForest)
scipy.stats            → power calc, chi-square SRM, BH correction
statsmodels            → DiD regression
anthropic              → Claude API (use official SDK, not raw HTTP)
fastapi, uvicorn       → API layer
streamlit              → UI
python-dotenv          → env vars
pytest                 → tests
```

Do not add new dependencies without a comment explaining why they are necessary.

### Statistical Correctness Rules
These are non-negotiable. If unsure, implement the conservative option.

**Winsorization**
- Always winsorize BEFORE computing CUPED theta
- Cap at `WINSORIZE_UPPER_PCT` from env (default 0.99)
- Never winsorize binary outcomes (reply rate, conversion flag)
- Apply same transformation to both treatment and control

**CUPED**
- Covariate must be pre-experiment period only — never leak post-experiment data
- Validate that covariate correlates with outcome (log Pearson r) — warn if r < 0.1
- Report variance reduction percentage alongside ATE

**CATE / EconML**
- Use T-Learner as default — it's the most defensible in interviews
- CausalForest is available but only run if N > 5000 per segment
- Always report confidence intervals, not just point estimates
- Log-transform continuous features before fitting (offset=1.0)

**SRM Detection**
- Run SRM check BEFORE reporting any experiment results
- If SRM detected: surface a warning in the UI, do not suppress it
- Use alpha=0.01 for SRM (more conservative than experiment alpha)

**Multiple Testing**
- Apply BH correction whenever reporting results for more than one segment simultaneously
- Never apply Bonferroni — too conservative for this use case, and explain why if asked

**DiD**
- Always check parallel trends visually (plot pre-period trends)
- Include time fixed effects in the regression

### Claude API Usage
```python
# Always use the official SDK
import anthropic
client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env

# Standard call pattern
response = client.messages.create(
    model="claude-opus-4-5",
    max_tokens=1024,
    system=SYSTEM_PROMPT,     # always pass system separately, not in messages
    messages=[{"role": "user", "content": prompt}]
)
result = response.content[0].text
```

- Always parse Claude responses as JSON where structured output is expected
- Wrap all Claude API calls in try/except `anthropic.APIError`
- Log token usage (`response.usage`) to SQLite for cost tracking
- Never expose raw API errors to the Streamlit UI — return a graceful fallback message

### DuckDB Patterns
```python
import duckdb

# Always use context manager
with duckdb.connect(os.getenv("DATABASE_URL")) as conn:
    df = conn.execute("SELECT ...").df()

# Parameterized queries only — no f-string SQL
conn.execute("SELECT * FROM events WHERE segment = ?", [segment_name])
```

### FastAPI Patterns
- Pydantic models for all request/response bodies — no raw dicts in route signatures
- Return consistent error shape: `{"error": str, "detail": str}`
- All routes must have response_model declared
- No business logic in route handlers — delegate to core/ modules

### Testing Requirements
Every module in `core/` needs tests before moving on. Minimum coverage:

| Module | Required tests |
|---|---|
| `preprocess.py` | Winsorize clips correctly; log handles zeros; no mutation of input series |
| `causal.py` | CUPED recovers known ATE on synthetic data; SRM detects 60/40 split at p<0.01; BH rejects fewer than Bonferroni |
| `experiment.py` | Power calc matches scipy reference; CUPED adjustment reduces N by expected amount |
| `outreach.py` | Only high-uplift segments get messages; holdout fraction is correct; JSON parse succeeds |
| `narrative.py` | Output contains SHIP/ITERATE/ABORT; guardrail section present; falls back gracefully on API error |

Run tests with: `pytest tests/ -v --tb=short`

### Error Handling Hierarchy
```
data/synthetic.py     → raise ValueError with clear message if params invalid
core/*.py             → raise domain-specific exceptions (CausalEstimationError, etc.)
api/routes/*.py       → catch all exceptions, return HTTP 422/500 with error body
ui/app.py             → catch all exceptions, display st.error() — never crash the app
```

### Environment Setup
```bash
# On first run
cp .env.example .env
# Fill in ANTHROPIC_API_KEY
pip install -r requirements.txt
python data/seed_db.py        # generates synthetic data + seeds DuckDB
pytest tests/ -v              # must pass before starting API
uvicorn api.main:app --reload # start backend
streamlit run ui/app.py       # start UI (separate terminal)
```

### What NOT to Do
- Do not use `pd.DataFrame.iterrows()` — use vectorized operations
- Do not store API keys in code or comments
- Do not generate outreach for segments below CATE threshold — this is a correctness requirement, not a style preference
- Do not skip the SRM check — if you're uncertain where to place it, put it first
- Do not use `st.experimental_*` Streamlit APIs — use stable APIs only
- Do not mock the Claude API in production code paths — only in tests
- Do not hardcode the ground truth effect sizes anywhere except `data/ground_truth.py`

### Git Commit Convention
```
feat(module): short description
fix(module): what was wrong and what changed
test(module): what is being tested
chore: dependency updates, config changes
```

One logical change per commit. Do not bundle unrelated changes.

### Railway Deployment Checklist
Before pushing to Railway:
- [ ] All env vars set in Railway dashboard (not .env file)
- [ ] `requirements.txt` pinned versions
- [ ] `Procfile` defines both web (FastAPI) and worker (Streamlit) processes
- [ ] `GET /health` endpoint returns 200
- [ ] Demo reset endpoint tested: `GET /api/demo/reset`
- [ ] No hardcoded localhost URLs in UI — use `API_BASE_URL` env var

---

## Goals

1. Demonstrate causal inference depth in a product context (CATE, CUPED, DiD)
2. Show product sense through metric hierarchy framing (NSM → primary → guardrail)
3. Integrate LLM-powered personalized outreach as a *treatment* in the causal loop
4. Deploy live with a demo scenario that has known ground truth

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                    Streamlit UI                      │
│   Funnel View | Experiment Designer | Outreach Lab  │
└──────────────────────┬──────────────────────────────┘
                       │ HTTP
┌──────────────────────▼──────────────────────────────┐
│                   FastAPI Backend                    │
│                                                      │
│  /analyze      → Funnel + causal attribution        │
│  /experiment   → Design + power + SRM               │
│  /segment      → CATE + HTE by subgroup             │
│  /outreach     → Segment-conditional message gen    │
│  /results      → Lift measurement + narrative       │
└──────┬───────────────┬──────────────────────────────┘
       │               │
┌──────▼──────┐  ┌─────▼──────────────────────────────┐
│  Data Layer │  │         ML / Causal Layer           │
│  DuckDB     │  │  EconML (S/T-Learner, CausalForest) │
│  SQLite log │  │  CUPED · DiD · Power calc           │
│  Synthetic  │  │  Winsorization · BH correction      │
│  funnel data│  │  SRM detection                      │
└─────────────┘  └─────────────────────┬───────────────┘
                                        │
                        ┌───────────────▼──────────────┐
                        │        Claude API             │
                        │  Outreach generation          │
                        │  Result narrative             │
                        │  Metric hierarchy enforcement │
                        └──────────────────────────────┘
```

---

## Module Breakdown

### 1. Data Layer (`data/`)

**Synthetic funnel generator** (`data/synthetic.py`)
- Events: `impression → click → signup → activation → conversion`
- Fields: `user_id, timestamp, stage, channel, company_size, industry, treatment, outcome`
- Treatment effect: heterogeneous by segment
  - `enterprise + paid_search` → true ATE = +18% activation
  - `SMB + organic` → true ATE = +3% activation (near zero)
  - `mid_market` → true ATE = +9% activation
- Ground truth stored separately for evaluation
- N = 50,000 users, 90-day window

**Storage**
- DuckDB for funnel queries (fast, in-process)
- SQLite for experiment logs, outreach records, narrative history

---

### 2. Preprocessing (`core/preprocess.py`)

```python
def winsorize(series: pd.Series, upper_pct: float = 0.99) -> pd.Series:
    """Cap at 99th percentile. Default for experiment estimator + CUPED."""

def log_transform(series: pd.Series, offset: float = 1.0) -> pd.Series:
    """log(x + offset). Use for CATE model inputs, not experiment estimator."""

def preprocess_metric(
    series: pd.Series,
    method: Literal["winsorize", "log", "none"] = "winsorize",
    upper_pct: float = 0.99,
) -> pd.Series:
    """
    Decision rule:
    - experiment/CUPED pipeline → winsorize (preserves units, reduces variance)
    - ML model inputs → log_transform (stabilizes regression residuals)
    - raw reporting → none
    """
```

**Why winsorize for experiment layer:**
- Preserves metric interpretability in original units
- Standard practice at Meta/Uber for A/B test estimators
- Reduces variance without changing estimand (unlike log which shifts it)
- CUPED covariate computed on winsorized pre-experiment metric

---

### 3. Causal Core (`core/causal.py`)

#### CUPED
```python
def cuped_adjustment(
    df: pd.DataFrame,
    metric_col: str,         # post-experiment outcome (winsorized)
    covariate_col: str,      # pre-experiment metric (same, winsorized)
    treatment_col: str,
) -> dict:
    # theta = cov(Y, X_pre) / var(X_pre)
    # Y_cuped = Y - theta * (X_pre - mean(X_pre))
    # Returns: ATE, SE, p-value, variance_reduction_pct
```

#### CATE / HTE
```python
def estimate_cate(
    df: pd.DataFrame,
    outcome_col: str,
    treatment_col: str,
    feature_cols: list[str],
    method: Literal["s_learner", "t_learner", "causal_forest"] = "t_learner",
) -> pd.DataFrame:
    # Returns per-user CATE estimates + segment-level aggregations
    # Reuses UpliftBench patterns (EconML)
```

#### DiD (campaign-level)
```python
def diff_in_diff(
    df: pd.DataFrame,
    pre_window: tuple,
    post_window: tuple,
    treatment_group: str,
    outcome_col: str,
) -> dict:
    # For measuring causal lift of a GTM campaign launch event
```

#### SRM Detection
```python
def detect_srm(
    n_treatment: int,
    n_control: int,
    expected_split: float = 0.5,
    alpha: float = 0.01,
) -> dict:
    # Chi-square test. Returns: srm_detected bool, p_value, recommendation
```

#### Multiple Testing Correction
```python
def bh_correction(p_values: list[float], alpha: float = 0.05) -> list[bool]:
    # Benjamini-Hochberg for segment-level analyses
```

---

### 4. Experiment Designer (`core/experiment.py`)

```python
def design_experiment(
    baseline_rate: float,
    mde: float,               # minimum detectable effect
    alpha: float = 0.05,
    power: float = 0.80,
    use_cuped: bool = True,
    variance_reduction: float = 0.30,  # typical CUPED gain
) -> dict:
    # Returns: required_n, duration_days (given daily traffic),
    #          recommended_split, guardrail_metrics
```

Output format matches product-sense framing:
```json
{
  "required_n_per_arm": 4200,
  "duration_days": 14,
  "primary_metric": "activation_rate",
  "guardrail_metrics": ["unsubscribe_rate", "spam_complaint_rate"],
  "notes": "CUPED reduces required N by ~23% vs. naive estimator"
}
```

---

### 5. Outreach Module (`core/outreach.py`)

**Core framing: outreach message = treatment variant**

```python
def generate_outreach(
    segment: dict,            # {industry, company_size, funnel_stage, cate_estimate}
    product_context: str,
    tone: Literal["warm", "direct", "technical"] = "direct",
) -> dict:
    # Calls Claude API
    # Returns: subject, body, predicted_uplift_group, holdout_flag
```

**System prompt enforces:**
```
You are a GTM strategist. Generate outreach for the following segment.
Structure: [Persona insight] → [Specific pain point] → [Value prop] → [Single CTA]
Constraints:
- Max 120 words
- No generic openers ("I hope this finds you well")
- Ground the pain point in the segment's behavioral data
- Do not fabricate metrics
Output JSON: { "subject": "...", "body": "...", "cta": "..." }
```

**Segment targeting logic:**
- Only generate outreach for segments where `cate_estimate > threshold` (configurable, default: top 40% by predicted uplift)
- Auto-assign 20% holdout within each segment for lift measurement
- Log all sends to SQLite with `segment_id, message_hash, timestamp`

---

### 6. Results & Narrative (`core/narrative.py`)

```python
def generate_result_narrative(
    experiment_result: dict,   # ATE, CI, p-value, segment breakdown
    metric_hierarchy: dict,    # NSM, primary, secondary, guardrails
    recommendation: str,       # ship / iterate / abort
) -> str:
    # Calls Claude API
    # Enforces: metric moved X% → driven by segment Y → guardrail Z held/broke
    # → recommendation with reasoning
```

**System prompt:**
```
You are a data scientist presenting experiment results to a PM.
Structure your response exactly as:
1. What happened (primary metric, 1 sentence)
2. Why (segment breakdown, causal mechanism)
3. Guardrail status (each guardrail: held / breached + magnitude)
4. Recommendation: SHIP / ITERATE / ABORT with one-sentence rationale
Do not editorialize. Be direct. Max 150 words.
```

---

### 7. API Layer (`api/`)

```
POST /api/analyze          → funnel summary + causal attribution
POST /api/experiment/design → power calc + experiment spec
POST /api/segment/cate     → HTE estimates by segment
POST /api/outreach/generate → segment-conditional messages
POST /api/outreach/results  → lift measurement on outreach campaign
POST /api/narrative         → LLM result explanation
GET  /api/demo/reset        → reload synthetic data with known ground truth
```

---

### 8. UI Layer (`ui/`)

Three tabs in Streamlit:

**Tab 1: Funnel Intelligence**
- Funnel visualization (impression → conversion)
- DiD chart for campaign event
- Causal attribution breakdown by channel/segment

**Tab 2: Experiment Lab**
- Inputs: baseline rate, MDE, traffic
- Output: experiment spec card (N, duration, guardrails)
- CUPED toggle showing variance reduction
- SRM checker post-experiment

**Tab 3: Outreach Lab**
- Segment selector (dropdown by industry × company_size)
- CATE estimate shown per segment
- "Generate outreach" button → message preview
- Holdout experiment auto-configured
- Results panel: lift estimate + LLM narrative

---

## Demo Scenario

**Setup:** A B2B SaaS company launched a new onboarding flow (treatment). GTMLens analyzes the experiment.

**Known ground truth:**
- Enterprise + paid search: +18% activation (significant)
- SMB + organic: +3% activation (not significant)
- Mid-market: +9% activation (significant)

**Demo flow:**
1. User opens Funnel tab → sees activation drop at signup→activation stage
2. Runs CATE → discovers heterogeneous effect across segments
3. Goes to Outreach Lab → tool recommends targeting enterprise segment
4. Generates tailored outreach for enterprise cohort
5. Simulates 14-day experiment → results show +17.2% lift (close to ground truth)
6. Narrative: "Activation rate increased 17.2% for enterprise accounts. Effect concentrated in paid search channel. Unsubscribe rate held flat. **SHIP.**"

**Evaluation:** Tool's CATE estimate vs. known ground truth — quantifies estimation accuracy on the README.

---

## Preprocessing Decision Log

| Context | Method | Rationale |
|---|---|---|
| Experiment ATE estimator | Winsorize @ 99th pct | Preserves units, reduces variance, standard industry practice |
| CUPED covariate | Winsorize @ 99th pct | Covariate and outcome must use same transformation |
| CATE model features | Log transform (offset=1) | Stabilizes regression residuals for EconML models |
| Outreach reply rate | None | Binary outcome, no transformation needed |
| LTV / revenue metrics | Winsorize @ 99th pct | Heavy right skew, preserve dollar interpretability |

---

## File Structure

```
gtmlens/
├── CLAUDE.md
├── README.md
├── requirements.txt
├── .env.example
│
├── data/
│   ├── synthetic.py          # Funnel data generator
│   ├── ground_truth.py       # Known effect sizes for eval
│   └── seed_db.py            # Populate DuckDB + SQLite
│
├── core/
│   ├── preprocess.py         # Winsorize, log transform
│   ├── causal.py             # CUPED, CATE, DiD, SRM, BH
│   ├── experiment.py         # Power calc, experiment design
│   ├── outreach.py           # Segment targeting + Claude API
│   └── narrative.py          # Result narrative + Claude API
│
├── api/
│   ├── main.py               # FastAPI app
│   └── routes/
│       ├── analyze.py
│       ├── experiment.py
│       ├── outreach.py
│       └── narrative.py
│
├── ui/
│   └── app.py                # Streamlit app (3 tabs)
│
└── tests/
    ├── test_preprocess.py
    ├── test_causal.py
    ├── test_experiment.py
    └── test_outreach.py
```

---

## Environment Variables

```bash
ANTHROPIC_API_KEY=
DATABASE_URL=./data/gtmlens.duckdb
SQLITE_PATH=./data/logs.db
CUPED_VARIANCE_REDUCTION_TARGET=0.30
CATE_UPLIFT_THRESHOLD=0.40      # top 40% segments get outreach
WINSORIZE_UPPER_PCT=0.99
LOG_OFFSET=1.0
HOLDOUT_FRACTION=0.20
```

---

## Resume / Interview Talking Points

- "Winsorize at 99th percentile for the experiment estimator to reduce variance without shifting the estimand — same approach used in production at Meta and Uber"
- "CUPED reduces required sample size by ~23%, cutting experiment runtime from 18 to 14 days"
- "Outreach is modeled as a treatment variant — we measure causal lift on reply rate, not just open rate"
- "CATE identifies heterogeneous effects; we only generate outreach for segments where predicted uplift clears the threshold — precision over volume"
- "BH correction across segment-level tests controls false discovery rate without being as conservative as Bonferroni"

---

## Out of Scope (MVP)

- Real email sending (Sendgrid etc.)
- CRM integration
- Multi-channel sequencing
- Fine-tuning / RAG on company data
- Auth / multi-tenant

---

## Success Criteria

- [ ] CATE estimates within ±3pp of ground truth on demo scenario
- [ ] Power calculator matches standard formula with CUPED adjustment
- [ ] Outreach generated only for high-uplift segments (threshold enforced)
- [ ] LLM narrative always follows NSM → primary → guardrail → recommendation structure
- [ ] SRM correctly detected on intentionally imbalanced test case
- [ ] Deployed on Railway, demo accessible via public URL
- [ ] README includes CATE accuracy table vs. ground truth
