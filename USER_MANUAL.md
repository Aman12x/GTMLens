# GTMLens — User Manual

GTMLens finds which customer segments actually respond to outreach (not just who opened an email), generates targeted messages for those segments, and measures whether the outreach *caused* the activation.

**Just want to explore?** Click **Try demo** on the login screen — no account needed. The demo runs on a synthetic dataset with known ground truth so you can see every tab without uploading anything.

---

## Quick start (your own data)

| Step | Tab | What you do |
|---|---|---|
| 1 | Data | Upload your funnel CSV |
| 2 | Data | Upload your contacts CSV |
| 3 | Outreach | Pick a segment, generate a message, send |
| 4 | Data | After the campaign, paste who activated |

Results tab updates automatically once step 4 is done.

---

## Step 1 — Funnel CSV

One row per user. Three columns are required; everything else is optional.

| Column | Required | What to put |
|---|---|---|
| `user_id` | yes | Any unique identifier |
| `treatment` | yes | `1` if this user received outreach, `0` if they didn't |
| `activated` | yes | `1` if they converted/activated, `0` if they didn't |
| `company_size` | no | `SMB`, `mid_market`, or `enterprise` |
| `channel` | no | `organic`, `paid_search`, `social`, `referral`, or `email` |
| `industry` | no | Any string |

**You don't need to rename your columns.** Common names are accepted automatically — `converted` works for `activated`, `variant` or `group` works for `treatment`, `customer_id` works for `user_id`.

**You need both treated and control rows.** GTMLens measures the *difference* in activation between people who got outreach and people who didn't. If everyone in your dataset received outreach, upload it anyway — but note the causal estimate will be weaker without a clean holdout.

**Which activation event to use:** pick the one closest to revenue with enough volume. If you have 500 users, a 30% activation rate (150 activations) is workable; a 2% rate (10 activations) isn't. If you have multiple events (trial → paid → expansion), use the one you most want to move.

Download the **Sample CSV** button on the Data tab to see the exact format.

---

## Step 2 — Contacts CSV

One row per person you want to email.

| Column | Required | Notes |
|---|---|---|
| `email` | yes | Used for sending and tracking activations |
| `first_name` | no | Personalises the greeting line |
| `company` | no | Context for message generation |
| `company_size` | no | Must match your funnel data exactly (`enterprise` not `Enterprise`) |
| `channel` | no | Must match your funnel data exactly |

`company_size` and `channel` are how GTMLens finds the right contacts when you send to a segment. If they don't match your funnel data, the send will find zero contacts.

---

## Step 3 — Send outreach (Outreach tab)

1. The **Segment CATE analysis** table shows predicted activation lift per segment, significance-tested and corrected for multiple comparisons
2. Click **Use** on a recommended segment (green rows cleared both significance and the uplift threshold)
3. Edit the product context to describe what your product actually does
4. Click **Generate outreach** — Claude writes a subject, body, and CTA for that segment
5. Review the message, then click **Send to N contacts**

**Holdout badge:** if the generated message shows a "Holdout" badge, it was generated for a preview-only user in the 20% control group — don't send it. GTMLens reserves 20% of each segment as a control group so it can measure whether the email *caused* the activation, rather than just correlating with it.

---

## Step 4 — Import activations (Data tab)

After your campaign runs, come back to the Data tab. In the **Import activation results** card, paste the email addresses of contacts who activated — one per line or comma-separated. Click **Mark as activated**.

The Results tab will switch from "historical baseline" to **real campaign lift**: activation rate among people who got the email minus activation rate among the holdout group, per segment.

---

## Results tab

| Column | What it means |
|---|---|
| Predicted | T-Learner CATE estimate at the time you sent |
| Observed | Actual treatment rate minus holdout rate |
| Model accuracy | How far the prediction was from reality (±3pp is good) |
| Signal | Green = outreach lifted activation · Red = it didn't |

If the banner says **"Showing historical segment baseline"**, you haven't imported activation results yet (step 4). The numbers shown are from your original funnel upload, not your campaign.

---

## FAQ

**Upload fails: "missing required columns"**
Download the Sample CSV from the Data tab and compare. Your column names don't have to match exactly — `converted`, `variant`, `customer_id` are all accepted — but if your name isn't on the alias list, rename it manually.

**Upload fails: "needs both treated and control users"**
Every row in your dataset has the same treatment value. You need at least some users who *didn't* receive the outreach. Without a control group there's no counterfactual to compare against.

**Segments tab shows no recommended segments**
Either no segments are statistically significant, or the predicted lift is below the threshold. Try uploading more data (more users per segment), or check whether your `company_size` and `channel` columns have enough variation.

**"Send to segment" finds 0 contacts**
The `company_size` or `channel` values in your contacts CSV don't match your funnel data. Capitalisation matters: `enterprise` ≠ `Enterprise`. Re-upload the contacts CSV with matching values.

**Results tab still shows "historical baseline" after step 4**
Activation import only works for contacts who were sent outreach through GTMLens. If the emails went out through another tool, GTMLens has no record of them and can't track activations against them.

**The same contact always ends up in the holdout group**
Holdout assignment is deterministic — it's based on the contact's email and the segment name. The same person always lands in the same bucket for a given segment. This is intentional: it prevents someone from being a control in one measurement and treated in another.

**Want to start over**
Re-uploading a funnel CSV replaces the existing data entirely. Re-uploading contacts updates them in place. If you're in the demo, use the **Reset demo** button in the top nav.
