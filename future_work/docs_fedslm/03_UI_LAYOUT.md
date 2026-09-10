# FedSLM — UI Layout & Design Specification

**Companion to** `01_PRD.md` and `02_TECH_STACK.md` · v1.0 · 10 August 2026
**Platform:** Streamlit multipage app, run locally (`streamlit run app/Home.py`)

---

## 1. The brief

**Subject.** A federated, multi-agent healthcare SLM research prototype.

**Audience.** Your FYP supervisor and evaluation panel — technical, sceptical, and giving you roughly five minutes of real attention before forming an opinion. Secondary audience: you, using it daily to inspect experiment output.

**The page's single job:** *make the two invisible claims visible.*

1. **Raw data never crossed the hospital boundary.** Only weights did.
2. **This answer has a chain of custody.** Four components handled it; here is what each one did and why the verdict came out the way it did.

Everything in this specification exists to serve one of those two sentences. A chart that does not is decoration and gets cut.

### 1.1 Design direction — "Ward Board"

The visual world we are borrowing from is the hospital ward itself: the **status board** at the nurses' station (condensed signage, colour-coded states, dense information read at a glance) and the **patient chart** (a paper record where every entry is timestamped, initialled, and attributable).

That gives us an instrument-grade interface: dark, quiet, monospaced numerals, hairline rules, no ornament. The colour restraint is the point — in a ward, colour *means* something, so we spend it only on state.

**What we are deliberately avoiding.** The three looks that current AI-generated design defaults to: cream background with a high-contrast serif and a terracotta accent; near-black with one acid-green accent; and a broadsheet layout of hairline rules and dense newspaper columns. This design is dark, but it uses *three* muted clinical status hues rather than one neon accent, and its ground is a warm graphite rather than a blue-black. The warmth is what keeps it reading as a room rather than a terminal.

---

## 2. Design tokens

### 2.1 Colour

Six core values. Every colour on screen is one of these or a documented tint of one.

| Token | Hex | Role |
|---|---|---|
| `--graphite` | `#191C1F` | Page ground. Warm-neutral dark, not black, not blue-black. |
| `--slate` | `#22262B` | Panel and card surfaces, one step up from ground. |
| `--rule` | `#31363C` | Hairline borders, dividers, chart gridlines. 1px, never heavier. |
| `--bone` | `#E8E5DE` | Primary text. Warm off-white — printed chart paper, not screen white. |
| `--ash` | `#8A9299` | Labels, captions, axis text, inactive states. |
| `--iodine` | `#3E9E8A` | The single accent. Named for the antiseptic; muted surgical teal. |

**Status colours** — used *only* for verdicts and risk. Never decorative.

| Token | Hex | Meaning |
|---|---|---|
| `--pass` | `#3E9E8A` | PASS (same value as `--iodine` — the accent *is* the good state) |
| `--review` | `#C98B3A` | REVIEW |
| `--fail` | `#C25668` | FAIL |

**Series colours** — hospital identity in charts. Deliberately cool-to-warm and desaturated so they never compete with status hues.

| Token | Hex | Series |
|---|---|---|
| `--hosp-a` | `#7FA6CE` | Hospital A |
| `--hosp-b` | `#A98BC4` | Hospital B |
| `--hosp-c` | `#C9B36E` | Hospital C |
| `--global` | `#E8E5DE` | Global model — drawn in `--bone` |

> The global model is drawn in the same colour as the body text. That is intentional and it encodes something true: the three hospital lines are inputs; the bone line is the thing you are actually here to read.

### 2.2 Typography

Three faces, three registers — signage, record, instrument. All free on Google Fonts.

| Role | Face | Usage |
|---|---|---|
| **Display / signage** | **Archivo Condensed**, 600 | Page titles, section eyebrows. Uppercase, letter-spacing `0.08em`. Condensed because ward signage is condensed — it fits more label into less rail. |
| **Body / record** | **IBM Plex Sans**, 400/500 | All prose, answers, agent output, table cells. Designed for technical documentation; it reads as a record, not as marketing. |
| **Data / instrument** | **IBM Plex Mono**, 400/500 | **Every numeral on the site.** Metrics, deltas, round counters, MB, latency, elapsed ms, IDs. `font-variant-numeric: tabular-nums`. |

The mono-for-all-numbers rule is not cosmetic. This is a results dashboard; when numbers are in tabular mono, columns align, deltas are comparable at a glance, and the interface reads as an instrument rather than a document with numbers typed into it.

**Scale** (1.25 ratio, 16px base):

| Token | Size / line-height | Use |
|---|---|---|
| `--t-display` | 32 / 36 px | Page title |
| `--t-h2` | 22 / 28 px | Section head |
| `--t-h3` | 17 / 24 px | Panel head |
| `--t-body` | 15 / 24 px | Body copy |
| `--t-small` | 13 / 20 px | Captions, help text |
| `--t-eyebrow` | 11 / 16 px | Uppercase labels, `0.08em` tracking |
| `--t-metric` | 30 / 34 px | Big numbers, mono, 500 |

### 2.3 Space, shape, motion

- **Spacing scale:** 4 / 8 / 12 / 16 / 24 / 32 / 48 px. Nothing off-scale.
- **Radius:** `2px` on panels, `0` on rules and rails. Instruments do not have rounded corners; softness here would read as consumer app.
- **Borders:** 1px `--rule`. No drop shadows anywhere — depth comes from the ground/surface step, which is how a status board works.
- **Motion:** one purpose only — showing sequence. Agent stations fill in as each one completes (150 ms fade + 4px rise). The boundary line has a slow 3 s dash-offset drift when a round is running, and is static otherwise. Nothing else animates. All of it is wrapped in `@media (prefers-reduced-motion: reduce)`.

---

## 3. The signature element: the Provenance Rail

One visual form, used twice, encoding both project claims.

### 3.1 As the agent chain (Ask page)

A vertical rail beside the answer. Four stations — the numbering is legitimate here because this genuinely *is* an ordered pipeline, and the order carries information the reader needs.

```
│
●━━  01  CLINICAL AGENT                          412 ms
│         Drafted answer · confidence 0.71
│
●━━  02  EVIDENCE AGENT                          890 ms
│         5 passages retrieved · grounding 0.83
│
●━━  03  SAFETY AGENT                            230 ms
│         2 flags · risk MEDIUM
│
●━━  04  VERIFIER                                 95 ms
│         reliability 0.76
│
┢━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃   ⬤  REVIEW              ┃   ← the stamp
┃   grounded, but 2 claims ┃
┃   lack support           ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━┛
```

- Nodes fill top-to-bottom as agents complete, so the user watches the chain form.
- Node colour: `--ash` pending → `--iodine` complete → `--review`/`--fail` if that station raised the concern that decided the verdict.
- The stamp at the foot is the only element on the page allowed a heavy 2px border. It is the one loud thing, and everything around it stays quiet so it lands.

### 3.2 As the federation boundary (Federated Training page)

The same rail rotated, with a dashed line through it — the boundary.

```
  HOSPITAL A          HOSPITAL B          HOSPITAL C
  ▣ 18,432 records    ▣ 12,907 records    ▣ 22,115 records
       │                   │                   │
       ●                   ●                   ●
       │                   │                   │
╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌  DATA BOUNDARY
       │                   │                   │
       ▼                   ▼                   ▼
        LoRA adapter · 7.8 MB each · weights only
                           │
                           ▼
              ┌────────────────────────┐
              │   FedAvg AGGREGATION   │
              │   round 06 / 10        │
              └────────────────────────┘

  ┌─────────────────────────────┬─────────────────────────────┐
  │  WEIGHTS TRANSFERRED        │  RECORDS TRANSFERRED        │
  │  281.4 MB                   │  0                          │
  └─────────────────────────────┴─────────────────────────────┘
```

**That second counter is the whole thesis of the project as one number.** It is pinned at `0`, rendered in `--bone` mono at `--t-metric`, and sits beside a counter that visibly climbs. Nothing else on the page argues the privacy claim as directly, which is why the page is built around it rather than around the loss curve.

---

## 4. Global chrome

### 4.1 Sidebar

```
┌──────────────────────────┐
│  FedSLM                  │  Archivo Condensed, uppercase
│  federated · multi-agent │  --t-eyebrow, --ash
│  ────────────────────────│
│                          │
│  ◈  Ask                  │  ← active: 2px --iodine left bar
│  ◈  Hospitals & Data     │
│  ◈  Federated Training   │
│  ◈  Results              │
│  ◈  Runs                 │
│                          │
│  ────────────────────────│
│  SYSTEM                  │  --t-eyebrow
│                          │
│  Model                   │
│  ▸ FedSLM (global)  ▾    │  selector: which of the 4 systems
│                          │
│  Adapter   round-10      │  mono
│  Device    cpu           │  mono
│  Index     48,203 psg    │  mono
│                          │
│  ────────────────────────│
│  ⚠ Research prototype.   │  --t-small, --ash
│  Not medical advice.     │
└──────────────────────────┘
```

The system selector in the sidebar is what makes the four-way comparison tangible — the panel can switch from FedSLM to Single SLM mid-demo and ask the same question again.

### 4.2 Page header (every page)

```
FEDERATED TRAINING                                    ◷ run 2026-08-08 14:32
────────────────────────────────────────────────────────────────────────────
Ten rounds of FedAvg across three hospitals. Weights aggregate; data does not.
```

Eyebrow-style uppercase title, a hairline rule, then one sentence in `--ash` stating what the page shows. One sentence, always — if a page needs two, the page is doing two jobs and should be split.

### 4.3 The disclaimer

Fixed in the sidebar footer, and repeated inline beneath every generated answer. Never a dismissible banner — a research prototype that produces health text should not let you close the notice that it is a research prototype.

---

## 5. Page layouts

### 5.1 Page 1 — Ask *(the demo page; build it first)*

```
ASK                                              system: FedSLM (global) ▾
──────────────────────────────────────────────────────────────────────────
Submit a clinical question. Four components handle it in sequence.

┌────────────────────────────────────────────────────────────────────────┐
│  Ask a question                                                        │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │ Which drug class is first-line for stage 1 hypertension?         │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│  Try:  ▫ MCQ example   ▫ Yes/no/maybe example   ▫ Unsafe-query probe   │
│                                                    [  Run pipeline  ]  │
└────────────────────────────────────────────────────────────────────────┘

┌───── PROVENANCE ────────┐  ┌───── ANSWER ─────────────────────────────┐
│                         │  │                                          │
│  ●━ 01 CLINICAL   412ms │  │  Thiazide diuretics, ACE inhibitors,      │
│  │    conf 0.71         │  │  ARBs, and calcium channel blockers are   │
│  │                      │  │  all considered first-line …             │
│  ●━ 02 EVIDENCE   890ms │  │                                          │
│  │    grounding 0.83    │  │  ┌─ supported ──────────────────────┐    │
│  │                      │  │  │ "thiazide diuretics … first-line" │   │
│  ●━ 03 SAFETY     230ms │  │  │  ▸ PubMedQA 24817392   sim 0.88   │   │
│  │    risk MEDIUM       │  │  └──────────────────────────────────┘    │
│  │                      │  │  ┌─ unsupported ────────────────────┐    │
│  ●━ 04 VERIFIER    95ms │  │  │ "start at 12.5 mg daily"          │   │
│  │                      │  │  │  ▸ no passage supports this       │   │
│  ┏━━━━━━━━━━━━━━━━━━━━┓ │  │  └──────────────────────────────────┘    │
│  ┃ ⬤ REVIEW           ┃ │  │                                          │
│  ┃ 2 claims unsupported┃ │  │  ⚠ Research prototype. Not medical      │
│  ┗━━━━━━━━━━━━━━━━━━━━┛ │  │    advice.                               │
│                         │  │                                          │
│  ▸ full trace (JSON)    │  └──────────────────────────────────────────┘
└─────────────────────────┘
   ~34% width                       ~66% width

┌──── RETRIEVED EVIDENCE ────────────────────────────────────────────────┐
│  #  score  source          passage                                     │
│  1  0.88   PubMedQA 2481…  Thiazide-type diuretics remain among …     │
│  2  0.81   PubMedQA 3319…  In a cohort of 1,204 patients …            │
│  …                                                                     │
└────────────────────────────────────────────────────────────────────────┘
```

**Behaviour.** The rail fills station by station using `st.empty()` placeholders, so the user watches the pipeline run rather than staring at a spinner. Claims in the answer are marked inline: supported claims get a `--iodine` left border, unsupported claims get `--review`. That inline marking is what turns "grounding: 0.83" from a number into something the panel can actually see.

**Unsafe-query probe** is a first-class example button, not hidden. Demonstrating a correct **FAIL** is more persuasive than three PASSes.

### 5.2 Page 2 — Hospitals & Data

```
HOSPITALS & DATA
──────────────────────────────────────────────────────────────────────────
Three simulated hospitals with deliberately unequal data. Split: non-IID ▾

┌── HOSPITAL A ───────┐ ┌── HOSPITAL B ───────┐ ┌── HOSPITAL C ───────┐
│ general medicine    │ │ pharmacology        │ │ surgery · pediatrics│
│                     │ │                     │ │                     │
│ 18,432              │ │ 12,907              │ │ 22,115              │
│ records             │ │ records             │ │ records             │
│                     │ │                     │ │                     │
│ ▇▇▇▇▇▁▁▂▁           │ │ ▁▂▇▇▇▇▇▁▁           │ │ ▁▁▁▂▇▇▇▇▇           │
│ subject mix         │ │ subject mix         │ │ subject mix         │
└─────────────────────┘ └─────────────────────┘ └─────────────────────┘
   --hosp-a border         --hosp-b border         --hosp-c border

┌──── SUBJECT DISTRIBUTION ACROSS HOSPITALS ─────────────────────────────┐
│  stacked horizontal bars, 21 MedMCQA subjects × 3 hospitals            │
│  the visible unevenness IS the deliverable of this page                │
└────────────────────────────────────────────────────────────────────────┘

┌──── IID vs NON-IID ───────────┐ ┌──── TOKEN LENGTHS ───────────────────┐
│  side-by-side heatmaps        │ │  overlaid distributions, 3 hospitals │
│  Dirichlet α=100  |  α=0.3    │ │                                      │
└───────────────────────────────┘ └──────────────────────────────────────┘

┌──── SAMPLE RECORDS ────────────────────────────────────────────────────┐
│  hospital ▾   dataset ▾              — inspect what each shard holds   │
└────────────────────────────────────────────────────────────────────────┘
```

The IID/non-IID toggle at the top re-renders every chart on the page. Flipping it in the viva is a five-second demonstration of what non-IID means, and it beats explaining Dirichlet concentration verbally.

### 5.3 Page 3 — Federated Training

```
FEDERATED TRAINING                                       run: fedavg_noniid ▾
──────────────────────────────────────────────────────────────────────────
Ten rounds of FedAvg across three hospitals. Weights aggregate; data does not.

┌──── THE BOUNDARY ──────────────────────────────────────────────────────┐
│                                                                        │
│         [ the horizontal Provenance Rail from §3.2 ]                   │
│                                                                        │
│  WEIGHTS TRANSFERRED          RECORDS TRANSFERRED                      │
│  281.4 MB                     0                                        │
│  7.8 MB × 3 clients × 2 × 6   raw data never leaves a client           │
└────────────────────────────────────────────────────────────────────────┘

┌──── CONVERGENCE ───────────────────────┬──── ACCURACY ─────────────────┐
│  loss vs round                         │  global accuracy vs round     │
│  ─── hospital A (local)   --hosp-a     │  ─── global      --bone       │
│  ─── hospital B (local)   --hosp-b     │  ┈┈┈ centralised --ash dashed │
│  ─── hospital C (local)   --hosp-c     │      (the reference line)     │
│  ━━━ global               --bone 2px   │                               │
└────────────────────────────────────────┴───────────────────────────────┘

┌──── ROUND LOG ─────────────────────────────────────────────────────────┐
│ round  loss_a  loss_b  loss_c  global_loss  acc     sec    MB          │
│ 01     2.418   2.502   2.377   2.431        0.312   184.2  46.9        │
│ 02     2.104   2.288   2.041   2.144        0.348   179.8  46.9        │
│ …      all mono, tabular figures, delta arrows in --iodine / --review  │
└────────────────────────────────────────────────────────────────────────┘

┌──── STRATEGY ─────────────────┐ ┌──── COST ────────────────────────────┐
│  FedAvg vs FedProx, non-IID   │ │  wall-clock and MB per round         │
└───────────────────────────────┘ └──────────────────────────────────────┘
```

The dashed **centralised reference line** on the accuracy chart is the honest comparison — it shows what federation costs you relative to pooling all the data. If the federated line sits below it, that gap is a result, and the chart should show it plainly rather than omit the reference.

### 5.4 Page 4 — Results

```
RESULTS
──────────────────────────────────────────────────────────────────────────
Four systems, one held-out test set, identical decoding parameters.

┌──── SYSTEM COMPARISON ─────────────────────────────────────────────────┐
│                    fed?  agents?  ACC    F1     GND    HALL   REL  LAT │
│  1 Single SLM       ·      ·      0.412  0.398  0.31   0.44   0.42  0.6│
│  2 Multi-Agent SLM  ·      ✓      0.428  0.417  0.79   0.16   0.68  1.9│
│  3 Federated SLM    ✓      ·      0.451  0.442  0.33   0.41   0.46  0.6│
│  4 FedSLM           ✓      ✓      0.474  0.466  0.82   0.12   0.77  2.1│
│                                                                        │
│  best value per column in --iodine; latency cost shown, not hidden     │
└────────────────────────────────────────────────────────────────────────┘

┌──── ABLATION ─────────────────┐ ┌──── RELIABILITY BREAKDOWN ───────────┐
│  grouped bars, 4 systems      │ │  stacked: correctness · grounding ·  │
│  × selected metric ▾          │ │  safety · confidence, per system     │
└───────────────────────────────┘ └──────────────────────────────────────┘

┌──── VERDICT MIX ──────────────┐ ┌──── ACCURACY vs LATENCY ─────────────┐
│  PASS/REVIEW/FAIL proportions │ │  scatter — the real tradeoff, one    │
│  per system, status colours   │ │  point per system                    │
└───────────────────────────────┘ └──────────────────────────────────────┘

┌──── WHAT THIS SHOWS ───────────────────────────────────────────────────┐
│  Written interpretation, including where FedSLM does not win.          │
│  Every figure links to the run artifact that produced it.              │
└────────────────────────────────────────────────────────────────────────┘
```

Two rules for this page, both from the PRD's no-fabrication requirement:

1. **Any metric with no completed run renders as `not run`** in `--ash`, never as a blank cell and never as an estimate.
2. **Every number links to its artifact** in `results/`. A panel member who clicks a cell should land on the file that produced it.

Latency has its own column and its own scatter plot because FedSLM is roughly 3× slower than the single model. Showing that cost prominently is more credible than burying it.

### 5.5 Page 5 — Runs

```
RUNS
──────────────────────────────────────────────────────────────────────────
Every saved query trace and experiment artifact.

┌── FILTERS ──────┐  ┌──── TRACE LIST ──────────────────────────────────┐
│ verdict     ▾   │  │ time      system    verdict   gnd   lat   query  │
│ system      ▾   │  │ 14:32:07  FedSLM    ⬤ PASS    0.91  2.1s  Whic…  │
│ date        ▾   │  │ 14:31:44  FedSLM    ⬤ REVIEW  0.62  2.4s  What…  │
│ min grounding   │  │ 14:30:12  Single    ⬤ FAIL    0.18  0.7s  How …  │
└─────────────────┘  └──────────────────────────────────────────────────┘

┌──── SELECTED TRACE ────────────────────────────────────────────────────┐
│  the full Provenance Rail replayed, plus raw JSON  [ download trace ]  │
└────────────────────────────────────────────────────────────────────────┘
```

The verdict filter is the most useful control here — during development you will mostly want to look at FAILs, since those are where the reliability layer is either working or misfiring.

---

## 6. Component inventory

Build these once in `app/components/`, use everywhere.

| Component | File | Notes |
|---|---|---|
| `page_header(title, subtitle)` | `header.py` | Eyebrow title, hairline rule, one-sentence subtitle |
| `provenance_rail(stages, verdict)` | `rail.py` | **The signature.** Vertical mode for Ask, horizontal for Federated |
| `verdict_stamp(verdict, reason)` | `rail.py` | Glyph + word + reason. Never colour alone |
| `metric_tile(label, value, unit, delta)` | `metrics.py` | Mono tabular numerals, optional delta arrow |
| `hospital_card(id, name, count, hist)` | `hospital.py` | Coloured left border per hospital |
| `evidence_panel(passages)` | `evidence.py` | Ranked list with similarity scores |
| `claim_marked_text(answer, claims)` | `claims.py` | Inline supported/unsupported borders |
| `comparison_table(df)` | `tables.py` | Best-per-column highlighting, `not run` handling |
| `boundary_diagram(state)` | `boundary.py` | The dashed data boundary + dual counters |
| `chart_theme()` | `charts.py` | Applies tokens to every Plotly figure |

**`chart_theme()` is not optional.** Every Plotly chart routes through it: `--graphite` paper, `--rule` gridlines at 1px, `--ash` axis text in IBM Plex Mono, no chart title (the panel header is the title), no legend box border, series colours from the token table. Ten charts styled individually will drift apart; ten charts through one theme function stay one system.

---

## 7. Streamlit implementation notes

Streamlit's theming is limited. Here is the honest split between what you configure and what you inject.

### 7.1 `.streamlit/config.toml` — what Streamlit gives you natively

```toml
[theme]
base                = "dark"
primaryColor        = "#3E9E8A"   # --iodine
backgroundColor     = "#191C1F"   # --graphite
secondaryBackgroundColor = "#22262B"  # --slate
textColor           = "#E8E5DE"   # --bone
font                = "sans serif"

[server]
runOnSave = true
```

### 7.2 `app/assets/theme.css` — everything else

Load once in `Home.py` with `st.markdown(open(...).read(), unsafe_allow_html=True)`.

```css
@import url('https://fonts.googleapis.com/css2?family=Archivo+Narrow:wght@600&family=IBM+Plex+Sans:wght@400;500&family=IBM+Plex+Mono:wght@400;500&display=swap');

:root{
  --graphite:#191C1F; --slate:#22262B; --rule:#31363C;
  --bone:#E8E5DE;     --ash:#8A9299;   --iodine:#3E9E8A;
  --pass:#3E9E8A;     --review:#C98B3A; --fail:#C25668;
  --hosp-a:#7FA6CE;   --hosp-b:#A98BC4; --hosp-c:#C9B36E;
}

html, body, [class*="css"]{ font-family:'IBM Plex Sans',sans-serif; }

/* signage */
.fs-title{
  font-family:'Archivo Narrow',sans-serif; font-weight:600;
  text-transform:uppercase; letter-spacing:.08em;
  font-size:32px; line-height:36px; color:var(--bone);
}
.fs-eyebrow{
  font-family:'Archivo Narrow',sans-serif; font-size:11px;
  text-transform:uppercase; letter-spacing:.08em; color:var(--ash);
}

/* instrument — every numeral on the site */
.fs-num, .fs-metric, [data-testid="stMetricValue"]{
  font-family:'IBM Plex Mono',monospace;
  font-variant-numeric:tabular-nums;
}

/* panels: depth from the ground/surface step, never shadow */
.fs-panel{
  background:var(--slate); border:1px solid var(--rule);
  border-radius:2px; padding:16px 20px;
}

/* the stamp — the one heavy element on the page */
.fs-stamp{ border:2px solid; border-radius:2px; padding:12px 16px; }
.fs-stamp--pass  { border-color:var(--pass);   color:var(--pass); }
.fs-stamp--review{ border-color:var(--review); color:var(--review); }
.fs-stamp--fail  { border-color:var(--fail);   color:var(--fail); }

/* claim marking */
.fs-claim--supported  { border-left:2px solid var(--iodine); padding-left:10px; }
.fs-claim--unsupported{ border-left:2px solid var(--review); padding-left:10px; }

@media (prefers-reduced-motion: reduce){
  *{ animation:none !important; transition:none !important; }
}
```

### 7.3 What Streamlit cannot do, and the workaround

| Want | Reality | Do this |
|---|---|---|
| Custom sidebar nav | Streamlit auto-generates from `pages/` filenames | Name files `1_Ask.py`, `2_Hospitals_&_Data.py` … and style via CSS |
| The Provenance Rail | No native component | Build it as one HTML string in `rail.py`, render with `unsafe_allow_html=True` |
| Progressive agent reveal | No streaming primitive for this | `ph = st.empty()`, re-render the rail HTML after each agent completes |
| Custom fonts | Not exposed in `config.toml` | `@import` in the injected CSS |
| Charts matching the theme | Plotly defaults will not match | Route everything through `chart_theme()` |

**Do not fight Streamlit.** If a piece of layout needs three CSS hacks to survive a Streamlit version bump, simplify the layout instead. The dashboard has to still work on demo day.

---

## 8. States

Empty and error states are direction, not decoration — say what happened and what to do next.

| State | Message |
|---|---|
| No experiment results yet | **No results yet.** Run `python -m experiments.e1_comparison` to populate this page. |
| No federated run selected | **No federated run found.** Train a global model in `notebooks/06_federated.ipynb`, then place the log in `results/`. |
| Adapter missing | **Global adapter not found at `checkpoints/global/`.** Download it from Drive, or switch to the Single SLM in the sidebar. |
| Model loading (first query) | Skeleton rail with all four stations in `--ash`, caption: *Loading model — first query takes about 30 seconds.* |
| Pipeline running | Completed stations fill; the current one pulses at 1.2 s; pending stay `--ash` |
| Retrieval returned nothing | **No supporting passages found.** The verifier will treat this answer as ungrounded. |
| Experiment not run | `not run` in `--ash` mono. **Never** an estimate, never a blank cell. |

---

## 9. Copy rules

- **Sentence case everywhere** except signage eyebrows, which are uppercase.
- **Name things as the user sees them.** "Hospital A", not "client_0". "Round 6 of 10", not "fl_round_idx=5".
- **Actions say what happens.** The button says **Run pipeline**; the result heading says **Pipeline complete**. Same verb through the whole flow.
- **Errors do not apologise and are never vague.** "Global adapter not found at `checkpoints/global/`" beats "Something went wrong."
- **One job per element.** A label labels; help text explains; a caption cites. Nothing does two.
- **The disclaimer is always the same sentence:** *Research prototype. Not medical advice.* Same words, every location — consistency is what makes people stop reading past it.

---

## 10. Accessibility & responsive

- **Never colour alone.** Every verdict carries a glyph and the word: `⬤ PASS`, `⬤ REVIEW`, `⬤ FAIL`. Hospital series carry line style as well as hue in charts.
- **Contrast.** `--bone` on `--graphite` ≈ 13:1. `--ash` on `--graphite` ≈ 5.5:1. `--iodine` on `--graphite` ≈ 5.8:1. All clear WCAG AA for their sizes.
- **Focus.** 2px `--iodine` outline with a 2px offset on every interactive element. Streamlit's default focus ring is weak — override it.
- **Reduced motion.** Respected globally (§7.2).
- **Layout.** Three-column card rows collapse to one column under 900px; the Ask page rail moves above the answer under 1100px. The wide tables scroll inside their own container — the page body never scrolls horizontally.
- **Demo reality:** the viva will be on a projector, likely 1280×720 at a distance. Test at that size. `--t-body` at 15px is the floor; do not go smaller for the sake of fitting more in.

---

## 11. UI build order

The dashboard is Phase 9, but two pieces get built early because they pay for themselves during development.

| # | Build | When | Why then |
|---|---|---|---|
| 1 | `theme.css`, `config.toml`, `page_header`, `chart_theme` | Phase 3 | Every chart you make from here on is already on-brand — no restyling pass later |
| 2 | Page 2 — Hospitals & Data | Phase 3 | You need to *see* the non-IID split to trust your partitioning code |
| 3 | `provenance_rail` + Page 1 — Ask | Phase 5 | It is the fastest way to debug the agent pipeline |
| 4 | Page 3 — Federated Training | Phase 6 | Reads the round CSV; makes convergence problems obvious |
| 5 | Page 4 — Results | Phase 8 | Needs completed experiments |
| 6 | Page 5 — Runs | Phase 9 | Convenience layer, last |
| 7 | Polish, projector test, reduced-motion check | Phase 9 | — |

---

## 12. The one-line test

If a panel member looks at this dashboard for ten seconds and does not come away knowing that **weights moved and data did not**, and that **every answer carries a verdict with a reason**, the design has failed regardless of how it looks.

Every layout decision above was made against that test.

---

**End of specification.** Documents `01_PRD.md`, `02_TECH_STACK.md`, and `03_UI_LAYOUT.md` together define the project. Implementation begins on approval, starting at Phase 1.
