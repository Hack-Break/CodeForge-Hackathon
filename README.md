# NeuralPath — AI-Adaptive Onboarding Engine
### Technical Disclosure & Algorithm Documentation

> **"Google Maps for Skills"** — computes the shortest, personalized path from current skill position to target role competency.

---

## Table of Contents
1. [System Architecture](#architecture)
2. [Algorithm Deep Dive](#algorithms)
3. [Dataset Citations](#datasets)
4. [Originality Statement](#originality)
5. [Validation Metrics](#metrics)
6. [API Reference](#api)
7. [Quick Start](#quickstart)

---

## 1. System Architecture <a name="architecture"></a>

```
┌─────────────────────────────────────────────────────────────────────┐
│                        INPUT LAYER                                  │
│   Resume (PDF/DOCX/TXT)          Job Description (PDF/DOCX/Text)   │
└─────────────────────┬───────────────────────┬───────────────────────┘
                       │                       │
                       ▼                       ▼
              ┌────────────────────────────────────────┐
              │   NLP Extraction Pipeline  (v3.0)      │
              │                                        │
              │  L1 spaCy NER  (ORG/PRODUCT entities)  │
              │  L2 PhraseMatcher  (1 400-skill lexicon)│
              │  L3 BERT NER  (dslim/bert-base-NER)    │
              │       ↓ merge + deduplicate            │
              │  L4 Groq Llama 3.3 70B  (scores only) │
              │  JSON: skill, proficiency, years,      │
              │        importance, required_level       │
              └────────────────────┬───────────────────┘
                                   │
                    ┌──────────────┼──────────────┐
                    ▼              ▼              ▼
           ┌──────────────┐ ┌──────────┐ ┌────────────────┐
           │  O*NET Slug  │ │  Domain  │ │  Proficiency   │
           │  Matcher     │ │ Detector │ │  Scorer (BKT)  │
           └──────┬───────┘ └────┬─────┘ └───────┬────────┘
                  │              │               │
                  └──────────────┴───────────────┘
                                 │
                                 ▼
                    ┌────────────────────────┐
                    │   BKT Gap Analyzer     │
                    │   raw_gap = req − cur  │
                    │   adj_gap = raw × 1.15 │
                    │   → SKIP/FAST/REQUIRED │
                    └────────────┬───────────┘
                                 │
                                 ▼
              ┌──────────────────────────────────────┐
              │   73-Node Knowledge Graph            │
              │   Prerequisite Chain Expansion       │
              │   Adaptive Subgraph Construction     │
              └──────────────────┬───────────────────┘
                                 │
              ┌──────────────────┼──────────────────┐
              ▼                  ▼                  ▼
        ┌──────────┐      ┌──────────┐      ┌──────────┐
        │ Dijkstra │      │    A*    │      │    DP    │
        │ (≤3 req) │      │ (4–8 req)│      │ (9+ req) │
        └──────────┘      └──────────┘      └──────────┘
                                 │
                                 ▼
              ┌──────────────────────────────────────┐
              │   Groq Llama 3.3 70B — Reasoning     │
              │   Traces + Deterministic Fallback     │
              └──────────────────┬───────────────────┘
                                 │
                                 ▼
              ┌──────────────────────────────────────┐
              │   PathPlan: ordered modules +        │
              │   hours_saved + confidence +         │
              │   domain_info + radar/timeline data  │
              └──────────────────────────────────────┘
```

---

## 2. Algorithm Deep Dive <a name="algorithms"></a>

### 2.0 NLP Skill Extraction Pipeline (v3.0)
**File:** `backend/nlp_extractor.py`

Skill detection is a **4-layer pipeline**. The LLM is intentionally constrained to scoring only — it never decides what skills exist.

```
Text input
  │
  ├─ Layer 1 — spaCy NER (en_core_web_sm)
  │    Recognises ORG / PRODUCT / WORK_OF_ART entities.
  │    Catches brand names and framework names spaCy was trained on.
  │
  ├─ Layer 2 — spaCy PhraseMatcher  ← primary extraction layer
  │    Exact-match against 1 400-term SKILL_LEXICON spanning 12 domains.
  │    Deterministic and auditable — same input always yields same output.
  │
  ├─ Layer 3 — BERT NER (dslim/bert-base-NER)
  │    Token-classification fallback for skills not in the lexicon.
  │    Chunked at 400 words to respect BERT's 512-token context limit.
  │    Gracefully skipped if model is unavailable.
  │
  └─ Merge + deduplicate
       Source priority: phrase_matcher > spacy_ner > bert_ner
       Cap: 15 skills per document
         │
         ▼
     Layer 4 — Groq Llama 3.3 70B  (SCORING ONLY)
       Receives the NLP-extracted skill list (no raw text entity hunting).
       Returns: proficiency ∈ [0.1, 0.9], years, importance level.
       Fallback: deterministic neutral scores if Groq unavailable.
```

**Why separate extraction from scoring?**

| Concern | Approach |
|---------|----------|
| Hallucination | NLP layers only return skills that literally appear in the text |
| Auditability | PhraseMatcher is a deterministic lexicon lookup — every hit is traceable |
| Resilience | If Groq is down, pipeline still extracts skills and returns neutral scores |
| Separation of concerns | LLM's strength is judgment (how good?), not detection (is it there?) |

**Diagnostics:** `GET /api/nlp/status` — per-layer hit counts and model names from the most recent call.

---

### 2.1 BKT Proficiency Scoring Model
**File:** `backend/proficiency_scorer.py`

Converts raw resume signals into a calibrated score in [0.05, 0.98].

```
SkillScore = 0.5 × LLM_score
           + 0.5 × (Base + YearsBonus + ComplexityBonus
                    + RecencyModifier + LeadershipBonus
                    + EducationBonus + PrimaryBonus)
```

**Component weights:**

| Component | Values |
|-----------|--------|
| **Base** (from LLM evidence level) | mentioned=0.15, used=0.30, proficient=0.55, expert=0.80 |
| **YearsBonus** | `min(0.20, 0.08 × log(1 + years))` — log-scaled, cap 0.20 |
| **ComplexityBonus** | academic=0.00, personal=0.03, internship=0.05, production=0.12, scale=0.16 |
| **RecencyModifier** | ≤0.5yr: +0.05 / ≤1yr: +0.02 / ≤2yr: 0 / ≤4yr: −0.05 / ≤6yr: −0.10 / >6yr: −0.15 |
| **LeadershipBonus** | none=0.00, team=+0.05, org=+0.08 |
| **EducationBonus** | none=0.00, course=+0.03, degree=+0.08, publication=+0.12 |

**Confidence:** `min(0.95, 0.55 + n_signals × 0.08)` — more resume signals → higher confidence

**Inspiration:** Bayesian Knowledge Tracing (Corbett & Anderson, 1994). All weights and formula structure are original.

---

### 2.2 BKT Gap Classification
**File:** `backend/gap_analyzer.py`

```python
raw_gap     = max(0.0, required − current)
adjusted_gap = raw_gap × (2.0 − 0.85)    # if 0 < current < required
             = raw_gap                     # otherwise

if adjusted_gap ≤ 0.10:   action = SKIP
elif adjusted_gap ≤ 0.30: action = FAST_TRACK
else:                      action = REQUIRED
```

**BKT Slip-Factor (0.85):** Partial knowledge is often overconfident. Multiplying by `(2.0 − 0.85) = 1.15` widens the gap by 15% for candidates who have some but not full knowledge. All three thresholds are tunable via environment variables.

---

### 2.3 Path Optimisation Algorithms
**File:** `backend/optimizer.py`

The engine auto-selects based on problem size:

#### Algorithm 1 — Dijkstra (≤3 required modules)
**Objective:** Minimize total learning hours.
```
Uses: nx.dijkstra_path(G, START, END, weight="hours × difficulty_multiplier")
Edge weight: estimated_hours × DIFFICULTY_MULTIPLIER[difficulty]
DIFFICULTY_MULTIPLIER = {1: 0.75, 2: 0.90, 3: 1.00, 4: 1.20, 5: 1.45}
```
Original contribution: graph construction, edge weighting, virtual node wiring.

#### Algorithm 2 — A* with Custom Heuristic (4–8 required modules)
**Objective:** Prioritise critical JD skills — find high-importance modules first.
```
h(node) = −gap_score × 5    if action == REQUIRED and importance == "critical"
         = 0.0               otherwise
```
Negative heuristic creates a priority boost for critical-gap nodes. NetworkX provides the A* skeleton; the heuristic function is 100% original.

#### Algorithm 3 — DP Full Coverage (9+ required modules)
**Objective:** Guarantee every required module is included, ordered by dependencies.
```python
topo_order = nx.topological_sort(G)  # respects all prerequisite edges

priority(node) = (action_tier, −gap_score)
# action_tier: REQUIRED+critical=0, REQUIRED=1, FAST_TRACK=2
```
Dual-key sort within topological layers: critical gaps first, then by severity. Guarantees 100% coverage and topological validity simultaneously. Entirely original algorithm.

---

### 2.4 Prerequisite Chain Expansion
**File:** `backend/optimizer.py :: _expand_with_prerequisites()`

For every REQUIRED/FAST_TRACK skill, automatically injects missing prerequisites:

```
for each skill in gap_map where action != SKIP:
    ancestors = nx.ancestors(KNOWLEDGE_GRAPH, skill_id)  # topological order
    for each ancestor not already in gap_map:
        current_score  = scored_skills.get(ancestor, 0.0)
        required_level = max(0.4, difficulty × 0.15)
        raw_gap        = max(0, required_level − current_score)
        adj_gap        = BKT_adjust(raw_gap)
        action         = classify(adj_gap)
        gap_map[ancestor] = GapResult(...)  # inject as new module
```

Ensures the plan is always learnable — no module appears without its foundations. Entirely original algorithm.

---

### 2.5 Knowledge Graph
**File:** `backend/knowledge_graph.py`

```
73 nodes × 65 directed edges
Domains: software (18), ml (26), cloud (9), data-eng (5), security (3), product (4), general (8)

Example dependency chain:
python-basics → numpy-pandas → data-preprocessing → classical-ml
                                                          ↓
                              math-foundations → deep-learning-fundamentals → pytorch
                                                                                  ↓
                                                                            transformers → nlp-transformers → llm-fundamentals → rag
```

O*NET SOC codes are attached to relevant nodes as metadata. The graph structure (edges, difficulty scores, base_hours, prerequisite design) is entirely original — O*NET has no such graph.

---

## 3. Dataset Citations <a name="datasets"></a>

### Dataset 1 — O*NET Database
| Field | Value |
|-------|-------|
| **Name** | O*NET 28.3 (Occupational Information Network) |
| **Provider** | U.S. Department of Labor / ETA |
| **URL** | https://www.onetcenter.org/db_releases.html |
| **License** | Public Domain (U.S. Government Work) |
| **Format** | CSV/SQL/XML — 28,000+ occupations × 900 skills |

**How NeuralPath uses it:**
- O*NET SOC codes attached to 24 skill nodes as `onet_codes` metadata
- 11 domain profiles mapped to O*NET occupation groups
- Skill importance levels calibrated against O*NET's Level/Importance scales
- Skill taxonomy used as a reference for selecting the 73 graph nodes

**Original contribution on top of O*NET:**
- O*NET has NO prerequisite edges — we designed all 65 dependency edges
- O*NET has NO difficulty scores — we assigned all 1–5 difficulty ratings
- O*NET has NO learning-time estimates — we estimated all `base_hours`
- BKT scoring, gap classification, and path algorithms are entirely our own

---

### Dataset 2 — Resume Dataset (Kaggle)
| Field | Value |
|-------|-------|
| **Name** | Resume Dataset |
| **Provider** | Kaggle — Sneha Anbhawal |
| **URL** | https://www.kaggle.com/datasets/snehaanbhawal/resume-dataset/data |
| **License** | CC BY 4.0 |
| **Format** | CSV — 2,484 labelled resumes, 24 job categories |

**How NeuralPath uses it:**
- Calibrated Claude Sonnet 4 skill extraction prompt (manual review of 100 samples)
- Tuned BASE_SCORE thresholds in proficiency_scorer.py
- Validated domain_detector.py across the 24 category labels

**Validation result:** 94.2% skill extraction accuracy on 100 held-out resumes (manual review)

---

### Dataset 3 — Jobs and Job Description Dataset (Kaggle)
| Field | Value |
|-------|-------|
| **Name** | Jobs and Job Description |
| **Provider** | Kaggle — Kshitiz Regmi |
| **URL** | https://www.kaggle.com/datasets/kshitizregmi/jobs-and-job-description |
| **License** | CC BY 4.0 |
| **Format** | CSV — ~19,000 job postings with skill requirements |

**How NeuralPath uses it:**
- Tuned importance-level thresholds (critical/important/nice-to-have) via frequency analysis
- Calibrated required_level priors per skill per domain
- Validated domain detection keyword lists (F1=0.89 on 200 JD samples)

---

### Model Citations

| Model | Version | Provider | License | How Used |
|-------|---------|----------|---------|----------|
| **Groq Llama 3.3 70B** | llama-3.3-70b-versatile | Groq / Meta | Free API tier available | Skill scoring (Layer 4) + reasoning traces |
| **spaCy en_core_web_sm** | 3.8.x | Explosion AI | MIT | NLP extraction Layers 1–2: NER + PhraseMatcher |
| **BERT NER** | dslim/bert-base-NER | David S. Lim / HuggingFace | MIT | NLP extraction Layer 3: token-classification fallback |
| **NetworkX** | 3.3 | NetworkX Developers | BSD-3-Clause | Graph data structure + Dijkstra/A*/topological sort primitives |
| **pdfplumber** | 0.11.0 | Jeremy Singer-Vine | MIT | PDF text extraction |
| **FastAPI** | 0.111.0 | Sebastián Ramírez | MIT | REST API framework |
| **Pydantic** | 2.7.1 | Pydantic team | MIT | Data validation |

**No models were fine-tuned.** All models are used via API or inference pipeline with no weight updates.

---

## 4. Originality Statement <a name="originality"></a>

Per hackathon rules: *"The Adaptive Logic must be your original implementation."*

### What is 100% original NeuralPath code:

| Component | File | Originality |
|-----------|------|-------------|
| NLP Extraction Pipeline | `nlp_extractor.py` | 1 400-term lexicon, _canonicalise(), _deduplicate(), 4-layer architecture — all original |
| 73-node Skill Knowledge Graph | `knowledge_graph.py` | Nodes, edges, difficulty, hours — all original |
| BKT Proficiency Scoring formula | `proficiency_scorer.py` | Formula, weights, thresholds — all original |
| BKT Gap Classification | `gap_analyzer.py` | Slip-factor, three-tier classification — original |
| A* custom heuristic | `optimizer.py` | h(n) = −gap×5 for critical nodes — original |
| DP dual-key priority function | `optimizer.py` | (action_tier, −gap_score) sort — original |
| Prerequisite Chain Expansion | `optimizer.py` | Ancestor injection algorithm — original |
| Domain Detection scoring | `domain_detector.py` | Keywords, title weighting, confidence — original |
| ML Level Assessment | `ml_pathway.py` | 4-tier score, level thresholds — original |
| 5-track ML Curriculum | `ml_pathway.py` | Track structure, sequences — original |
| Reasoning trace structure | `reasoning.py` | 6-field trace format — original |

### What uses pre-trained models / libraries AS TOOLS:

| Tool | Used For | Our contribution on top |
|------|----------|------------------------|
| spaCy en_core_web_sm | NER entity detection (Layer 1) | PhraseMatcher patterns, _canonicalise(), 4-layer orchestration |
| dslim/bert-base-NER | Token-classification fallback (Layer 3) | Chunking loop, sub-word stripping, priority-based deduplication |
| Groq Llama 3.3 70B | Skill scoring + reasoning traces | Scoring-only constraint, deterministic fallback, 50% BKT blend |
| NetworkX | Graph storage, Dijkstra/A* primitives | Graph schema, A* heuristic, DP algorithm |
| pdfplumber | Binary PDF → text | Multi-strategy fallback parser |

---

## 5. Validation Metrics <a name="metrics"></a>

Live metrics are available at `GET /api/transparency/metrics`.

| # | Metric | Value | Dataset | Method |
|---|--------|-------|---------|--------|
| 1 | Skill Resolver Accuracy | **94.2%** | Resume Dataset (Kaggle, n=100) | Manual review against category ground truth |
| 2 | Domain Detection F1 (Macro) | **0.89** | Jobs Dataset (Kaggle, n=200) | Macro-F1 across 11 domain classes |
| 3 | Gap Classification Precision | **100%** | Synthetic ground truth (n=10) | BKT action vs known (current, required) pairs |
| 4 | Proficiency Scorer Monotonicity | **100%** | Synthetic fixtures (n=7) | Expert scores higher than fresher |
| 5 | Prerequisite Coverage | **100%** | Knowledge Graph (73 nodes) | No dangling prerequisite references |
| 6 | Avg Time Saved vs Traditional | **35–60%** | Synthetic profiles (n=5 levels) | Adaptive vs full-curriculum hours |
| 7 | Pathway Topological Validity | **100%** | Synthetic chains (n=3) | No module before its prerequisites |
| 8 | BKT Slip-Factor Correctness | **100%** | Synthetic pairs (n=5) | adj_gap > raw_gap for partial knowledge |

**Measured against:**
- O*NET 28.3 — taxonomy alignment
- Resume Dataset (Kaggle, CC BY 4.0) — skill extraction accuracy
- Jobs Dataset (Kaggle, CC BY 4.0) — domain detection F1
- Synthetic test suites — all algorithmic correctness metrics

---

## 6. API Reference <a name="api"></a>

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET`  | `/api/health` | System health + graph stats |
| `GET`  | `/api/nlp/status` | NLP extraction diagnostics (per-layer hit counts) |
| `GET`  | `/api/cache/stats` | Cache hit/miss metrics |
| `DELETE` | `/api/cache` | Clear caches |
| `GET`  | `/api/graph/stats` | 73-node graph metadata |
| `GET`  | `/api/graph/domains` | 11 O*NET-aligned domains |
| `POST` | `/api/analyze` | Full adaptive pathway (any domain) |
| `POST` | `/api/pathway/mldl` | ML/DL 5-track curriculum |
| `POST` | `/api/compare` | Side-by-side scenario comparison |
| `POST` | `/api/analytics/radar` | Skill radar chart data |
| `POST` | `/api/analytics/timeline` | Phase-based roadmap timeline |
| `POST` | `/api/analytics/savings` | Time saved vs traditional |
| `GET`  | `/api/transparency/disclosure` | **Full dataset + model citations** |
| `GET`  | `/api/transparency/metrics` | **All 8 validation metrics** |
| `GET`  | `/api/transparency/algorithms` | **Algorithm deep dive (JSON)** |

**Interactive docs:** `http://localhost:8000/api/docs`

---

## 7. Quick Start <a name="quickstart"></a>

### Local
```bash
pip install -r backend/requirements.txt
python -m spacy download en_core_web_sm   # download NLP model
cp .env.example .env          # set GROQ_API_KEY
uvicorn backend.main:app --reload --port 8000
```

### Docker
```bash
docker-compose up --build
```

### Tests
```bash
pytest backend/tests.py -v    # 129 tests (spaCy-dependent tests skip if deps absent)
```

### Verify NLP pipeline
```bash
curl http://localhost:8000/api/nlp/status | python -m json.tool
```

### Check Transparency
```bash
curl http://localhost:8000/api/transparency/disclosure | python -m json.tool
curl http://localhost:8000/api/transparency/metrics    | python -m json.tool
curl http://localhost:8000/api/transparency/algorithms | python -m json.tool
```

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `GROQ_API_KEY` | **required** | Groq API key (free at console.groq.com) |
| `GAP_SKIP_THRESHOLD` | `0.10` | Gap ≤ this → SKIP |
| `GAP_FAST_THRESHOLD` | `0.30` | Gap ≤ this → FAST_TRACK |
| `BKT_SLIP_FACTOR` | `0.85` | BKT overconfidence adjustment |

---

## File Structure

```
backend/
├── main.py               # FastAPI — 15 endpoints
├── nlp_extractor.py      # 4-layer NLP pipeline: spaCy NER + PhraseMatcher + BERT + Groq scoring
├── knowledge_graph.py    # 73-node skill graph + prereq chains
├── proficiency_scorer.py # BKT scoring: years × complexity × recency × leadership
├── gap_analyzer.py       # BKT gap → SKIP / FAST_TRACK / REQUIRED
├── optimizer.py          # Dijkstra / A* / DP path algorithms
├── reasoning.py          # Groq reasoning traces + deterministic fallback
├── domain_detector.py    # 11-domain O*NET classifier
├── ml_pathway.py         # Dedicated ML/DL 5-track curriculum
├── analytics.py          # Radar chart, timeline, time-saved builders
├── cache.py              # LRU + TTL in-memory cache
├── dataset_registry.py   # Dataset + model citations (transparency)
├── validation.py         # 8 internal validation metrics
├── skill_extractor.py    # Legacy LLM-only extractor (superseded by nlp_extractor.py)
├── embedder.py           # O*NET slug matching
├── parser.py             # PDF / DOCX / TXT text extraction
├── models.py             # Pydantic response schemas
└── tests.py              # 102-test pytest suite (14 test classes)
```

---

*Built for the AI-Adaptive Onboarding Engine Hackathon 2025*
*NLP: spaCy · dslim/bert-base-NER · Groq Llama 3.3 70B · O*NET 28.3 · NetworkX · FastAPI*
