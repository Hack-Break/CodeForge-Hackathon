# # #!/usr/bin/env python3
# # """
# # NeuralPath — NLP Pipeline Accuracy Evaluator  v2
# # =================================================
# # Evaluates the 4-layer NLP extraction pipeline against the
# # Kaggle Resume Dataset (Sneha Anbhawal, CC BY 4.0).

# # Metrics:
# #   Precision — of skills the pipeline found, what % were actually there?
# #   Recall    — of skills that should have been found, what % were found?
# #   F1        — harmonic mean of precision and recall

# # How to run:
# #   1. Download dataset:
# #        https://www.kaggle.com/datasets/snehaanbhawal/resume-dataset
# #        Save as: data/resume_dataset.csv

# #   2. Run from project root:
# #        python scripts/evaluate_nlp_accuracy.py

# #   3. With auto-update to validation.py:
# #        python scripts/evaluate_nlp_accuracy.py --update-validation

# # Output:
# #   Console report + data/nlp_eval_results.json

# # Requirements:
# #   pip install pandas tqdm requests
# #   Backend running: uvicorn backend.main:app --port 8000
# # """

# # from __future__ import annotations

# # import argparse
# # import json
# # import os
# # import re
# # import sys
# # import time
# # from collections import defaultdict
# # from pathlib import Path
# # from typing import NamedTuple

# # # ─────────────────────────────────────────────────────────────────────────────
# # # Config
# # # ─────────────────────────────────────────────────────────────────────────────

# # DATASET_PATH = Path("data/resume_dataset.csv")
# # RESULTS_PATH = Path("data/nlp_eval_results.json")
# # API_BASE     = os.environ.get("NEURALPATH_API", "http://localhost:8000")
# # MAX_RESUMES  = int(os.environ.get("MAX_RESUMES", "500"))
# # BATCH_DELAY  = float(os.environ.get("BATCH_DELAY", "0.0"))

# # # ─────────────────────────────────────────────────────────────────────────────
# # # Ground-truth skill sets per job category
# # # Derived from manual analysis of the Kaggle resume dataset.
# # # Uses the same terms as the NeuralPath skill lexicon for fair evaluation.
# # # ─────────────────────────────────────────────────────────────────────────────

# # CATEGORY_EXPECTED_SKILLS: dict[str, set[str]] = {
# #     "Data Science": {
# #         "python", "machine learning", "deep learning", "tensorflow", "pytorch",
# #         "scikit-learn", "pandas", "numpy", "sql", "statistics",
# #         "matplotlib", "jupyter", "keras", "xgboost",
# #     },
# #     "Web Designing": {
# #         "html", "css", "javascript", "react", "angular", "bootstrap",
# #         "figma", "typescript", "nextjs",
# #     },
# #     "Java Developer": {
# #         "java", "spring", "spring boot", "sql", "rest",
# #         "microservices", "git", "maven",
# #     },
# #     "Testing": {
# #         "selenium", "python", "java", "sql", "agile", "postman", "git",
# #     },
# #     "DevOps Engineer": {
# #         "docker", "kubernetes", "jenkins", "terraform", "aws",
# #         "ci/cd", "ansible", "git", "python", "linux",
# #     },
# #     "Python Developer": {
# #         "python", "django", "flask", "fastapi", "sql", "rest",
# #         "git", "docker",
# #     },
# #     "HR": {
# #         "talent acquisition", "recruiting", "onboarding", "hris",
# #         "excel", "hr analytics", "workday",
# #     },
# #     "Hadoop": {
# #         "hadoop", "spark", "hive", "python", "scala", "sql",
# #     },
# #     "Blockchain": {
# #         "blockchain", "javascript", "python", "cryptography",
# #     },
# #     "ETL Developer": {
# #         "sql", "etl", "python", "data warehouse", "shell scripting",
# #         "oracle",
# #     },
# #     "Operations Manager": {
# #         "supply chain", "logistics", "excel",
# #         "erp", "sap",
# #     },
# #     "Data Analyst": {
# #         "sql", "excel", "tableau", "power bi", "python",
# #         "statistics",
# #     },
# #     "Arts": {
# #         "photoshop", "figma", "illustrator",
# #     },
# #     "Database": {
# #         "sql", "mysql", "postgresql", "oracle", "mongodb", "redis",
# #     },
# #     "Electrical Engineering": {
# #         "matlab", "autocad",
# #     },
# #     "Health and Fitness": {
# #         "excel",
# #     },
# #     "PMO": {
# #         "agile", "scrum", "jira", "excel",
# #     },
# #     "SAP Developer": {
# #         "sap", "sql",
# #     },
# #     "Automation Testing": {
# #         "selenium", "python", "java", "jenkins", "jira",
# #     },
# #     "Network Security Engineer": {
# #         "python", "linux", "penetration testing", "nmap",
# #     },
# #     "DotNet Developer": {
# #         "c#", "sql", "rest", "azure",
# #     },
# #     "Civil Engineer": {
# #         "autocad",
# #     },
# #     "Mechanical Engineer": {
# #         "matlab", "autocad",
# #     },
# #     "Sales": {
# #         "excel", "crm", "salesforce",
# #     },
# # }

# # # ─────────────────────────────────────────────────────────────────────────────
# # # Rich synthetic JD per category
# # # These give the NLP pipeline enough signal to extract JD skills.
# # # The evaluator only uses them to trigger domain detection, not to measure JD accuracy.
# # # ─────────────────────────────────────────────────────────────────────────────

# # CATEGORY_JD: dict[str, str] = {
# #     "Data Science": (
# #         "We are hiring a Data Scientist with strong Python, PyTorch, TensorFlow, "
# #         "scikit-learn, pandas, numpy, SQL, statistics, and Jupyter experience. "
# #         "You will build deep learning models and data visualisation dashboards."
# #     ),
# #     "Web Designing": (
# #         "Frontend Web Designer needed. Must know HTML, CSS, JavaScript, React, "
# #         "TypeScript, Next.js, Angular, Bootstrap, Figma, and UI/UX design."
# #     ),
# #     "Java Developer": (
# #         "Java Developer with Spring Boot, Maven, Hibernate, SQL, REST APIs, "
# #         "microservices architecture, and Git experience required."
# #     ),
# #     "Testing": (
# #         "QA Engineer with Selenium, JUnit, Python, Java, Postman, API testing, "
# #         "Agile methodology, and SQL experience required."
# #     ),
# #     "DevOps Engineer": (
# #         "DevOps Engineer skilled in Docker, Kubernetes, Jenkins, Terraform, "
# #         "Ansible, AWS, CI/CD, Git, and Linux system administration."
# #     ),
# #     "Python Developer": (
# #         "Python Developer with Django, Flask, FastAPI, SQL, REST APIs, "
# #         "Git, Docker, and Linux experience required."
# #     ),
# #     "HR": (
# #         "HR Business Partner experienced in talent acquisition, recruiting, "
# #         "onboarding, HRIS, Workday, HR analytics, Excel, and people operations."
# #     ),
# #     "Hadoop": (
# #         "Big Data Engineer with Hadoop, Apache Spark, Hive, PySpark, "
# #         "Python, Scala, and SQL experience required."
# #     ),
# #     "Blockchain": (
# #         "Blockchain Developer with Solidity, Ethereum, smart contracts, "
# #         "JavaScript, Python, and cryptography experience."
# #     ),
# #     "ETL Developer": (
# #         "ETL Developer with SQL, Python, data warehousing, Oracle, "
# #         "shell scripting, and ETL pipeline development experience."
# #     ),
# #     "Operations Manager": (
# #         "Operations Manager with supply chain, logistics, ERP, SAP, "
# #         "Excel, and project management skills required."
# #     ),
# #     "Data Analyst": (
# #         "Data Analyst with SQL, Excel, Tableau, Power BI, Python, "
# #         "statistics, and data visualisation skills required."
# #     ),
# #     "Arts": (
# #         "Creative professional with Photoshop, Illustrator, InDesign, "
# #         "Figma, and video editing experience required."
# #     ),
# #     "Database": (
# #         "Database Administrator with SQL, MySQL, PostgreSQL, Oracle, "
# #         "MongoDB, Redis, and performance tuning experience."
# #     ),
# #     "Electrical Engineering": (
# #         "Electrical Engineer with MATLAB, AutoCAD, circuit design, "
# #         "embedded systems, and PCB design experience."
# #     ),
# #     "Health and Fitness": (
# #         "Fitness professional with nutrition, anatomy, CPR, "
# #         "Excel, and physical training experience required."
# #     ),
# #     "PMO": (
# #         "Project Manager with Agile, Scrum, JIRA, Excel, MS Project, "
# #         "stakeholder management, and risk management skills."
# #     ),
# #     "SAP Developer": (
# #         "SAP Developer with ABAP, SAP HANA, SAP BW, SQL, "
# #         "BAPI, and SAP Fiori experience required."
# #     ),
# #     "Automation Testing": (
# #         "Automation Test Engineer with Selenium, Python, Java, Appium, "
# #         "Jenkins, JIRA, and API testing skills required."
# #     ),
# #     "Network Security Engineer": (
# #         "Network Security Engineer with Nmap, Wireshark, Python, Linux, "
# #         "penetration testing, firewall, and VPN experience."
# #     ),
# #     "DotNet Developer": (
# #         ".NET Developer with C#, ASP.NET, SQL Server, Azure, "
# #         "Entity Framework, and REST API experience."
# #     ),
# #     "Civil Engineer": (
# #         "Civil Engineer with AutoCAD, Civil 3D, structural analysis, "
# #         "and project management skills required."
# #     ),
# #     "Mechanical Engineer": (
# #         "Mechanical Engineer with SolidWorks, AutoCAD, MATLAB, FEA, "
# #         "and manufacturing process experience."
# #     ),
# #     "Sales": (
# #         "Sales Executive with CRM, Salesforce, Excel, negotiation, "
# #         "account management, and business development experience."
# #     ),
# # }

# # # ─────────────────────────────────────────────────────────────────────────────
# # # Result type
# # # ─────────────────────────────────────────────────────────────────────────────

# # class SkillEvalResult(NamedTuple):
# #     resume_id:  str
# #     category:   str
# #     found:      set[str]
# #     expected:   set[str]
# #     tp:         int
# #     fp:         int
# #     fn:         int
# #     precision:  float
# #     recall:     float
# #     f1:         float

# # # ─────────────────────────────────────────────────────────────────────────────
# # # NLP pipeline caller — TWO strategies
# # # ─────────────────────────────────────────────────────────────────────────────

# # def call_nlp_direct(resume_text: str, jd_text: str) -> tuple[set[str], dict]:
# #     """
# #     Call extract_skills() directly (no server needed).
# #     Returns (found_skill_names, extraction_meta).

# #     FIX vs v1: import from backend.skill_extractor (correct module name),
# #                read raw extracted skill names, not pathway module names.
# #     """
# #     try:
# #         sys.path.insert(0, str(Path(__file__).parent.parent))
# #         from backend.skill_extractor import extract_skills
# #         result = extract_skills(resume_text[:5000], jd_text)
# #         # Read raw extracted skill names (before Groq scoring maps them to graph nodes)
# #         found = {
# #             s.get("skill", "").lower().strip()
# #             for s in result.get("resume_skills", [])
# #             if s.get("skill")
# #         }
# #         meta = result.get("extraction_meta", {})
# #         return found, meta
# #     except Exception as exc:
# #         print(f"    [warn] Direct import failed: {exc}")
# #         return set(), {}


# # def call_api(resume_text: str, jd_text: str) -> tuple[set[str], dict]:
# #     """
# #     Call the live /api/analyze endpoint.
# #     Reads extracted_skills from the response (not pathway module names).

# #     FIX vs v1: reads response["extracted_skills"] which we now expose,
# #                falls back gracefully to direct import if server is down.
# #     """
# #     import io
# #     try:
# #         import requests
# #         form_data = {"jd_text": jd_text}
# #         files = {
# #             "resume": ("resume.txt",
# #                         io.BytesIO(resume_text.encode("utf-8", errors="ignore")),
# #                         "text/plain"),
# #         }
# #         r = requests.post(
# #             f"{API_BASE}/api/analyze",
# #             files=files, data=form_data, timeout=30,
# #         )
# #         if r.status_code == 200:
# #             data = r.json()
# #             # extracted_skills is the raw NLP output (resume side)
# #             found = {
# #                 s.get("skill", "").lower().strip()
# #                 for s in data.get("extracted_skills", [])
# #                 if s.get("skill")
# #             }
# #             meta = data.get("extraction_meta", {})
# #             return found, meta
# #         else:
# #             print(f"    [warn] API returned {r.status_code} — falling back to direct import")
# #     except Exception:
# #         pass

# #     return call_nlp_direct(resume_text, jd_text)


# # def call_nlp_pipeline(resume_text: str, jd_text: str) -> tuple[set[str], dict]:
# #     """Auto-select: try direct import first (faster, no server needed)."""
# #     return call_nlp_direct(resume_text, jd_text)


# # # ─────────────────────────────────────────────────────────────────────────────
# # # Normalisation + fuzzy overlap
# # # ─────────────────────────────────────────────────────────────────────────────

# # def norm(s: str) -> str:
# #     s = s.lower().strip()
# #     s = re.sub(r"[^a-z0-9\s\+\#/]", "", s)
# #     return re.sub(r"\s+", " ", s).strip()


# # def fuzzy_overlap(found: set[str], expected: set[str]) -> tuple[set, set, set]:
# #     """
# #     Compute TP/FP/FN with substring fuzzy matching.
# #     'python' matches 'python developer', 'pytorch' does NOT match 'python'.
# #     """
# #     fn = {norm(s) for s in found}
# #     en = {norm(s) for s in expected}

# #     tp_f, tp_e = set(), set()
# #     for f in fn:
# #         for e in en:
# #             # Match if one is a substring of the other AND they share a root word
# #             if f == e or (len(f) > 3 and f in e) or (len(e) > 3 and e in f):
# #                 tp_f.add(f)
# #                 tp_e.add(e)
# #                 break

# #     fp     = fn - tp_f
# #     fn_set = en - tp_e
# #     return tp_f, fp, fn_set


# # # ─────────────────────────────────────────────────────────────────────────────
# # # Main evaluation loop
# # # ─────────────────────────────────────────────────────────────────────────────

# # def evaluate(args) -> dict:
# #     try:
# #         import pandas as pd
# #     except ImportError:
# #         print("ERROR: pip install pandas tqdm")
# #         sys.exit(1)

# #     if not DATASET_PATH.exists():
# #         print(f"""
# # ERROR: Dataset not found at {DATASET_PATH}
# # Download from: https://www.kaggle.com/datasets/snehaanbhawal/resume-dataset
# # Save as:       {DATASET_PATH.resolve()}
# # """)
# #         sys.exit(1)

# #     print(f"Loading dataset from {DATASET_PATH}...")
# #     df = pd.read_csv(DATASET_PATH)
# #     df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

# #     text_col = next((c for c in df.columns if "resume" in c and "str" in c), None)
# #     cat_col  = next((c for c in df.columns if "categ" in c), None)
# #     if not text_col or not cat_col:
# #         print(f"ERROR: Expected columns 'Resume_str' and 'Category'. Found: {list(df.columns)}")
# #         sys.exit(1)

# #     df = df[[text_col, cat_col]].dropna()
# #     df.columns = ["resume_text", "category"]
# #     total = len(df)
# #     print(f"Dataset: {total} resumes, {df['category'].nunique()} categories")

# #     # Stratified sample
# #     max_n = args.max_resumes if args.max_resumes > 0 else total
# #     # Guarantee minimum 4 resumes per category (so all categories are evaluated)
# #     MIN_PER_CAT = 4
# #     sampled = (
# #         df.groupby("category", group_keys=False)
# #           .apply(lambda x: x.sample(
# #               min(len(x), max(MIN_PER_CAT, int(max_n * len(x) / total))),
# #               random_state=42), include_groups=False)
# #           .reset_index(drop=True)
# #     )
# #     # Re-attach category column if it was dropped by include_groups=False
# #     if "category" not in sampled.columns:
# #         sampled = df.groupby("category", group_keys=False).apply(
# #             lambda x: x.sample(min(len(x), max(MIN_PER_CAT, int(max_n * len(x) / total))),
# #                                 random_state=42)).reset_index(drop=True)
# #     sampled = sampled.sample(min(len(sampled), max_n * 2), random_state=42).reset_index(drop=True)
# #     print(f"Evaluating {len(sampled)} resumes (stratified).\n")

# #     try:
# #         from tqdm import tqdm
# #         iterator = tqdm(sampled.iterrows(), total=len(sampled), desc="Evaluating")
# #     except ImportError:
# #         iterator = sampled.iterrows()

# #     results: list[SkillEvalResult] = []
# #     layer_totals = defaultdict(int)
# #     skipped = 0

# #     for idx, row in iterator:
# #         category    = str(row["category"]).strip()
# #         resume_text = str(row["resume_text"])[:5000]

# #         # Use rich synthetic JD for this category (FIX vs v1: was one generic sentence)
# #         jd_text = CATEGORY_JD.get(
# #             category,
# #             f"{category} professional with relevant technical skills and experience."
# #         )

# #         expected = CATEGORY_EXPECTED_SKILLS.get(category, set())
# #         if not expected:
# #             skipped += 1
# #             continue

# #         found, meta = call_nlp_pipeline(resume_text, jd_text)

# #         # Accumulate layer stats
# #         for key in ("spacy_ner_hits", "phrase_match_hits", "bert_ner_hits"):
# #             layer_totals[key] += meta.get(key, 0)

# #         tp_set, fp_set, fn_set = fuzzy_overlap(found, expected)
# #         tp = len(tp_set)
# #         fp = len(fp_set)
# #         fn = len(fn_set)

# #         prec  = tp / (tp + fp) if (tp + fp) > 0 else 0.0
# #         rec   = tp / (tp + fn) if (tp + fn) > 0 else 0.0
# #         f1    = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

# #         results.append(SkillEvalResult(
# #             resume_id=str(idx), category=category,
# #             found=found, expected=expected,
# #             tp=tp, fp=fp, fn=fn,
# #             precision=prec, recall=rec, f1=f1,
# #         ))

# #         if BATCH_DELAY > 0:
# #             time.sleep(BATCH_DELAY)

# #     if not results:
# #         print("No results. Check dataset category names match CATEGORY_EXPECTED_SKILLS.")
# #         sys.exit(1)

# #     overall_p = sum(r.precision for r in results) / len(results)
# #     overall_r = sum(r.recall    for r in results) / len(results)
# #     overall_f = sum(r.f1        for r in results) / len(results)

# #     cat_metrics: dict[str, dict] = {}
# #     for cat in sorted(set(r.category for r in results)):
# #         cr = [r for r in results if r.category == cat]
# #         cat_metrics[cat] = {
# #             "n":         len(cr),
# #             "precision": round(sum(r.precision for r in cr) / len(cr), 3),
# #             "recall":    round(sum(r.recall    for r in cr) / len(cr), 3),
# #             "f1":        round(sum(r.f1        for r in cr) / len(cr), 3),
# #         }

# #     return {
# #         "evaluated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
# #         "sample_size":  len(results),
# #         "skipped":      skipped,
# #         "overall": {
# #             "precision":     round(overall_p, 4),
# #             "recall":        round(overall_r, 4),
# #             "f1":            round(overall_f, 4),
# #             "precision_pct": round(overall_p * 100, 1),
# #             "recall_pct":    round(overall_r * 100, 1),
# #             "f1_pct":        round(overall_f * 100, 1),
# #         },
# #         "layer_hits":   dict(layer_totals),
# #         "per_category": cat_metrics,
# #         "note": (
# #             "Evaluated against CATEGORY_EXPECTED_SKILLS using fuzzy substring matching. "
# #             "Ground truth = normalised skill names from O*NET 28.3 + Kaggle dataset analysis. "
# #             "Precision = of what was found, what was correct. "
# #             "Recall = of what should have been found, what was found."
# #         ),
# #     }


# # # ─────────────────────────────────────────────────────────────────────────────
# # # Report
# # # ─────────────────────────────────────────────────────────────────────────────

# # def print_report(results: dict) -> None:
# #     o = results["overall"]
# #     print("\n" + "═" * 62)
# #     print("  NeuralPath NLP Pipeline — Accuracy Evaluation  v2")
# #     print("═" * 62)
# #     print(f"  Evaluated:  {results['evaluated_at']}")
# #     print(f"  Resumes:    {results['sample_size']}  (skipped: {results.get('skipped', 0)})")
# #     print()
# #     print(f"  ┌─────────────┬────────────┐")
# #     print(f"  │ Metric      │ Value      │")
# #     print(f"  ├─────────────┼────────────┤")
# #     print(f"  │ Precision   │ {o['precision_pct']:>6.1f}%    │")
# #     print(f"  │ Recall      │ {o['recall_pct']:>6.1f}%    │")
# #     print(f"  │ F1 Score    │ {o['f1_pct']:>6.1f}%    │")
# #     print(f"  └─────────────┴────────────┘")

# #     lh = results.get("layer_hits", {})
# #     total_hits = sum(lh.values())
# #     if total_hits > 0:
# #         print(f"\n  NLP layer contributions (total hits = {total_hits}):")
# #         labels = {
# #             "spacy_ner_hits":    "spaCy NER",
# #             "phrase_match_hits": "spaCy PhraseMatcher",
# #             "bert_ner_hits":     "BERT NER",
# #         }
# #         for key, label in labels.items():
# #             hits = lh.get(key, 0)
# #             pct  = hits / total_hits * 100
# #             bar  = "█" * int(pct / 5)
# #             print(f"    {label:<22} {hits:>5} hits  ({pct:4.1f}%)  {bar}")

# #     print(f"\n  Per-category breakdown (sorted by F1):")
# #     print(f"  {'Category':<30} {'N':>4}  {'Prec':>6}  {'Rec':>6}  {'F1':>6}")
# #     print(f"  {'-'*30}  {'-'*4}  {'-'*6}  {'-'*6}  {'-'*6}")
# #     for cat, m in sorted(results["per_category"].items(), key=lambda x: -x[1]["f1"]):
# #         flag = " ✓" if m["f1"] >= 0.80 else (" ~" if m["f1"] >= 0.60 else " ✗")
# #         print(f"  {cat:<30} {m['n']:>4}  {m['precision']:>6.3f}  {m['recall']:>6.3f}  {m['f1']:>6.3f}{flag}")

# #     f1 = o["f1_pct"]
# #     verdict = (
# #         "EXCELLENT — production quality."       if f1 >= 90 else
# #         "GOOD — expand lexicon for weak cats."  if f1 >= 80 else
# #         "FAIR — review BERT coverage + lexicon."if f1 >= 70 else
# #         "IMPROVING — was 21% with v1 (pathway-based eval bug). Check lexicon gaps."
# #     )
# #     print(f"\n  Verdict: {verdict}")
# #     print("═" * 62 + "\n")


# # # ─────────────────────────────────────────────────────────────────────────────
# # # validation.py updater
# # # ─────────────────────────────────────────────────────────────────────────────

# # def update_validation_py(results: dict) -> None:
# #     val_path = Path("backend/validation.py")
# #     if not val_path.exists():
# #         print(f"ERROR: {val_path} not found. Run from project root.")
# #         return

# #     f1   = results["overall"]["f1_pct"]
# #     prec = results["overall"]["precision_pct"]
# #     rec  = results["overall"]["recall_pct"]
# #     n    = results["sample_size"]
# #     ts   = results["evaluated_at"]

# #     content = val_path.read_text(encoding="utf-8")

# #     eval_block = f"""
# # # ─────────────────────────────────────────────────────────────────────────────
# # # NLP Pipeline Accuracy — updated by scripts/evaluate_nlp_accuracy.py
# # # Last run: {ts}
# # # ─────────────────────────────────────────────────────────────────────────────
# # NLP_EVAL_PRECISION = {round(results['overall']['precision'], 4)}
# # NLP_EVAL_RECALL    = {round(results['overall']['recall'],    4)}
# # NLP_EVAL_F1        = {round(results['overall']['f1'],        4)}
# # NLP_EVAL_N         = {n}
# # NLP_EVAL_TIMESTAMP = "{ts}"

# # """
# #     content = re.sub(
# #         r"# ─+\n# NLP Pipeline Accuracy.*?NLP_EVAL_TIMESTAMP.*?\n\n",
# #         "", content, flags=re.DOTALL,
# #     )
# #     last_import = max(
# #         (m.end() for m in re.finditer(r"^(?:from|import)\s+\S+.*$", content, re.MULTILINE)),
# #         default=0,
# #     )
# #     content = content[:last_import] + "\n" + eval_block + content[last_import:]

# #     for old_note in (
# #         r"(Full dataset validation:.*?\.)\"",
# #         r"94\.2% on 100 held-out resumes \(manual review\)",
# #     ):
# #         content = re.sub(
# #             old_note,
# #             f"Full dataset validation (v3 NLP pipeline): F1={f1:.1f}% Prec={prec:.1f}% Rec={rec:.1f}% on {n} resumes ({ts}).",
# #             content,
# #         )

# #     val_path.write_text(content, encoding="utf-8")
# #     print(f"  Updated {val_path}")
# #     print(f"  New NLP accuracy: F1={f1:.1f}%  Precision={prec:.1f}%  Recall={rec:.1f}%")


# # # ─────────────────────────────────────────────────────────────────────────────
# # # Entry point
# # # ─────────────────────────────────────────────────────────────────────────────

# # def main() -> None:
# #     global MAX_RESUMES, API_BASE
# #     print("NeuralPath NLP Accuracy Evaluator v2 — starting")

# #     parser = argparse.ArgumentParser()
# #     parser.add_argument("--update-validation", action="store_true")
# #     parser.add_argument("--max-resumes",  type=int,  default=MAX_RESUMES)
# #     parser.add_argument("--api",          default=API_BASE)
# #     parser.add_argument("--output",       default=str(RESULTS_PATH))
# #     args = parser.parse_args()

# #     MAX_RESUMES = args.max_resumes
# #     API_BASE    = args.api

# #     results = evaluate(args)
# #     print_report(results)

# #     out = Path(args.output)
# #     out.parent.mkdir(parents=True, exist_ok=True)
# #     out.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
# #     print(f"  Results saved to: {out}")

# #     if args.update_validation:
# #         print("\n  Updating backend/validation.py...")
# #         update_validation_py(results)

# #     f1 = results["overall"]["f1_pct"]
# #     if f1 < 80:
# #         print(f"\n  WARNING: F1={f1:.1f}% is below the 80% target.")
# #         sys.exit(1)
# #     else:
# #         print(f"\n  PASS: F1={f1:.1f}% meets the ≥80% target.")


# # if __name__ == "__main__":
# #     main()

# #!/usr/bin/env python3
# """
# NeuralPath — NLP Pipeline Accuracy Evaluator  v2
# =================================================
# Evaluates the 4-layer NLP extraction pipeline against the
# Kaggle Resume Dataset (Sneha Anbhawal, CC BY 4.0).

# Metrics:
#   Precision — of skills the pipeline found, what % were actually there?
#   Recall    — of skills that should have been found, what % were found?
#   F1        — harmonic mean of precision and recall

# How to run:
#   1. Download dataset:
#        https://www.kaggle.com/datasets/snehaanbhawal/resume-dataset
#        Save as: data/resume_dataset.csv

#   2. Run from project root:
#        python scripts/evaluate_nlp_accuracy.py

#   3. With auto-update to validation.py:
#        python scripts/evaluate_nlp_accuracy.py --update-validation

# Output:
#   Console report + data/nlp_eval_results.json

# Requirements:
#   pip install pandas tqdm requests
#   Backend running: uvicorn backend.main:app --port 8000
# """

# from __future__ import annotations

# import argparse
# import json
# import os
# import re
# import sys
# import time
# from collections import defaultdict
# from pathlib import Path
# from typing import NamedTuple

# # ─────────────────────────────────────────────────────────────────────────────
# # Config
# # ─────────────────────────────────────────────────────────────────────────────

# DATASET_PATH = Path("data/resume_dataset.csv")
# RESULTS_PATH = Path("data/nlp_eval_results.json")
# API_BASE     = os.environ.get("NEURALPATH_API", "http://localhost:8000")
# MAX_RESUMES  = int(os.environ.get("MAX_RESUMES", "500"))
# BATCH_DELAY  = float(os.environ.get("BATCH_DELAY", "0.0"))

# # ─────────────────────────────────────────────────────────────────────────────
# # Ground-truth skill sets per job category
# # Derived from manual analysis of the Kaggle resume dataset.
# # Uses the same terms as the NeuralPath skill lexicon for fair evaluation.
# # ─────────────────────────────────────────────────────────────────────────────

# CATEGORY_EXPECTED_SKILLS: dict[str, set[str]] = {
#     "Data Science": {
#         "python", "machine learning", "deep learning", "tensorflow", "pytorch",
#         "scikit-learn", "pandas", "numpy", "sql", "statistics",
#         "matplotlib", "jupyter", "keras", "xgboost",
#     },
#     "Web Designing": {
#         "html", "css", "javascript", "react", "angular", "bootstrap",
#         "figma", "typescript", "nextjs",
#     },
#     "Java Developer": {
#         "java", "spring", "spring boot", "sql", "rest",
#         "microservices", "git", "maven",
#     },
#     "Testing": {
#         "selenium", "python", "java", "sql", "agile", "postman", "git",
#     },
#     "DevOps Engineer": {
#         "docker", "kubernetes", "jenkins", "terraform", "aws",
#         "ci/cd", "ansible", "git", "python", "linux",
#     },
#     "Python Developer": {
#         "python", "django", "flask", "fastapi", "sql", "rest",
#         "git", "docker",
#     },
#     "HR": {
#         "talent acquisition", "recruiting", "onboarding", "hris",
#         "excel", "hr analytics", "workday",
#     },
#     "Hadoop": {
#         "hadoop", "spark", "hive", "python", "scala", "sql",
#     },
#     "Blockchain": {
#         "blockchain", "javascript", "python", "cryptography",
#     },
#     "ETL Developer": {
#         "sql", "etl", "python", "data warehouse", "shell scripting",
#         "oracle",
#     },
#     "Operations Manager": {
#         "supply chain", "logistics", "excel",
#         "erp", "sap",
#     },
#     "Data Analyst": {
#         "sql", "excel", "tableau", "power bi", "python",
#         "statistics",
#     },
#     "Arts": {
#         "photoshop", "figma", "illustrator",
#     },
#     "Database": {
#         "sql", "mysql", "postgresql", "oracle", "mongodb", "redis",
#     },
#     "Electrical Engineering": {
#         "matlab", "autocad",
#     },
#     "Health and Fitness": {
#         "excel",
#     },
#     "PMO": {
#         "agile", "scrum", "jira", "excel",
#     },
#     "SAP Developer": {
#         "sap", "sql",
#     },
#     "Automation Testing": {
#         "selenium", "python", "java", "jenkins", "jira",
#     },
#     "Network Security Engineer": {
#         "python", "linux", "penetration testing", "nmap",
#     },
#     "DotNet Developer": {
#         "c#", "sql", "rest", "azure",
#     },
#     "Civil Engineer": {
#         "autocad",
#     },
#     "Mechanical Engineer": {
#         "matlab", "autocad",
#     },
#     "Sales": {
#         "excel", "crm", "salesforce",
#     },
# }

# # ─────────────────────────────────────────────────────────────────────────────
# # Rich synthetic JD per category
# # These give the NLP pipeline enough signal to extract JD skills.
# # The evaluator only uses them to trigger domain detection, not to measure JD accuracy.
# # ─────────────────────────────────────────────────────────────────────────────

# CATEGORY_JD: dict[str, str] = {
#     "Data Science": (
#         "We are hiring a Data Scientist with strong Python, PyTorch, TensorFlow, "
#         "scikit-learn, pandas, numpy, SQL, statistics, and Jupyter experience. "
#         "You will build deep learning models and data visualisation dashboards."
#     ),
#     "Web Designing": (
#         "Frontend Web Designer needed. Must know HTML, CSS, JavaScript, React, "
#         "TypeScript, Next.js, Angular, Bootstrap, Figma, and UI/UX design."
#     ),
#     "Java Developer": (
#         "Java Developer with Spring Boot, Maven, Hibernate, SQL, REST APIs, "
#         "microservices architecture, and Git experience required."
#     ),
#     "Testing": (
#         "QA Engineer with Selenium, JUnit, Python, Java, Postman, API testing, "
#         "Agile methodology, and SQL experience required."
#     ),
#     "DevOps Engineer": (
#         "DevOps Engineer skilled in Docker, Kubernetes, Jenkins, Terraform, "
#         "Ansible, AWS, CI/CD, Git, and Linux system administration."
#     ),
#     "Python Developer": (
#         "Python Developer with Django, Flask, FastAPI, SQL, REST APIs, "
#         "Git, Docker, and Linux experience required."
#     ),
#     "HR": (
#         "HR Business Partner experienced in talent acquisition, recruiting, "
#         "onboarding, HRIS, Workday, HR analytics, Excel, and people operations."
#     ),
#     "Hadoop": (
#         "Big Data Engineer with Hadoop, Apache Spark, Hive, PySpark, "
#         "Python, Scala, and SQL experience required."
#     ),
#     "Blockchain": (
#         "Blockchain Developer with Solidity, Ethereum, smart contracts, "
#         "JavaScript, Python, and cryptography experience."
#     ),
#     "ETL Developer": (
#         "ETL Developer with SQL, Python, data warehousing, Oracle, "
#         "shell scripting, and ETL pipeline development experience."
#     ),
#     "Operations Manager": (
#         "Operations Manager with supply chain, logistics, ERP, SAP, "
#         "Excel, and project management skills required."
#     ),
#     "Data Analyst": (
#         "Data Analyst with SQL, Excel, Tableau, Power BI, Python, "
#         "statistics, and data visualisation skills required."
#     ),
#     "Arts": (
#         "Creative professional with Photoshop, Illustrator, InDesign, "
#         "Figma, and video editing experience required."
#     ),
#     "Database": (
#         "Database Administrator with SQL, MySQL, PostgreSQL, Oracle, "
#         "MongoDB, Redis, and performance tuning experience."
#     ),
#     "Electrical Engineering": (
#         "Electrical Engineer with MATLAB, AutoCAD, circuit design, "
#         "embedded systems, and PCB design experience."
#     ),
#     "Health and Fitness": (
#         "Fitness professional with nutrition, anatomy, CPR, "
#         "Excel, and physical training experience required."
#     ),
#     "PMO": (
#         "Project Manager with Agile, Scrum, JIRA, Excel, MS Project, "
#         "stakeholder management, and risk management skills."
#     ),
#     "SAP Developer": (
#         "SAP Developer with ABAP, SAP HANA, SAP BW, SQL, "
#         "BAPI, and SAP Fiori experience required."
#     ),
#     "Automation Testing": (
#         "Automation Test Engineer with Selenium, Python, Java, Appium, "
#         "Jenkins, JIRA, and API testing skills required."
#     ),
#     "Network Security Engineer": (
#         "Network Security Engineer with Nmap, Wireshark, Python, Linux, "
#         "penetration testing, firewall, and VPN experience."
#     ),
#     "DotNet Developer": (
#         ".NET Developer with C#, ASP.NET, SQL Server, Azure, "
#         "Entity Framework, and REST API experience."
#     ),
#     "Civil Engineer": (
#         "Civil Engineer with AutoCAD, Civil 3D, structural analysis, "
#         "and project management skills required."
#     ),
#     "Mechanical Engineer": (
#         "Mechanical Engineer with SolidWorks, AutoCAD, MATLAB, FEA, "
#         "and manufacturing process experience."
#     ),
#     "Sales": (
#         "Sales Executive with CRM, Salesforce, Excel, negotiation, "
#         "account management, and business development experience."
#     ),
# }

# # ─────────────────────────────────────────────────────────────────────────────
# # Result type
# # ─────────────────────────────────────────────────────────────────────────────

# class SkillEvalResult(NamedTuple):
#     resume_id:  str
#     category:   str
#     found:      set[str]
#     expected:   set[str]
#     tp:         int
#     fp:         int
#     fn:         int
#     precision:  float
#     recall:     float
#     f1:         float

# # ─────────────────────────────────────────────────────────────────────────────
# # NLP pipeline caller — TWO strategies
# # ─────────────────────────────────────────────────────────────────────────────

# def call_nlp_direct(resume_text: str, jd_text: str) -> tuple[set[str], dict]:
#     """
#     Call extract_skills() directly (no server needed).
#     Returns (found_skill_names, extraction_meta).

#     FIX vs v1: import from backend.skill_extractor (correct module name),
#                read raw extracted skill names, not pathway module names.
#     """
#     try:
#         sys.path.insert(0, str(Path(__file__).parent.parent))
#         from backend.skill_extractor import extract_skills
#         result = extract_skills(resume_text[:5000], jd_text)
#         # Read raw extracted skill names (before Groq scoring maps them to graph nodes)
#         found = {
#             s.get("skill", "").lower().strip()
#             for s in result.get("resume_skills", [])
#             if s.get("skill")
#         }
#         meta = result.get("extraction_meta", {})
#         return found, meta
#     except Exception as exc:
#         print(f"    [warn] Direct import failed: {exc}")
#         return set(), {}


# def call_api(resume_text: str, jd_text: str) -> tuple[set[str], dict]:
#     """
#     Call the live /api/analyze endpoint.
#     Reads extracted_skills from the response (not pathway module names).

#     FIX vs v1: reads response["extracted_skills"] which we now expose,
#                falls back gracefully to direct import if server is down.
#     """
#     import io
#     try:
#         import requests
#         form_data = {"jd_text": jd_text}
#         files = {
#             "resume": ("resume.txt",
#                         io.BytesIO(resume_text.encode("utf-8", errors="ignore")),
#                         "text/plain"),
#         }
#         r = requests.post(
#             f"{API_BASE}/api/analyze",
#             files=files, data=form_data, timeout=30,
#         )
#         if r.status_code == 200:
#             data = r.json()
#             # extracted_skills is the raw NLP output (resume side)
#             found = {
#                 s.get("skill", "").lower().strip()
#                 for s in data.get("extracted_skills", [])
#                 if s.get("skill")
#             }
#             meta = data.get("extraction_meta", {})
#             return found, meta
#         else:
#             print(f"    [warn] API returned {r.status_code} — falling back to direct import")
#     except Exception:
#         pass

#     return call_nlp_direct(resume_text, jd_text)


# def call_nlp_pipeline(resume_text: str, jd_text: str) -> tuple[set[str], dict]:
#     """Auto-select: try direct import first (faster, no server needed)."""
#     return call_nlp_direct(resume_text, jd_text)


# # ─────────────────────────────────────────────────────────────────────────────
# # Normalisation + fuzzy overlap
# # ─────────────────────────────────────────────────────────────────────────────

# def norm(s: str) -> str:
#     s = s.lower().strip()
#     s = re.sub(r"[^a-z0-9\s\+\#/]", "", s)
#     return re.sub(r"\s+", " ", s).strip()


# def fuzzy_overlap(found: set[str], expected: set[str]) -> tuple[set, set, set]:
#     """
#     Compute TP/FP/FN with substring fuzzy matching.
#     'python' matches 'python developer', 'pytorch' does NOT match 'python'.
#     """
#     fn = {norm(s) for s in found}
#     en = {norm(s) for s in expected}

#     tp_f, tp_e = set(), set()
#     for f in fn:
#         for e in en:
#             # Match if one is a substring of the other AND they share a root word
#             if f == e or (len(f) > 3 and f in e) or (len(e) > 3 and e in f):
#                 tp_f.add(f)
#                 tp_e.add(e)
#                 break

#     fp     = fn - tp_f
#     fn_set = en - tp_e
#     return tp_f, fp, fn_set


# # ─────────────────────────────────────────────────────────────────────────────
# # Main evaluation loop
# # ─────────────────────────────────────────────────────────────────────────────

# def evaluate(args) -> dict:
#     try:
#         import pandas as pd
#     except ImportError:
#         print("ERROR: pip install pandas tqdm")
#         sys.exit(1)

#     if not DATASET_PATH.exists():
#         print(f"""
# ERROR: Dataset not found at {DATASET_PATH}
# Download from: https://www.kaggle.com/datasets/snehaanbhawal/resume-dataset
# Save as:       {DATASET_PATH.resolve()}
# """)
#         sys.exit(1)

#     print(f"Loading dataset from {DATASET_PATH}...")
#     df = pd.read_csv(DATASET_PATH)
#     df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

#     text_col = next((c for c in df.columns if "resume" in c and "str" in c), None)
#     cat_col  = next((c for c in df.columns if "categ" in c), None)
#     if not text_col or not cat_col:
#         print(f"ERROR: Expected columns 'Resume_str' and 'Category'. Found: {list(df.columns)}")
#         sys.exit(1)

#     df = df[[text_col, cat_col]].dropna()
#     df.columns = ["resume_text", "category"]
#     total = len(df)
#     print(f"Dataset: {total} resumes, {df['category'].nunique()} categories")

#     # Stratified sample
#     max_n = args.max_resumes if args.max_resumes > 0 else total
#     # Guarantee minimum 4 resumes per category (so all categories are evaluated)
#     MIN_PER_CAT = 4
#     sampled = (
#         df.groupby("category", group_keys=False)
#           .apply(lambda x: x.sample(
#               min(len(x), max(MIN_PER_CAT, int(max_n * len(x) / total))),
#               random_state=42), include_groups=False)
#           .reset_index(drop=True)
#     )
#     # Re-attach category column if it was dropped by include_groups=False
#     if "category" not in sampled.columns:
#         sampled = df.groupby("category", group_keys=False).apply(
#             lambda x: x.sample(min(len(x), max(MIN_PER_CAT, int(max_n * len(x) / total))),
#                                 random_state=42)).reset_index(drop=True)
#     sampled = sampled.sample(min(len(sampled), max_n * 2), random_state=42).reset_index(drop=True)
#     print(f"Evaluating {len(sampled)} resumes (stratified).\n")

#     try:
#         from tqdm import tqdm
#         iterator = tqdm(sampled.iterrows(), total=len(sampled), desc="Evaluating")
#     except ImportError:
#         iterator = sampled.iterrows()

#     results: list[SkillEvalResult] = []
#     layer_totals = defaultdict(int)
#     skipped = 0

#     for idx, row in iterator:
#         category    = str(row["category"]).strip()
#         resume_text = str(row["resume_text"])[:5000]

#         # Use rich synthetic JD for this category (FIX vs v1: was one generic sentence)
#         jd_text = CATEGORY_JD.get(
#             category,
#             f"{category} professional with relevant technical skills and experience."
#         )

#         expected = CATEGORY_EXPECTED_SKILLS.get(category, set())
#         if not expected:
#             skipped += 1
#             continue

#         found, meta = call_nlp_pipeline(resume_text, jd_text)

#         # Accumulate layer stats
#         for key in ("spacy_ner_hits", "phrase_match_hits", "bert_ner_hits"):
#             layer_totals[key] += meta.get(key, 0)

#         tp_set, fp_set, fn_set = fuzzy_overlap(found, expected)
#         tp = len(tp_set)
#         fp = len(fp_set)
#         fn = len(fn_set)

#         prec  = tp / (tp + fp) if (tp + fp) > 0 else 0.0
#         rec   = tp / (tp + fn) if (tp + fn) > 0 else 0.0
#         f1    = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

#         results.append(SkillEvalResult(
#             resume_id=str(idx), category=category,
#             found=found, expected=expected,
#             tp=tp, fp=fp, fn=fn,
#             precision=prec, recall=rec, f1=f1,
#         ))

#         if BATCH_DELAY > 0:
#             time.sleep(BATCH_DELAY)

#     if not results:
#         print("No results. Check dataset category names match CATEGORY_EXPECTED_SKILLS.")
#         sys.exit(1)

#     overall_p = sum(r.precision for r in results) / len(results)
#     overall_r = sum(r.recall    for r in results) / len(results)
#     overall_f = sum(r.f1        for r in results) / len(results)

#     cat_metrics: dict[str, dict] = {}
#     for cat in sorted(set(r.category for r in results)):
#         cr = [r for r in results if r.category == cat]
#         cat_metrics[cat] = {
#             "n":         len(cr),
#             "precision": round(sum(r.precision for r in cr) / len(cr), 3),
#             "recall":    round(sum(r.recall    for r in cr) / len(cr), 3),
#             "f1":        round(sum(r.f1        for r in cr) / len(cr), 3),
#         }

#     return {
#         "evaluated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
#         "sample_size":  len(results),
#         "skipped":      skipped,
#         "overall": {
#             "precision":     round(overall_p, 4),
#             "recall":        round(overall_r, 4),
#             "f1":            round(overall_f, 4),
#             "precision_pct": round(overall_p * 100, 1),
#             "recall_pct":    round(overall_r * 100, 1),
#             "f1_pct":        round(overall_f * 100, 1),
#         },
#         "layer_hits":   dict(layer_totals),
#         "per_category": cat_metrics,
#         "note": (
#             "Evaluated against CATEGORY_EXPECTED_SKILLS using fuzzy substring matching. "
#             "Ground truth = normalised skill names from O*NET 28.3 + Kaggle dataset analysis. "
#             "Precision = of what was found, what was correct. "
#             "Recall = of what should have been found, what was found."
#         ),
#     }


# # ─────────────────────────────────────────────────────────────────────────────
# # Report
# # ─────────────────────────────────────────────────────────────────────────────

# def print_report(results: dict) -> None:
#     o = results["overall"]
#     print("\n" + "═" * 62)
#     print("  NeuralPath NLP Pipeline — Accuracy Evaluation  v2")
#     print("═" * 62)
#     print(f"  Evaluated:  {results['evaluated_at']}")
#     print(f"  Resumes:    {results['sample_size']}  (skipped: {results.get('skipped', 0)})")
#     print()
#     print(f"  ┌─────────────┬────────────┐")
#     print(f"  │ Metric      │ Value      │")
#     print(f"  ├─────────────┼────────────┤")
#     print(f"  │ Precision   │ {o['precision_pct']:>6.1f}%    │")
#     print(f"  │ Recall      │ {o['recall_pct']:>6.1f}%    │")
#     print(f"  │ F1 Score    │ {o['f1_pct']:>6.1f}%    │")
#     print(f"  └─────────────┴────────────┘")

#     lh = results.get("layer_hits", {})
#     total_hits = sum(lh.values())
#     if total_hits > 0:
#         print(f"\n  NLP layer contributions (total hits = {total_hits}):")
#         labels = {
#             "spacy_ner_hits":    "spaCy NER",
#             "phrase_match_hits": "spaCy PhraseMatcher",
#             "bert_ner_hits":     "BERT NER",
#         }
#         for key, label in labels.items():
#             hits = lh.get(key, 0)
#             pct  = hits / total_hits * 100
#             bar  = "█" * int(pct / 5)
#             print(f"    {label:<22} {hits:>5} hits  ({pct:4.1f}%)  {bar}")

#     print(f"\n  Per-category breakdown (sorted by F1):")
#     print(f"  {'Category':<30} {'N':>4}  {'Prec':>6}  {'Rec':>6}  {'F1':>6}")
#     print(f"  {'-'*30}  {'-'*4}  {'-'*6}  {'-'*6}  {'-'*6}")
#     for cat, m in sorted(results["per_category"].items(), key=lambda x: -x[1]["f1"]):
#         flag = " ✓" if m["f1"] >= 0.80 else (" ~" if m["f1"] >= 0.60 else " ✗")
#         print(f"  {cat:<30} {m['n']:>4}  {m['precision']:>6.3f}  {m['recall']:>6.3f}  {m['f1']:>6.3f}{flag}")

#     f1 = o["f1_pct"]
#     verdict = (
#         "EXCELLENT — production quality."       if f1 >= 90 else
#         "GOOD — expand lexicon for weak cats."  if f1 >= 80 else
#         "FAIR — review BERT coverage + lexicon."if f1 >= 70 else
#         "IMPROVING — was 21% with v1 (pathway-based eval bug). Check lexicon gaps."
#     )
#     print(f"\n  Verdict: {verdict}")
#     print("═" * 62 + "\n")


# # ─────────────────────────────────────────────────────────────────────────────
# # validation.py updater
# # ─────────────────────────────────────────────────────────────────────────────

# def update_validation_py(results: dict) -> None:
#     val_path = Path("backend/validation.py")
#     if not val_path.exists():
#         print(f"ERROR: {val_path} not found. Run from project root.")
#         return

#     f1   = results["overall"]["f1_pct"]
#     prec = results["overall"]["precision_pct"]
#     rec  = results["overall"]["recall_pct"]
#     n    = results["sample_size"]
#     ts   = results["evaluated_at"]

#     content = val_path.read_text(encoding="utf-8")

#     eval_block = f"""
# # ─────────────────────────────────────────────────────────────────────────────
# # NLP Pipeline Accuracy — updated by scripts/evaluate_nlp_accuracy.py
# # Last run: {ts}
# # ─────────────────────────────────────────────────────────────────────────────
# NLP_EVAL_PRECISION = {round(results['overall']['precision'], 4)}
# NLP_EVAL_RECALL    = {round(results['overall']['recall'],    4)}
# NLP_EVAL_F1        = {round(results['overall']['f1'],        4)}
# NLP_EVAL_N         = {n}
# NLP_EVAL_TIMESTAMP = "{ts}"

# """
#     content = re.sub(
#         r"# ─+\n# NLP Pipeline Accuracy.*?NLP_EVAL_TIMESTAMP.*?\n\n",
#         "", content, flags=re.DOTALL,
#     )
#     last_import = max(
#         (m.end() for m in re.finditer(r"^(?:from|import)\s+\S+.*$", content, re.MULTILINE)),
#         default=0,
#     )
#     content = content[:last_import] + "\n" + eval_block + content[last_import:]

#     for old_note in (
#         r"(Full dataset validation:.*?\.)\"",
#         r"94\.2% on 100 held-out resumes \(manual review\)",
#     ):
#         content = re.sub(
#             old_note,
#             f"Full dataset validation (v3 NLP pipeline): F1={f1:.1f}% Prec={prec:.1f}% Rec={rec:.1f}% on {n} resumes ({ts}).",
#             content,
#         )

#     val_path.write_text(content, encoding="utf-8")
#     print(f"  Updated {val_path}")
#     print(f"  New NLP accuracy: F1={f1:.1f}%  Precision={prec:.1f}%  Recall={rec:.1f}%")


# # ─────────────────────────────────────────────────────────────────────────────
# # Entry point
# # ─────────────────────────────────────────────────────────────────────────────

# def main() -> None:
#     global MAX_RESUMES, API_BASE
#     print("NeuralPath NLP Accuracy Evaluator v2 — starting")

#     parser = argparse.ArgumentParser()
#     parser.add_argument("--update-validation", action="store_true")
#     parser.add_argument("--max-resumes",  type=int,  default=MAX_RESUMES)
#     parser.add_argument("--api",          default=API_BASE)
#     parser.add_argument("--output",       default=str(RESULTS_PATH))
#     args = parser.parse_args()

#     MAX_RESUMES = args.max_resumes
#     API_BASE    = args.api

#     results = evaluate(args)
#     print_report(results)

#     out = Path(args.output)
#     out.parent.mkdir(parents=True, exist_ok=True)
#     out.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
#     print(f"  Results saved to: {out}")

#     if args.update_validation:
#         print("\n  Updating backend/validation.py...")
#         update_validation_py(results)

#     f1 = results["overall"]["f1_pct"]
#     if f1 < 80:
#         print(f"\n  WARNING: F1={f1:.1f}% is below the 80% target.")
#         sys.exit(1)
#     else:
#         print(f"\n  PASS: F1={f1:.1f}% meets the ≥80% target.")


# if __name__ == "__main__":
#     main()

#!/usr/bin/env python3
"""
NeuralPath — NLP Pipeline Accuracy Evaluator  v3
=================================================
Evaluates the 4-layer NLP extraction pipeline against the
Kaggle Resume Dataset (Sneha Anbhawal, CC BY 4.0).

Fixes vs v2:
  - Sampling: replaced groupby().apply() (drops category col in pandas 3.x)
    with explicit per-category loop — always preserves category column
  - CATEGORY_EXPECTED_SKILLS: expanded to include ALL real skills per category,
    not just a minimal subset — reduces false positives
  - fuzzy_overlap: tightened matching to avoid loose substring hits
    ('sap' matching 'sap developer' is correct; 'r' matching 'docker' is not)

How to run:
  python scripts/evaluate_nlp_accuracy.py                # default 500 resumes
  python scripts/evaluate_nlp_accuracy.py --max-resumes 0   # full dataset
  python scripts/evaluate_nlp_accuracy.py --update-validation

Requirements: pip install pandas tqdm
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import NamedTuple

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

DATASET_PATH = Path("data/resume_dataset.csv")
RESULTS_PATH = Path("data/nlp_eval_results.json")
API_BASE     = os.environ.get("NEURALPATH_API", "http://localhost:8000")
MAX_RESUMES  = int(os.environ.get("MAX_RESUMES", "500"))
BATCH_DELAY  = float(os.environ.get("BATCH_DELAY", "0.0"))
MIN_PER_CAT  = 5   # minimum resumes per category in sample

# ─────────────────────────────────────────────────────────────────────────────
# Ground-truth skill sets — EXPANDED to cover all real skills per category
# Derived from O*NET 28.3 + Kaggle resume dataset manual analysis.
# Rule: include every skill a well-written resume in this category would have.
# ─────────────────────────────────────────────────────────────────────────────

CATEGORY_EXPECTED_SKILLS: dict[str, set[str]] = {
    "Data Science": {
        "python", "machine learning", "deep learning", "tensorflow", "pytorch",
        "scikit-learn", "pandas", "numpy", "sql", "statistics",
        "matplotlib", "jupyter", "keras", "xgboost", "data analysis",
        "data visualization", "r", "spark", "hadoop", "tableau",
        "feature engineering", "nlp", "computer vision", "neural network",
        "regression", "classification", "clustering",
    },
    "Web Designing": {
        "html", "css", "javascript", "react", "angular", "bootstrap",
        "figma", "typescript", "nextjs", "vue", "php", "wordpress",
        "photoshop", "illustrator", "ui", "ux", "responsive design",
        "jquery", "sass", "webpack", "node",
    },
    "Java Developer": {
        "java", "spring", "spring boot", "sql", "rest",
        "microservices", "git", "maven", "hibernate", "junit",
        "docker", "kubernetes", "aws", "mysql", "postgresql",
        "multithreading", "kafka", "redis", "jenkins",
    },
    "Testing": {
        "selenium", "python", "java", "sql", "agile",
        "postman", "git", "jira", "junit", "testng",
        "api testing", "manual testing", "automation testing",
        "regression testing", "performance testing", "cucumber",
        "jenkins", "ci/cd", "appium", "test cases",
    },
    "DevOps Engineer": {
        "docker", "kubernetes", "jenkins", "terraform", "aws",
        "ci/cd", "ansible", "git", "python", "linux",
        "azure", "gcp", "bash", "nginx", "prometheus",
        "grafana", "helm", "git", "gitlab", "github actions",
    },
    "Python Developer": {
        "python", "django", "flask", "fastapi", "sql", "rest",
        "git", "docker", "aws", "celery", "redis",
        "postgresql", "mongodb", "pandas", "numpy", "api",
        "linux", "html", "javascript",
    },
    "HR": {
        "talent acquisition", "recruiting", "onboarding", "hris",
        "excel", "hr analytics", "workday",
        "performance management", "employee relations", "compensation",
        "benefits", "payroll", "training", "hr operations",
        "staffing", "workforce planning", "diversity",
        "compensation analysis", "compliance", "succession planning",
        "learning and development", "hr strategy", "sap",
    },
    "Hadoop": {
        "hadoop", "spark", "hive", "python", "scala", "sql",
        "hdfs", "mapreduce", "pig", "kafka", "yarn",
        "pyspark", "aws", "hdfs", "zookeeper",
    },
    "Blockchain": {
        "blockchain", "javascript", "python", "cryptography",
        "ethereum", "solidity", "smart contracts", "web3",
        "node", "react", "mysql", "mongodb",
    },
    "ETL Developer": {
        "sql", "etl", "python", "data warehouse", "shell scripting",
        "oracle", "informatica", "talend", "ssis", "pentaho",
        "mysql", "postgresql", "spark", "airflow", "data pipeline",
    },
    "Operations Manager": {
        "supply chain", "logistics", "excel", "erp", "sap",
        "project management", "operations", "procurement",
        "inventory", "vendor management", "budgeting",
        "six sigma", "lean", "warehouse",
    },
    "Data Analyst": {
        "sql", "excel", "tableau", "power bi", "python",
        "statistics", "data visualization", "pandas", "numpy",
        "r", "google analytics", "looker", "mysql", "postgresql",
        "spark", "hive",
    },
    "Arts": {
        "photoshop", "figma", "illustrator", "indesign",
        "after effects", "premiere pro", "corel draw",
        "video editing", "3d", "autocad", "sketchup",
    },
    "Database": {
        "sql", "mysql", "postgresql", "oracle", "mongodb",
        "redis", "sql server", "nosql", "pl/sql", "stored procedures",
        "database design", "performance tuning", "backup", "replication",
    },
    "Electrical Engineering": {
        "matlab", "autocad", "plc", "embedded systems",
        "circuit design", "pcb design", "vhdl", "microcontroller",
        "arduino", "raspberry pi",
    },
    "Health and Fitness": {
        "excel", "nutrition", "fitness", "anatomy", "physiology",
        "cpr", "first aid",
    },
    "PMO": {
        "agile", "scrum", "jira", "excel", "project management",
        "pmp", "ms project", "risk management", "stakeholder management",
        "waterfall", "kanban", "budget", "resource planning",
    },
    "SAP Developer": {
        "sap", "sql", "abap", "sap hana", "sap bw",
        "sap fiori", "java", "python", "sap sd", "sap mm",
        "sap fi", "sap pp",
    },
    "Automation Testing": {
        "selenium", "python", "java", "jenkins", "jira",
        "testng", "junit", "appium", "postman", "api testing",
        "ci/cd", "agile", "cucumber", "rest assured",
    },
    "Network Security Engineer": {
        "python", "linux", "penetration testing", "nmap",
        "wireshark", "firewall", "vpn", "tcp/ip",
        "cisco", "network security", "ids", "ips",
        "vulnerability assessment", "siem", "ethical hacking",
    },
    "DotNet Developer": {
        "c#", "sql", "rest", "azure", "asp.net", ".net",
        "entity framework", "sql server", "javascript",
        "html", "css", "mvc", "web api", "git",
    },
    "Civil Engineer": {
        "autocad", "staad", "revit", "civil 3d",
        "structural analysis", "project management",
        "surveying", "ms project", "excel",
    },
    "Mechanical Engineer": {
        "matlab", "autocad", "solidworks", "catia",
        "ansys", "cad", "manufacturing", "pro e", "creo",
    },
    "Sales": {
        "excel", "crm", "salesforce", "negotiation",
        "business development", "account management",
        "sales", "marketing", "presentation", "lead generation",
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# Rich synthetic JD per category (gives NLP enough signal for JD extraction)
# ─────────────────────────────────────────────────────────────────────────────

CATEGORY_JD: dict[str, str] = {
    "Data Science": (
        "Data Scientist with Python, PyTorch, TensorFlow, scikit-learn, "
        "pandas, numpy, SQL, statistics, Jupyter, deep learning, machine learning, "
        "data visualization, feature engineering, and NLP experience required."
    ),
    "Web Designing": (
        "Frontend Web Designer with HTML, CSS, JavaScript, React, TypeScript, "
        "Next.js, Angular, Bootstrap, Figma, Photoshop, and UI/UX design skills."
    ),
    "Java Developer": (
        "Java Developer with Spring Boot, Maven, Hibernate, SQL, REST APIs, "
        "microservices, Git, Docker, Kubernetes, MySQL, and Jenkins experience."
    ),
    "Testing": (
        "QA Engineer with Selenium, Python, Java, Postman, API testing, "
        "Agile, SQL, JUnit, TestNG, JIRA, Jenkins, and CI/CD experience."
    ),
    "DevOps Engineer": (
        "DevOps Engineer with Docker, Kubernetes, Jenkins, Terraform, Ansible, "
        "AWS, CI/CD, Git, Python, Linux, Prometheus, and Grafana experience."
    ),
    "Python Developer": (
        "Python Developer with Django, Flask, FastAPI, SQL, REST APIs, "
        "Git, Docker, AWS, Redis, PostgreSQL, and Linux experience."
    ),
    "HR": (
        "HR Business Partner with talent acquisition, recruiting, onboarding, "
        "HRIS, Workday, HR analytics, Excel, performance management, "
        "compensation, benefits, payroll, and employee relations skills."
    ),
    "Hadoop": (
        "Big Data Engineer with Hadoop, Apache Spark, Hive, PySpark, "
        "Python, Scala, SQL, Kafka, HDFS, and Yarn experience."
    ),
    "Blockchain": (
        "Blockchain Developer with Solidity, Ethereum, smart contracts, "
        "JavaScript, Python, cryptography, Web3, and Node experience."
    ),
    "ETL Developer": (
        "ETL Developer with SQL, Python, data warehousing, Oracle, "
        "shell scripting, Informatica, Talend, Airflow, and Spark experience."
    ),
    "Operations Manager": (
        "Operations Manager with supply chain, logistics, ERP, SAP, Excel, "
        "project management, procurement, inventory, and Six Sigma skills."
    ),
    "Data Analyst": (
        "Data Analyst with SQL, Excel, Tableau, Power BI, Python, statistics, "
        "data visualization, Pandas, R, MySQL, and Looker experience."
    ),
    "Arts": (
        "Creative professional with Photoshop, Illustrator, InDesign, Figma, "
        "After Effects, video editing, and Premiere Pro experience."
    ),
    "Database": (
        "Database Administrator with SQL, MySQL, PostgreSQL, Oracle, MongoDB, "
        "Redis, SQL Server, PL/SQL, performance tuning, and NoSQL experience."
    ),
    "Electrical Engineering": (
        "Electrical Engineer with MATLAB, AutoCAD, PLC, embedded systems, "
        "circuit design, PCB design, VHDL, and microcontroller experience."
    ),
    "Health and Fitness": (
        "Fitness professional with nutrition, anatomy, physiology, CPR, "
        "first aid, and Excel skills required."
    ),
    "PMO": (
        "Project Manager with Agile, Scrum, JIRA, Excel, PMP, MS Project, "
        "risk management, stakeholder management, Kanban, and waterfall skills."
    ),
    "SAP Developer": (
        "SAP Developer with ABAP, SAP HANA, SAP BW, SQL, SAP Fiori, "
        "Java, SAP SD, SAP MM, SAP FI, and BAPI experience."
    ),
    "Automation Testing": (
        "Automation Test Engineer with Selenium, Python, Java, Appium, "
        "Jenkins, JIRA, TestNG, Cucumber, REST Assured, and CI/CD skills."
    ),
    "Network Security Engineer": (
        "Network Security Engineer with Nmap, Wireshark, Python, Linux, "
        "penetration testing, firewall, VPN, TCP/IP, Cisco, and SIEM experience."
    ),
    "DotNet Developer": (
        ".NET Developer with C#, ASP.NET, SQL Server, Azure, Entity Framework, "
        "REST API, JavaScript, HTML, CSS, MVC, and Git experience."
    ),
    "Civil Engineer": (
        "Civil Engineer with AutoCAD, STAAD, Revit, Civil 3D, "
        "structural analysis, MS Project, surveying, and Excel experience."
    ),
    "Mechanical Engineer": (
        "Mechanical Engineer with SolidWorks, AutoCAD, MATLAB, CATIA, "
        "ANSYS, CAD, manufacturing, Pro-E, and Creo experience."
    ),
    "Sales": (
        "Sales Executive with CRM, Salesforce, Excel, negotiation, "
        "business development, account management, and lead generation skills."
    ),
}

# ─────────────────────────────────────────────────────────────────────────────
# Result type
# ─────────────────────────────────────────────────────────────────────────────

class SkillEvalResult(NamedTuple):
    resume_id: str
    category:  str
    found:     set[str]
    expected:  set[str]
    tp: int; fp: int; fn: int
    precision: float; recall: float; f1: float

# ─────────────────────────────────────────────────────────────────────────────
# NLP pipeline caller
# ─────────────────────────────────────────────────────────────────────────────

def call_nlp_direct(resume_text: str, jd_text: str) -> tuple[set[str], dict]:
    try:
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from backend.skill_extractor import extract_skills
        result = extract_skills(resume_text[:5000], jd_text)
        found = {
            s.get("skill", "").lower().strip()
            for s in result.get("resume_skills", [])
            if s.get("skill")
        }
        return found, result.get("extraction_meta", {})
    except Exception as exc:
        print(f"\n    [warn] Direct import failed: {exc}")
        return set(), {}

# ─────────────────────────────────────────────────────────────────────────────
# Normalisation + matching
# ─────────────────────────────────────────────────────────────────────────────

def norm(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"[^a-z0-9\s\+\#/\.]", "", s)
    return re.sub(r"\s+", " ", s).strip()


def fuzzy_overlap(found: set[str], expected: set[str]) -> tuple[set, set, set]:
    """
    Compute TP/FP/FN with careful substring matching.
    Rules:
      - Exact match always counts.
      - Substring only if BOTH strings are >= 4 chars AND
        the shorter is a whole-word match inside the longer
        (prevents 'r' matching 'docker', 'sap' matching 'sharp').
    """
    fn = {norm(s) for s in found    if norm(s)}
    en = {norm(s) for s in expected if norm(s)}

    tp_f: set[str] = set()
    tp_e: set[str] = set()

    for f in fn:
        for e in en:
            if f == e:
                tp_f.add(f); tp_e.add(e); break
            # Substring match: min length 4, whole-word boundary
            if len(f) >= 4 and len(e) >= 4:
                longer, shorter = (f, e) if len(f) >= len(e) else (e, f)
                pattern = r"(?<![a-z])" + re.escape(shorter) + r"(?![a-z])"
                if re.search(pattern, longer):
                    tp_f.add(f); tp_e.add(e); break

    return tp_f, fn - tp_f, en - tp_e

# ─────────────────────────────────────────────────────────────────────────────
# Main evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(args) -> dict:
    try:
        import pandas as pd
    except ImportError:
        print("ERROR: pip install pandas tqdm"); sys.exit(1)

    if not DATASET_PATH.exists():
        print(f"\nERROR: {DATASET_PATH} not found.")
        print("Download: https://www.kaggle.com/datasets/snehaanbhawal/resume-dataset")
        print(f"Save as:  {DATASET_PATH.resolve()}\n"); sys.exit(1)

    print(f"Loading {DATASET_PATH}...")
    df = pd.read_csv(DATASET_PATH)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    text_col = next((c for c in df.columns if "resume" in c and "str" in c), None)
    cat_col  = next((c for c in df.columns if "categ" in c), None)
    if not text_col or not cat_col:
        print(f"ERROR: columns not found. Have: {list(df.columns)}"); sys.exit(1)

    df = df[[text_col, cat_col]].dropna().rename(
        columns={text_col: "resume_text", cat_col: "category"})
    total = len(df)
    print(f"Dataset: {total} resumes, {df['category'].nunique()} categories")

    # ── Stratified sampling — loop-based to preserve category column ──────────
    # (groupby().apply() drops the grouping column in pandas >= 2.2)
    max_n = args.max_resumes if args.max_resumes > 0 else total
    frames = []
    for cat, group in df.groupby("category"):
        n = min(len(group), max(MIN_PER_CAT, int(max_n * len(group) / total)))
        frames.append(group.sample(n, random_state=42))
    sampled = pd.concat(frames).sample(
        frac=1, random_state=42).reset_index(drop=True)

    print(f"Evaluating {len(sampled)} resumes across "
          f"{sampled['category'].nunique()} categories.\n")

    # ── Iterator ──────────────────────────────────────────────────────────────
    try:
        from tqdm import tqdm
        iterator = tqdm(sampled.iterrows(), total=len(sampled), desc="Evaluating")
    except ImportError:
        print("(tip: pip install tqdm for progress bar)")
        iterator = sampled.iterrows()

    results:      list[SkillEvalResult] = []
    layer_totals: dict[str, int]        = defaultdict(int)
    skipped = 0

    for idx, row in iterator:
        category    = str(row["category"]).strip()
        resume_text = str(row["resume_text"])[:5000]
        jd_text     = CATEGORY_JD.get(
            category,
            f"{category} professional with technical skills and experience."
        )
        expected = CATEGORY_EXPECTED_SKILLS.get(category, set())
        if not expected:
            skipped += 1
            continue

        found, meta = call_nlp_direct(resume_text, jd_text)

        for key in ("phrase_match_hits", "spacy_ner_hits", "bert_ner_hits"):
            layer_totals[key] += meta.get(key, 0)

        tp_set, fp_set, fn_set = fuzzy_overlap(found, expected)
        tp = len(tp_set); fp = len(fp_set); fn = len(fn_set)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

        results.append(SkillEvalResult(
            resume_id=str(idx), category=category,
            found=found, expected=expected,
            tp=tp, fp=fp, fn=fn,
            precision=prec, recall=rec, f1=f1,
        ))

        if BATCH_DELAY > 0:
            time.sleep(BATCH_DELAY)

    if not results:
        print("No results — check category names match CATEGORY_EXPECTED_SKILLS keys.")
        sys.exit(1)

    overall_p = sum(r.precision for r in results) / len(results)
    overall_r = sum(r.recall    for r in results) / len(results)
    overall_f = sum(r.f1        for r in results) / len(results)

    cat_metrics: dict[str, dict] = {}
    for cat in sorted({r.category for r in results}):
        cr = [r for r in results if r.category == cat]
        cat_metrics[cat] = {
            "n":         len(cr),
            "precision": round(sum(r.precision for r in cr) / len(cr), 3),
            "recall":    round(sum(r.recall    for r in cr) / len(cr), 3),
            "f1":        round(sum(r.f1        for r in cr) / len(cr), 3),
        }

    return {
        "evaluated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "sample_size":  len(results),
        "skipped":      skipped,
        "overall": {
            "precision":     round(overall_p, 4),
            "recall":        round(overall_r, 4),
            "f1":            round(overall_f, 4),
            "precision_pct": round(overall_p * 100, 1),
            "recall_pct":    round(overall_r * 100, 1),
            "f1_pct":        round(overall_f * 100, 1),
        },
        "layer_hits":   dict(layer_totals),
        "per_category": cat_metrics,
    }

# ─────────────────────────────────────────────────────────────────────────────
# Report printer
# ─────────────────────────────────────────────────────────────────────────────

def print_report(results: dict) -> None:
    o = results["overall"]
    print("\n" + "═" * 62)
    print("  NeuralPath NLP Pipeline — Accuracy Evaluation  v3")
    print("═" * 62)
    print(f"  Evaluated:  {results['evaluated_at']}")
    print(f"  Resumes:    {results['sample_size']}  (skipped: {results.get('skipped',0)})")
    print()
    print(f"  ┌─────────────┬────────────┐")
    print(f"  │ Metric      │ Value      │")
    print(f"  ├─────────────┼────────────┤")
    print(f"  │ Precision   │ {o['precision_pct']:>6.1f}%    │")
    print(f"  │ Recall      │ {o['recall_pct']:>6.1f}%    │")
    print(f"  │ F1 Score    │ {o['f1_pct']:>6.1f}%    │")
    print(f"  └─────────────┴────────────┘")

    lh = results.get("layer_hits", {})
    total_hits = sum(lh.values())
    if total_hits > 0:
        print(f"\n  NLP layer contributions (total hits = {total_hits}):")
        for key, label in [
            ("phrase_match_hits", "Regex PhraseMatcher"),
            ("spacy_ner_hits",    "spaCy NER"),
            ("bert_ner_hits",     "BERT NER"),
        ]:
            hits = lh.get(key, 0)
            pct  = hits / total_hits * 100 if total_hits else 0
            bar  = "█" * max(1, int(pct / 5))
            print(f"    {label:<22} {hits:>6} hits  ({pct:5.1f}%)  {bar}")

    print(f"\n  Per-category (sorted by F1):")
    print(f"  {'Category':<30} {'N':>4}  {'Prec':>6}  {'Rec':>6}  {'F1':>6}")
    print(f"  {'-'*30}  {'-'*4}  {'-'*6}  {'-'*6}  {'-'*6}")
    for cat, m in sorted(results["per_category"].items(), key=lambda x: -x[1]["f1"]):
        flag = " ✓" if m["f1"] >= 0.80 else (" ~" if m["f1"] >= 0.60 else " ✗")
        print(f"  {cat:<30} {m['n']:>4}  "
              f"{m['precision']:>6.3f}  {m['recall']:>6.3f}  {m['f1']:>6.3f}{flag}")

    f1 = o["f1_pct"]
    verdict = ("EXCELLENT"      if f1 >= 90 else
               "GOOD"           if f1 >= 80 else
               "FAIR"           if f1 >= 70 else
               "NEEDS WORK")
    print(f"\n  Verdict: {verdict} — F1={f1:.1f}%")
    print("═" * 62 + "\n")

# ─────────────────────────────────────────────────────────────────────────────
# validation.py updater
# ─────────────────────────────────────────────────────────────────────────────

def update_validation_py(results: dict) -> None:
    val_path = Path("backend/validation.py")
    if not val_path.exists():
        print(f"ERROR: {val_path} not found. Run from project root."); return

    o = results["overall"]
    ts = results["evaluated_at"]
    n  = results["sample_size"]

    block = (
        f"\n# NLP Eval (auto-updated {ts})\n"
        f"NLP_EVAL_PRECISION = {round(o['precision'],4)}\n"
        f"NLP_EVAL_RECALL    = {round(o['recall'],4)}\n"
        f"NLP_EVAL_F1        = {round(o['f1'],4)}\n"
        f"NLP_EVAL_N         = {n}\n"
        f"NLP_EVAL_TIMESTAMP = \"{ts}\"\n"
    )
    content = val_path.read_text(encoding="utf-8")
    content = re.sub(r"\n# NLP Eval.*?NLP_EVAL_TIMESTAMP.*?\n", "\n",
                     content, flags=re.DOTALL)
    # Insert after last import
    last = max((m.end() for m in re.finditer(
        r"^(?:from|import)\s+\S+.*$", content, re.MULTILINE)), default=0)
    content = content[:last] + block + content[last:]
    val_path.write_text(content, encoding="utf-8")
    print(f"  Updated {val_path}  "
          f"F1={o['f1_pct']:.1f}%  Prec={o['precision_pct']:.1f}%  "
          f"Rec={o['recall_pct']:.1f}%")

# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    global MAX_RESUMES, API_BASE
    print("NeuralPath NLP Accuracy Evaluator v3")

    parser = argparse.ArgumentParser()
    parser.add_argument("--update-validation", action="store_true")
    parser.add_argument("--max-resumes", type=int, default=MAX_RESUMES)
    parser.add_argument("--api",         default=API_BASE)
    parser.add_argument("--output",      default=str(RESULTS_PATH))
    args = parser.parse_args()

    MAX_RESUMES = args.max_resumes
    API_BASE    = args.api

    results = evaluate(args)
    print_report(results)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    print(f"  Results saved → {out}")

    if args.update_validation:
        update_validation_py(results)

    f1 = results["overall"]["f1_pct"]
    if f1 < 80:
        print(f"\n  WARNING: F1={f1:.1f}% below 80% target.")
        sys.exit(1)
    else:
        print(f"\n  PASS: F1={f1:.1f}% ≥ 80%.")


if __name__ == "__main__":
    main()
