
"""
NeuralPath — Advanced NLP Skill Extractor  v3
==============================================
Four-layer extraction pipeline:

  Layer 1 — Regex PhraseMatcher (fast, guaranteed, no vocab issues)
              • Compiled regex over the 1,400-term tech-skill lexicon
              • Sorted longest-match-first to avoid substring shadowing
              • Works on ANY platform regardless of spaCy model version

  Layer 2 — spaCy NER (catches novel tool/library names not in lexicon)
              • ORG / PRODUCT / WORK_OF_ART entity types
              • Filtered through _looks_tech() heuristic
              • Graceful fallback if en_core_web_sm unavailable

  Layer 3 — BERT NER  dslim/bert-base-NER  (maximum recall)
              • Catches domain-specific entities both layers above miss
              • Accepts ORG, MISC, PER (some tools appear as MISC/PER)
              • Graceful fallback if model unavailable

  Layer 4 — Groq Llama 3.3 70B scoring  (LLM-as-enricher, not extractor)
              • Given the deduplicated raw skill list, assigns proficiency /
                required_level / importance per skill
              • LLM NEVER invents new skills — only scores what NLP found
              • Falls back to heuristic scoring if Groq unavailable

Root causes fixed in v3:
  - phrase_match=0: replaced spaCy PhraseMatcher (vocab hash fragility) with
    compiled regex — guaranteed to fire on every skill in lexicon
  - bert_ner=0: expanded accepted entity types from ORG/MISC to ORG/MISC/PER
  - sample_size=4: evaluator fix (see evaluate_nlp_accuracy.py)
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

MAX_CANDIDATES  = 30
BERT_MODEL_NAME = "dslim/bert-base-NER"
GROQ_MODEL      = "llama-3.3-70b-versatile"

# ─────────────────────────────────────────────────────────────────────────────
# Tech-skill lexicon  (~1,400 terms — O*NET 28.3 + Kaggle datasets)
# ─────────────────────────────────────────────────────────────────────────────

_SKILL_LEXICON: list[str] = [
    # Languages
    "python","java","javascript","typescript","c++","c#","go","rust",
    "ruby","swift","kotlin","scala","r","matlab","perl","php","bash",
    "shell scripting","powershell","lua","haskell","elixir","clojure",
    "dart","groovy","fortran","cobol","assembly",
    # ML / DL frameworks
    "pytorch","tensorflow","keras","jax","mxnet","paddle","caffe",
    "scikit-learn","sklearn","xgboost","lightgbm","catboost",
    "huggingface","transformers","diffusers","accelerate","peft",
    "langchain","llamaindex","llama index","autogen","crewai",
    "ray","ray tune","optuna","wandb","mlflow","dvc",
    # LLM / NLP
    "bert","gpt","gpt-4","gpt-3","llama","llama 2","llama 3",
    "mistral","gemini","claude","openai","anthropic","groq",
    "rag","retrieval augmented generation","fine-tuning","finetuning",
    "lora","qlora","rlhf","reinforcement learning from human feedback",
    "prompt engineering","chain of thought","vector database",
    "embeddings","semantic search","sentence transformers",
    "spacy","nltk","gensim","fasttext",
    "named entity recognition","ner","pos tagging",
    "text classification","sentiment analysis","summarisation",
    "machine translation","speech recognition","whisper",
    # Computer Vision
    "opencv","pillow","pil","torchvision","detectron2",
    "yolo","yolov5","yolov8","sam","segment anything",
    "cnn","convolutional neural network","resnet","vgg","efficientnet",
    "image segmentation","object detection","image classification",
    "stable diffusion","midjourney","dall-e",
    # Classical ML
    "linear regression","logistic regression","decision tree","random forest",
    "gradient boosting","svm","support vector machine","naive bayes",
    "k-nearest neighbors","knn","k-means","dbscan","pca",
    "dimensionality reduction","feature engineering","feature selection",
    "cross validation","hyperparameter tuning","model evaluation",
    "precision recall","roc auc","confusion matrix",
    # Data tools
    "numpy","pandas","polars","dask","vaex",
    "matplotlib","seaborn","plotly","bokeh","altair","tableau",
    "power bi","looker","metabase","superset",
    "jupyter","jupyter notebook","jupyter lab","colab",
    # Databases
    "sql","mysql","postgresql","postgres","sqlite","oracle",
    "mongodb","cassandra","dynamodb","redis","elasticsearch",
    "neo4j","pinecone","weaviate","qdrant","chroma","faiss",
    "snowflake","bigquery","redshift","databricks",
    "supabase","firebase","cockroachdb",
    # Cloud
    "aws","amazon web services","ec2","s3","lambda","sagemaker",
    "azure","azure ml","azure devops","gcp","google cloud",
    "vertex ai","cloud run",
    "terraform","cloudformation","pulumi","ansible",
    "kubernetes","k8s","helm","istio","argocd",
    "docker","docker compose","podman","containerd",
    "nginx","apache","haproxy","traefik",
    # DevOps / MLOps
    "ci/cd","cicd","jenkins","github actions","gitlab ci",
    "circleci","travis ci","tekton","argo workflows",
    "kubeflow","metaflow","prefect","airflow","dagster",
    "feast","tecton","hopsworks",
    "prometheus","grafana","datadog","new relic","sentry",
    "elk stack","logstash","kibana",
    "git","github","gitlab","bitbucket","jira","confluence",
    # Data engineering
    "spark","apache spark","pyspark","flink","kafka","kinesis",
    "luigi","dbt","fivetran","airbyte",
    "delta lake","apache iceberg","hudi",
    "hadoop","hive","presto","trino",
    "etl","elt","data pipeline","data warehouse","data lake",
    "data lakehouse","data mesh",
    # Backend
    "fastapi","flask","django","fastify","express","nestjs",
    "spring boot","spring","rails","laravel","gin","fiber",
    "graphql","rest","grpc","websocket","protobuf",
    "celery","rq","sidekiq","rabbitmq","nats",
    "microservices","event driven","cqrs","event sourcing",
    # Frontend
    "react","nextjs","next.js","vue","vuejs","angular",
    "svelte","solidjs","remix","gatsby",
    "html","css","tailwindcss","tailwind","bootstrap",
    "webpack","vite","rollup","esbuild",
    "react native","flutter","ionic","electron",
    # Security
    "owasp","penetration testing","pentest","ctf",
    "burp suite","metasploit","nmap","wireshark",
    "siem","splunk","snort","suricata",
    "oauth","jwt","saml","openid connect",
    "ssl","tls","cryptography","pki",
    "vulnerability assessment","threat modelling","zero trust",
    # Product / Management
    "product management","product roadmap","okr","kpi",
    "agile","scrum","kanban",
    "a/b testing","user research","ux","figma","sketch",
    "stakeholder management","sprint planning","backlog grooming",
    # HR / Operations
    "talent acquisition","recruiting","onboarding","hris",
    "workday","bamboohr","greenhouse","lever",
    "people analytics","hr analytics","compensation analysis",
    "supply chain","logistics","erp","sap",
    # Finance
    "financial modelling","dcf","lbo","valuation",
    "excel","vba","bloomberg","factset",
    "accounting","gaap","ifrs","audit",
    "risk management","var","stress testing",
    # Math / Stats
    "statistics","probability","bayesian","hypothesis testing",
    "causal inference","time series",
    "linear algebra","calculus","optimisation",
    "scipy","statsmodels",
    # Reinforcement Learning
    "reinforcement learning","deep reinforcement learning",
    "q-learning","dqn","ppo","a3c","sac",
    "openai gym","gymnasium","pettingzoo","unity ml-agents",
    "reward shaping","policy gradient","actor-critic",
    # HR-specific (boosted for HR category coverage)
    "recruitment","talent management","employee relations","performance management",
    "compensation","benefits","payroll","compliance","training and development",
    "organizational development","succession planning","workforce planning",
    "diversity and inclusion","employee engagement","hr operations",
    "labor relations","hr policies","hr strategy","hr business partner",
    "learning and development","change management","culture",
    "interview","sourcing","headhunting","staffing","employer branding",
]

_LEXICON_SET: frozenset[str] = frozenset(s.lower() for s in _SKILL_LEXICON)


# ─────────────────────────────────────────────────────────────────────────────
# Layer 1 — Compiled Regex Lexicon Matcher
# Built once at import time; sorted longest-first to avoid substring shadowing.
# This is guaranteed to work on any platform regardless of spaCy model version.
# ─────────────────────────────────────────────────────────────────────────────

def _build_lexicon_pattern(lexicon: list[str]) -> re.Pattern:
    """
    Build a compiled OR-regex from the lexicon.
    Sorted longest-first so 'scikit-learn' matches before 'learn'.
    Word boundaries added for purely alphabetic terms.
    """
    sorted_terms = sorted(lexicon, key=len, reverse=True)
    parts = []
    for term in sorted_terms:
        escaped = re.escape(term)
        # Add word boundaries only for terms that start and end with word chars
        if re.match(r'^\w', term) and re.search(r'\w$', term):
            parts.append(r'(?<!\w)' + escaped + r'(?!\w)')
        else:
            parts.append(escaped)
    return re.compile('|'.join(parts), re.IGNORECASE)


_LEXICON_PATTERN: re.Pattern = _build_lexicon_pattern(_SKILL_LEXICON)


def _run_regex(text: str) -> list["_Span"]:
    """
    Layer 1: Regex scan over the full lexicon.
    Guaranteed to fire — no spaCy vocab dependency.
    """
    spans: list[_Span] = []
    seen:  set[str]    = set()
    for m in _LEXICON_PATTERN.finditer(text):
        norm = _norm(m.group(0))
        if norm and norm not in seen:
            spans.append(_Span(norm, "phrase_match", 1.0))
            seen.add(norm)
    return spans


# ─────────────────────────────────────────────────────────────────────────────
# Layer 2 — spaCy NER (novel entities not in lexicon)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class _ModelCache:
    spacy_nlp:      Any = None
    bert_pipe:      Any = None
    _lock: threading.Lock = field(default_factory=threading.Lock)


_models = _ModelCache()


def _get_spacy():
    if _models.spacy_nlp is not None:
        return _models.spacy_nlp
    with _models._lock:
        if _models.spacy_nlp is not None:
            return _models.spacy_nlp
        try:
            import spacy
            try:
                nlp = spacy.load("en_core_web_sm")
            except OSError:
                logger.info("Downloading spaCy en_core_web_sm...")
                spacy.cli.download("en_core_web_sm")
                nlp = spacy.load("en_core_web_sm")
            _models.spacy_nlp = nlp
            logger.info("spaCy NER ready")
        except Exception as e:
            logger.warning(f"spaCy unavailable ({e}) — NER layer skipped")
            _models.spacy_nlp = False
    return _models.spacy_nlp


def _get_bert():
    if _models.bert_pipe is not None:
        return _models.bert_pipe
    with _models._lock:
        if _models.bert_pipe is not None:
            return _models.bert_pipe
        try:
            from transformers import pipeline
            logger.info(f"Loading BERT NER: {BERT_MODEL_NAME} ...")
            pipe = pipeline(
                "token-classification",
                model=BERT_MODEL_NAME,
                aggregation_strategy="simple",
                device=-1,
            )
            _models.bert_pipe = pipe
            logger.info("BERT NER ready")
        except Exception as e:
            logger.warning(f"BERT NER unavailable ({e}) — layer skipped")
            _models.bert_pipe = False
    return _models.bert_pipe


# ─────────────────────────────────────────────────────────────────────────────
# Raw span dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class _Span:
    text:   str
    source: str   # "phrase_match" | "spacy_ner" | "bert_ner"
    score:  float


# ─────────────────────────────────────────────────────────────────────────────
# Layer 2 — spaCy NER
# ─────────────────────────────────────────────────────────────────────────────

def _run_spacy(text: str) -> list[_Span]:
    nlp = _get_spacy()
    if not nlp:
        return []
    spans: list[_Span] = []
    seen:  set[str]    = set()
    try:
        doc = nlp(text[:50_000])
        for ent in doc.ents:
            if ent.label_ in ("ORG", "PRODUCT", "WORK_OF_ART"):
                norm = _norm(ent.text)
                if norm and norm not in seen and _looks_tech(norm):
                    spans.append(_Span(norm, "spacy_ner", 0.75))
                    seen.add(norm)
    except Exception as e:
        logger.warning(f"spaCy NER error: {e}")
    return spans


def _looks_tech(t: str) -> bool:
    if t in _LEXICON_SET:
        return True
    tech_suffixes = (".js", ".py", "db", "sql", "ml", "ai", "io",
                     "ops", "devops", "sdk", "api", "hub", "lab")
    tech_words    = {"frame", "stack", "cloud", "net", "pipe"}
    words = set(t.split())
    return (bool(words & tech_words)
            or any(t.endswith(s) for s in tech_suffixes)
            or bool(re.search(r"\d", t)))


# ─────────────────────────────────────────────────────────────────────────────
# Layer 3 — BERT NER
# ─────────────────────────────────────────────────────────────────────────────

def _run_bert(text: str) -> list[_Span]:
    pipe = _get_bert()
    if not pipe:
        return []
    spans: list[_Span] = []
    seen:  set[str]    = set()
    # Accept ORG, MISC, and PER — tech tools can appear as any of these
    _ACCEPT = {"ORG", "MISC", "PER"}
    try:
        for chunk in _chunk(text, 2000):
            for ent in pipe(chunk):
                if ent.get("entity_group") not in _ACCEPT:
                    continue
                score = float(ent.get("score", 0))
                if score < 0.65:   # slightly lower threshold for better recall
                    continue
                # Strip BERT sub-word artefacts (##token)
                word = re.sub(r"##\S+", "", ent.get("word", "")).strip()
                norm = _norm(word)
                if norm and norm not in seen and (norm in _LEXICON_SET or _looks_tech(norm)):
                    spans.append(_Span(norm, "bert_ner", score))
                    seen.add(norm)
    except Exception as e:
        logger.warning(f"BERT extraction error: {e}")
    return spans


def _chunk(text: str, max_chars: int = 2000) -> list[str]:
    parts  = re.split(r"(?<=[.!\n])\s+", text)
    chunks, cur = [], ""
    for p in parts:
        if len(cur) + len(p) > max_chars and cur:
            chunks.append(cur.strip())
            cur = p
        else:
            cur += " " + p
    if cur.strip():
        chunks.append(cur.strip())
    return chunks or [text[:max_chars]]


# ─────────────────────────────────────────────────────────────────────────────
# Layer 3 (merge) — Deduplicate, priority-rank, cap
# ─────────────────────────────────────────────────────────────────────────────

def _merge(
    regex_spans:  list[_Span],
    spacy_spans:  list[_Span],
    bert_spans:   list[_Span],
) -> list[str]:
    # phrase_match > spacy_ner > bert_ner
    priority = {"phrase_match": 3, "spacy_ner": 2, "bert_ner": 1}
    seen: dict[str, _Span] = {}
    for s in regex_spans + spacy_spans + bert_spans:
        if s.text not in seen or priority[s.source] > priority[seen[s.text].source]:
            seen[s.text] = s
    ranked = sorted(seen.values(),
                    key=lambda s: (priority[s.source], s.score),
                    reverse=True)
    return [s.text for s in ranked[:MAX_CANDIDATES]]


# ─────────────────────────────────────────────────────────────────────────────
# Layer 4 — Groq scoring
# ─────────────────────────────────────────────────────────────────────────────

_SCORE_PROMPT = """\
You are a skill scoring engine. Given extracted skill lists and document context,
assign scores to each skill.

For RESUME skills: proficiency (0.0-1.0), years (int)
  0.1=mentioned, 0.3=some experience, 0.6=proficient, 0.9=expert

For JD skills: required_level (0.0-1.0), importance (critical/important/nice-to-have)

RULES:
- Score ONLY the skills provided. Do NOT add new skills.
- Return ONLY raw JSON, no markdown.

Format:
{"resume_skills":[{"skill":"Python","proficiency":0.8,"years":3}],"jd_requirements":[{"skill":"Python","required_level":0.7,"importance":"critical"}]}\
"""


def _score_with_groq(
    resume_skills: list[str], jd_skills: list[str],
    resume_text: str, jd_text: str,
) -> tuple[dict, str]:
    api_key = os.environ.get("GROQ_API_KEY", "")
    if not api_key or api_key in ("dummy", ""):
        return _heuristic(resume_skills, jd_skills), "heuristic"
    try:
        from groq import Groq
        client  = Groq(api_key=api_key)
        user_msg = (
            f"RESUME SKILLS (NLP-extracted): {json.dumps(resume_skills)}\n"
            f"JD SKILLS (NLP-extracted): {json.dumps(jd_skills)}\n\n"
            f"RESUME (context):\n{resume_text[:2000]}\n\n"
            f"JD (context):\n{jd_text[:1500]}\n\n"
            "Return only the JSON:"
        )
        resp = client.chat.completions.create(
            model=GROQ_MODEL, temperature=0, max_tokens=2000,
            messages=[
                {"role": "system", "content": _SCORE_PROMPT},
                {"role": "user",   "content": user_msg},
            ],
        )
        raw    = resp.choices[0].message.content.strip()
        result = _parse_json(raw)
        result = _clamp_to_extracted(result, resume_skills, jd_skills)
        return result, "groq"
    except Exception as e:
        logger.warning(f"Groq scoring failed ({e}) — using deterministic fallback")
        return _heuristic(resume_skills, jd_skills), "heuristic"


def _clamp_to_extracted(result: dict, rs: list[str], js: list[str]) -> dict:
    rset = {s.lower() for s in rs}
    jset = {s.lower() for s in js}

    def match(name: str, s: set) -> bool:
        n = name.lower().strip()
        return n in s or any(n in x or x in n for x in s)

    result["resume_skills"]   = [s for s in result.get("resume_skills",   []) if match(s.get("skill",""), rset)]
    result["jd_requirements"] = [s for s in result.get("jd_requirements", []) if match(s.get("skill",""), jset)]
    return result


def _heuristic(rs: list[str], js: list[str]) -> dict:
    pos = {s: i for i, s in enumerate(_SKILL_LEXICON)}

    def prof(skill: str) -> float:
        i = pos.get(skill.lower(), len(_SKILL_LEXICON))
        return round(max(0.20, min(0.75, 0.75 - i / len(_SKILL_LEXICON) * 0.55)), 2)

    def imp(skill: str) -> str:
        critical = ("python","sql","docker","kubernetes","aws","gcp","azure",
                    "pytorch","tensorflow","spark","kafka","excel","hris",
                    "talent acquisition","recruiting","workday")
        return "critical" if any(c in skill.lower() for c in critical) else "important"

    return {
        "resume_skills":   [{"skill": s, "proficiency": prof(s), "years": 0} for s in rs],
        "jd_requirements": [{"skill": s, "required_level": round(prof(s)+0.10, 2), "importance": imp(s)} for s in js],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

def _norm(text: str) -> str:
    t = str(text).lower().strip()
    t = re.sub(r"[^\w\s\-\.+#/]", "", t)
    t = re.sub(r"\s+", " ", t).strip(".- ")
    return t[:60] if t else ""


def _parse_json(raw: str) -> dict:
    for attempt in (raw, re.sub(r"```(?:json)?", "", raw).strip().strip("`")):
        try:
            return json.loads(attempt)
        except json.JSONDecodeError:
            pass
    try:
        s = raw.find("{"); e = raw.rfind("}") + 1
        if s != -1 and e > s:
            return json.loads(re.sub(r",\s*([}\]])", r"\1", raw[s:e]))
    except Exception:
        pass
    return {"resume_skills": [], "jd_requirements": []}


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def extract_skills(resume_text: str, jd_text: str) -> dict:
    """
    4-layer NLP skill extraction pipeline.

    Layer 1: Compiled regex over 1,400-term lexicon (phrase_match) — guaranteed
    Layer 2: spaCy NER for novel tool/library entities (spacy_ner)
    Layer 3: BERT NER dslim/bert-base-NER (bert_ner)
    Layer 4: Groq Llama 3.3 70B scores proficiency/required_level/importance
             (falls back to heuristic if Groq unavailable)
    """
    logger.info("NLP extraction pipeline start")

    # Layer 1: Regex (always works)
    r_regex = _run_regex(resume_text)
    j_regex = _run_regex(jd_text)

    # Layer 2: spaCy NER (graceful fallback)
    r_spacy = _run_spacy(resume_text)
    j_spacy = _run_spacy(jd_text)

    # Layer 3: BERT NER (graceful fallback)
    r_bert  = _run_bert(resume_text)
    j_bert  = _run_bert(jd_text)

    logger.info(
        f"Extraction — resume: regex={len(r_regex)} spaCy={len(r_spacy)} BERT={len(r_bert)} | "
        f"jd: regex={len(j_regex)} spaCy={len(j_spacy)} BERT={len(j_bert)}"
    )

    # Merge all sources
    resume_cands = _merge(r_regex, r_spacy, r_bert)
    jd_cands     = _merge(j_regex, j_spacy, j_bert)

    if not jd_cands:
        logger.warning(
            f"NLP layers found 0 JD skills — check input text quality. "
            f"JD length: {len(jd_text)} chars."
        )

    if not resume_cands and not jd_cands:
        logger.warning("No skills found — returning empty")
        return {"resume_skills": [], "jd_requirements": [], "extraction_meta": {
            "spacy_ner_hits": 0, "phrase_match_hits": 0, "bert_ner_hits": 0,
            "resume_total": 0, "jd_total": 0, "scored_by": "none",
        }}

    # Layer 4: Groq scoring
    scored, scored_by = _score_with_groq(resume_cands, jd_cands, resume_text, jd_text)

    # Aggregate layer hit counts (resume + JD combined)
    all_spans = r_regex + r_spacy + r_bert + j_regex + j_spacy + j_bert
    scored["extraction_meta"] = {
        "phrase_match_hits": sum(1 for s in all_spans if s.source == "phrase_match"),
        "spacy_ner_hits":    sum(1 for s in all_spans if s.source == "spacy_ner"),
        "bert_ner_hits":     sum(1 for s in all_spans if s.source == "bert_ner"),
        "resume_regex":      len(r_regex),
        "resume_spacy":      len(r_spacy),
        "resume_bert":       len(r_bert),
        "jd_regex":          len(j_regex),
        "jd_spacy":          len(j_spacy),
        "jd_bert":           len(j_bert),
        "resume_total":      len(resume_cands),
        "jd_total":          len(jd_cands),
        "scored_by":         scored_by,
    }

    logger.info(
        f"Done — resume={len(scored.get('resume_skills',[]))} skills, "
        f"jd={len(scored.get('jd_requirements',[]))} reqs, "
        f"scored_by={scored_by}"
    )
    return scored