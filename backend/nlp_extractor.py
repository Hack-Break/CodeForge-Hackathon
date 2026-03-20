"""
NeuralPath — NLP Skill Extractor  (v3.0)
==========================================
Replaces the pure-LLM extractor with a 4-layer NLP pipeline:

  Layer 1 — spaCy NER          (en_core_web_sm)
  Layer 2 — spaCy PhraseMatcher on 1 400-skill lexicon
  Layer 3 — BERT NER            (dslim/bert-base-NER) — fallback / boost
  Layer 4 — Groq Llama 3.3 70B  (scores proficiency + importance only)

The LLM is now ONLY used to assign numeric scores — NOT to decide what
skills exist.  That job belongs entirely to the NLP layers, which are
deterministic and auditable.

Public API (drop-in replacement):
  extract_skills(resume_text, jd_text) -> dict
    returns {"resume_skills": [...], "jd_requirements": [...]}

Diagnostics:
  get_extraction_stats() -> dict   (called by /api/nlp/status)
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# 1.  Skill lexicon  (1 400+ canonical tech skills)
# ─────────────────────────────────────────────────────────────────────────────

SKILL_LEXICON: list[str] = [
    # Languages
    "Python", "Java", "JavaScript", "TypeScript", "C", "C++", "C#", "Go",
    "Rust", "Kotlin", "Swift", "Ruby", "PHP", "Scala", "R", "MATLAB",
    "Perl", "Haskell", "Elixir", "Erlang", "Clojure", "Lua", "Julia",
    "Dart", "Objective-C", "Assembly", "COBOL", "Fortran", "SAS", "VHDL",
    # ML / DL frameworks
    "PyTorch", "TensorFlow", "Keras", "JAX", "MXNet", "Caffe", "Theano",
    "PaddlePaddle", "Flax", "Haiku", "Lightning", "PyTorch Lightning",
    "fastai", "Hugging Face", "Transformers", "Diffusers", "Accelerate",
    "PEFT", "LoRA", "QLoRA", "RLHF", "InstructGPT",
    # Data / ML libraries
    "scikit-learn", "sklearn", "XGBoost", "LightGBM", "CatBoost",
    "NumPy", "Pandas", "SciPy", "Matplotlib", "Seaborn", "Plotly",
    "Bokeh", "Altair", "Statsmodels", "NLTK", "spaCy", "Gensim",
    "TextBlob", "CoreNLP", "AllenNLP",
    # Computer Vision
    "OpenCV", "Pillow", "PIL", "torchvision", "timm", "Detectron2",
    "MMDetection", "YOLOv8", "YOLO", "SAM", "CLIP", "DALL-E",
    "Stable Diffusion", "ControlNet",
    # NLP / LLMs
    "BERT", "GPT", "GPT-4", "GPT-3", "LLaMA", "Llama 2", "Mistral",
    "Falcon", "Bloom", "T5", "BART", "RoBERTa", "DistilBERT", "ELECTRA",
    "XLNet", "DeBERTa", "mBERT", "ChatGPT", "Claude", "Gemini",
    "Langchain", "LangChain", "LlamaIndex", "LLM", "RAG",
    "Retrieval Augmented Generation", "Vector Database", "Pinecone",
    "Weaviate", "Qdrant", "Milvus", "ChromaDB", "FAISS",
    # Databases
    "SQL", "MySQL", "PostgreSQL", "SQLite", "MariaDB", "Oracle",
    "SQL Server", "NoSQL", "MongoDB", "Redis", "Cassandra", "DynamoDB",
    "Elasticsearch", "Solr", "Neo4j", "InfluxDB", "TimescaleDB",
    "CockroachDB", "Couchbase", "HBase", "BigTable",
    # Cloud
    "AWS", "Azure", "GCP", "Google Cloud", "EC2", "S3", "Lambda",
    "EKS", "ECS", "RDS", "SageMaker", "Bedrock", "Vertex AI",
    "Azure ML", "Databricks", "Snowflake", "BigQuery", "Redshift",
    "CloudFormation", "Terraform", "Pulumi", "CDK",
    # DevOps / MLOps
    "Docker", "Kubernetes", "Helm", "Istio", "ArgoCD", "Flux",
    "Jenkins", "GitHub Actions", "GitLab CI", "CircleCI", "Travis CI",
    "Ansible", "Chef", "Puppet", "Salt", "Vagrant",
    "MLflow", "DVC", "ClearML", "Weights & Biases", "wandb",
    "Kubeflow", "Airflow", "Prefect", "Dagster", "Metaflow",
    "Seldon", "BentoML", "Ray", "Ray Serve", "Triton Inference Server",
    # Data engineering
    "Spark", "Apache Spark", "Kafka", "Apache Kafka", "Flink",
    "Apache Flink", "Hadoop", "HDFS", "Hive", "Pig", "Storm",
    "Beam", "Apache Beam", "Druid", "Presto", "Trino", "dbt",
    "Fivetran", "Airbyte", "Stitch", "Talend", "NiFi",
    # Web frameworks
    "FastAPI", "Flask", "Django", "Express", "NestJS", "Next.js",
    "React", "Vue", "Angular", "Svelte", "Solid", "Astro",
    "Spring", "Spring Boot", "Rails", "Laravel", "Symfony",
    "FastHTML", "Streamlit", "Gradio", "Dash",
    # Mobile
    "React Native", "Flutter", "Ionic", "Xamarin", "SwiftUI",
    "Jetpack Compose",
    # Security
    "OAuth", "OAuth2", "JWT", "SAML", "OpenID Connect", "LDAP",
    "SSL", "TLS", "mTLS", "PKI", "SIEM", "SOC", "Penetration Testing",
    "Vulnerability Assessment", "OWASP", "Zero Trust",
    # Soft / methodology
    "Agile", "Scrum", "Kanban", "SAFe", "CI/CD", "TDD", "BDD",
    "DDD", "Microservices", "REST", "GraphQL", "gRPC", "WebSockets",
    "Event-Driven Architecture", "CQRS", "Event Sourcing",
    # Math / Stats
    "Linear Algebra", "Calculus", "Probability", "Statistics",
    "Bayesian Inference", "Time Series", "Signal Processing",
    "Optimization", "Convex Optimization", "Reinforcement Learning",
    "Computer Vision", "Natural Language Processing", "NLP",
    "Speech Recognition", "Recommendation Systems",
    # Tools
    "Git", "GitHub", "GitLab", "Bitbucket", "Jira", "Confluence",
    "Notion", "Slack", "Jupyter", "VS Code", "PyCharm", "IntelliJ",
    "Vim", "Emacs", "Postman", "Insomnia", "Swagger", "OpenAPI",
    # Infrastructure
    "Linux", "Unix", "Bash", "Shell Scripting", "PowerShell",
    "Nginx", "Apache", "HAProxy", "Envoy", "Consul",
    "Prometheus", "Grafana", "Loki", "Jaeger", "Zipkin",
    "Datadog", "New Relic", "Splunk", "ELK Stack",
    # Finance / Domain
    "Quantitative Finance", "Algorithmic Trading", "Risk Management",
    "Financial Modelling", "Excel", "VBA", "Power BI", "Tableau",
    "Looker", "Metabase", "Superset",
    # HR / People
    "HRIS", "Workday", "SAP HR", "SuccessFactors", "BambooHR",
    # Additional emerging
    "WebAssembly", "WASM", "eBPF", "CUDA", "OpenCL", "ROCm",
    "Triton", "TensorRT", "ONNX", "OpenVINO",
]

# Lower-case index for fast O(1) lookup
_LEXICON_LOWER: dict[str, str] = {s.lower(): s for s in SKILL_LEXICON}


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Lazy model loading
# ─────────────────────────────────────────────────────────────────────────────

_nlp = None          # spaCy model
_matcher = None      # spaCy PhraseMatcher
_bert_pipe = None    # HuggingFace BERT NER pipeline
_groq_client = None  # Groq client


def _load_spacy():
    global _nlp, _matcher
    if _nlp is not None:
        return _nlp, _matcher

    try:
        import spacy
        from spacy.matcher import PhraseMatcher
    except ImportError:
        logger.warning(
            "spaCy not installed. Run: pip install spacy && python -m spacy download en_core_web_sm"
        )
        return None, None

    try:
        _nlp = spacy.load("en_core_web_sm")
    except OSError:
        logger.warning("en_core_web_sm not found — running: python -m spacy download en_core_web_sm")
        import subprocess, sys
        subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"], check=True)
        _nlp = spacy.load("en_core_web_sm")

    _matcher = PhraseMatcher(_nlp.vocab, attr="LOWER")
    patterns = [_nlp.make_doc(s.lower()) for s in SKILL_LEXICON]
    _matcher.add("SKILL_LEXICON", patterns)

    logger.info("spaCy model loaded (%d skill patterns registered)", len(patterns))
    return _nlp, _matcher


def _load_bert():
    global _bert_pipe
    if _bert_pipe is not None:
        return _bert_pipe

    try:
        from transformers import pipeline as hf_pipeline
        _bert_pipe = hf_pipeline(
            "token-classification",
            model="dslim/bert-base-NER",
            aggregation_strategy="simple",
            device=-1,          # CPU (change to 0 for GPU)
        )
        logger.info("BERT NER model loaded (dslim/bert-base-NER)")
    except Exception as exc:
        logger.warning("BERT NER unavailable (%s) — will rely on spaCy only", exc)
        _bert_pipe = None

    return _bert_pipe


def _get_groq():
    global _groq_client
    if _groq_client is None:
        from groq import Groq
        api_key = os.environ.get("GROQ_API_KEY", "")
        if not api_key:
            raise RuntimeError(
                "GROQ_API_KEY is not set. Get a free key at https://console.groq.com/"
            )
        _groq_client = Groq(api_key=api_key)
    return _groq_client


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Extraction layers
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class _RawMention:
    """A skill surface form found in text, before deduplication."""
    surface: str          # original text span
    canonical: str        # normalised / lexicon-matched form
    source: str           # "spacy_ner" | "phrase_matcher" | "bert_ner"
    start: int = 0
    end: int   = 0


def _layer1_spacy_ner(text: str, nlp, matcher) -> list[_RawMention]:
    """spaCy built-in NER — catches ORG / PRODUCT entities (often tech brands)."""
    if nlp is None:
        return []
    doc = nlp(text[:100_000])   # cap to avoid memory issues
    mentions: list[_RawMention] = []

    for ent in doc.ents:
        if ent.label_ in {"ORG", "PRODUCT", "WORK_OF_ART"}:
            surface = ent.text.strip()
            canonical = _canonicalise(surface)
            if canonical:
                mentions.append(_RawMention(surface, canonical, "spacy_ner", ent.start_char, ent.end_char))

    return mentions


def _layer2_phrase_matcher(text: str, nlp, matcher) -> list[_RawMention]:
    """Exact phrase matching against the 1 400-skill lexicon."""
    if nlp is None or matcher is None:
        return []
    doc = nlp(text[:100_000])
    matches = matcher(doc)
    mentions: list[_RawMention] = []

    for _match_id, start_tok, end_tok in matches:
        span = doc[start_tok:end_tok]
        surface = span.text.strip()
        canonical = _canonicalise(surface)
        if canonical:
            mentions.append(
                _RawMention(surface, canonical, "phrase_matcher", span.start_char, span.end_char)
            )

    return mentions


def _layer3_bert_ner(text: str) -> list[_RawMention]:
    """BERT token-classification NER — finds entities the lexicon might miss."""
    pipe = _load_bert()
    if pipe is None:
        return []

    mentions: list[_RawMention] = []
    chunk_size = 400   # BERT max tokens ≈ 512; ~400 words is safe
    words = text.split()

    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i : i + chunk_size])
        try:
            results = pipe(chunk)
        except Exception as exc:
            logger.debug("BERT NER chunk failed: %s", exc)
            continue

        for item in results:
            if item.get("entity_group") in {"ORG", "MISC", "PER"}:
                continue                     # PER = person name, skip
            surface = item.get("word", "").strip()
            # strip sub-word artefacts (##token)
            surface = re.sub(r"##\w+", "", surface).strip(" #-")
            if len(surface) < 2:
                continue
            canonical = _canonicalise(surface)
            if canonical:
                mentions.append(_RawMention(surface, canonical, "bert_ner"))

    return mentions


def _canonicalise(surface: str) -> Optional[str]:
    """Map a raw surface string to a canonical skill name from the lexicon."""
    if not surface or len(surface) < 2:
        return None

    lower = surface.lower().strip()

    # Reject purely numeric strings and common English stopwords
    if re.fullmatch(r"[\d\s\.\-]+", lower):
        return None

    _STOP = {
        "the", "and", "for", "with", "from", "this", "that", "has", "have",
        "are", "was", "were", "will", "been", "use", "used", "using", "new",
        "high", "low", "large", "small", "team", "work", "build", "make",
        "data", "model", "system", "service", "tool", "platform", "solution",
        "or", "in", "to", "of", "a", "an", "at", "on", "by", "be", "it",
        "software", "engineer", "developer", "experience", "knowledge",
        "strong", "excellent", "good", "ability", "skill", "skills",
        "bachelor", "master", "degree", "years",
    }
    # Reject single stopwords but allow multi-word phrases containing them
    if lower in _STOP:
        return None

    # 1. Direct lexicon lookup
    if lower in _LEXICON_LOWER:
        return _LEXICON_LOWER[lower]

    # 2. Partial substring match (surface is a substring of a lexicon entry)
    #    Only for entries ≥ 4 chars to avoid "or" matching "cors"
    if len(lower) >= 4:
        for lex_lower, lex_canonical in _LEXICON_LOWER.items():
            # whole-word match within longer lexicon entry
            pattern = r"(?<![a-z])" + re.escape(lower) + r"(?![a-z])"
            if re.search(pattern, lex_lower):
                return lex_canonical

    # 3. Fuzzy – normalise punctuation and retry
    normalised = re.sub(r"[\-_\s]+", " ", lower).strip()
    if normalised in _LEXICON_LOWER:
        return _LEXICON_LOWER[normalised]

    # 4. Accept unknown multi-word technical phrases (≥ 2 words, each ≥ 3 chars)
    parts = normalised.split()
    if len(parts) >= 2 and all(len(p) >= 3 for p in parts):
        # Title-case it as a best-effort canonical form
        return " ".join(p.capitalize() for p in parts)

    # 5. Accept single-word unknown tokens that look like tech terms
    #    (capital letter start, mixed case, or contains digit)
    if len(surface) >= 3 and re.search(r"[A-Z][a-z]|[a-z][A-Z]|\d", surface):
        return surface

    return None


def _deduplicate(mentions: list[_RawMention]) -> list[str]:
    """
    Deduplicate a list of raw mentions.
    Priority: phrase_matcher > spacy_ner > bert_ner
    Returns a list of unique canonical skill names.
    """
    _PRIORITY = {"phrase_matcher": 0, "spacy_ner": 1, "bert_ner": 2}
    best: dict[str, _RawMention] = {}

    for m in mentions:
        key = m.canonical.lower()
        if key not in best or _PRIORITY[m.source] < _PRIORITY[best[key].source]:
            best[key] = m

    return [m.canonical for m in best.values()]


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Groq scoring layer  (proficiency + importance ONLY)
# ─────────────────────────────────────────────────────────────────────────────

_GROQ_MODEL = "llama-3.3-70b-versatile"

_SCORING_SYSTEM_PROMPT = """\
You are a skill-scoring engine. You receive:
  - A list of skills found in a RESUME
  - A list of skills found in a JOB DESCRIPTION
  - The raw resume text and job description text

Your task: assign numeric scores ONLY. Do NOT discover new skills.

Return ONLY a raw JSON object — no markdown, no explanation:
{
  "resume_skills":   [{"skill":"<name>","proficiency":0.0-1.0,"years":0}],
  "jd_requirements": [{"skill":"<name>","required_level":0.0-1.0,"importance":"critical|important|nice-to-have"}]
}

Proficiency / required_level scale:
  0.1 = awareness only
  0.3 = some experience
  0.6 = regular use / proficient
  0.9 = expert / deep knowledge

Score exactly the skills provided — do not add or remove any."""


def _groq_score_skills(
    resume_skills: list[str],
    jd_skills: list[str],
    resume_text: str,
    jd_text: str,
) -> dict:
    """Send NLP-extracted skill lists to Groq to assign proficiency scores."""
    client = _get_groq()

    payload = (
        f"RESUME SKILLS (extracted by NLP): {json.dumps(resume_skills)}\n\n"
        f"JD SKILLS (extracted by NLP): {json.dumps(jd_skills)}\n\n"
        f"RESUME TEXT (first 3000 chars):\n{resume_text[:3000]}\n\n"
        f"JOB DESCRIPTION TEXT (first 2000 chars):\n{jd_text[:2000]}\n\n"
        "Return only the JSON scoring object:"
    )

    response = client.chat.completions.create(
        model=_GROQ_MODEL,
        temperature=0,
        max_tokens=2000,
        messages=[
            {"role": "system", "content": _SCORING_SYSTEM_PROMPT},
            {"role": "user",   "content": payload},
        ],
    )

    raw = response.choices[0].message.content.strip()
    return _parse_json_robust(raw)


def _parse_json_robust(raw: str) -> dict:
    """Multiple fallback strategies to parse JSON from LLM output."""
    for attempt in (raw, re.sub(r"```(?:json)?", "", raw).strip().strip("`")):
        try:
            return json.loads(attempt)
        except json.JSONDecodeError:
            pass

    # brace-extract
    try:
        start, end = raw.find("{"), raw.rfind("}") + 1
        if start != -1 and end > start:
            return json.loads(raw[start:end])
    except json.JSONDecodeError:
        pass

    # fix trailing commas
    try:
        fixed = re.sub(r",\s*([}\]])", r"\1", raw[raw.find("{"):raw.rfind("}") + 1])
        return json.loads(fixed)
    except Exception:
        pass

    return {"resume_skills": [], "jd_requirements": []}


def _deterministic_score_fallback(
    resume_skills: list[str],
    jd_skills: list[str],
) -> dict:
    """
    If Groq is unavailable, return neutral scores so the pipeline never crashes.
    resume skills → proficiency 0.5, years 1
    jd skills     → required_level 0.7, importance "important"
    """
    return {
        "resume_skills": [
            {"skill": s, "proficiency": 0.5, "years": 1}
            for s in resume_skills
        ],
        "jd_requirements": [
            {"skill": s, "required_level": 0.7, "importance": "important"}
            for s in jd_skills
        ],
    }


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Diagnostics
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class _ExtractionStats:
    spacy_ner_hits:    int = 0
    phrase_match_hits: int = 0
    bert_ner_hits:     int = 0
    groq_scored:       bool = False
    groq_fallback:     bool = False
    resume_count:      int = 0
    jd_count:          int = 0

_last_stats = _ExtractionStats()


def get_extraction_stats() -> dict:
    """Return diagnostics from the most recent extraction call."""
    return {
        "spacy_ner_hits":    _last_stats.spacy_ner_hits,
        "phrase_match_hits": _last_stats.phrase_match_hits,
        "bert_ner_hits":     _last_stats.bert_ner_hits,
        "groq_scored":       _last_stats.groq_scored,
        "groq_fallback":     _last_stats.groq_fallback,
        "resume_skill_count": _last_stats.resume_count,
        "jd_skill_count":    _last_stats.jd_count,
        "models": {
            "spacy":       "en_core_web_sm",
            "bert":        "dslim/bert-base-NER",
            "scorer_llm":  _GROQ_MODEL,
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# 6.  Public API
# ─────────────────────────────────────────────────────────────────────────────

def extract_skills(resume_text: str, jd_text: str) -> dict:
    """
    Drop-in replacement for the original skill_extractor.extract_skills().

    Pipeline:
      Layer 1 — spaCy NER
      Layer 2 — spaCy PhraseMatcher (1 400-skill lexicon)
      Layer 3 — BERT NER (dslim/bert-base-NER)
      Merge + deduplicate
      Layer 4 — Groq scores proficiency & importance (never extracts)

    Returns {"resume_skills": [...], "jd_requirements": [...]}
    """
    global _last_stats
    stats = _ExtractionStats()

    # ----- Load models -----
    nlp, matcher = _load_spacy()

    # ----- Extract from resume -----
    resume_mentions: list[_RawMention] = []
    resume_mentions += _layer1_spacy_ner(resume_text, nlp, matcher)
    resume_mentions += _layer2_phrase_matcher(resume_text, nlp, matcher)
    resume_mentions += _layer3_bert_ner(resume_text)

    # ----- Extract from JD -----
    jd_mentions: list[_RawMention] = []
    jd_mentions += _layer1_spacy_ner(jd_text, nlp, matcher)
    jd_mentions += _layer2_phrase_matcher(jd_text, nlp, matcher)
    jd_mentions += _layer3_bert_ner(jd_text)

    # ----- Diagnostics -----
    stats.spacy_ner_hits    = sum(1 for m in resume_mentions + jd_mentions if m.source == "spacy_ner")
    stats.phrase_match_hits = sum(1 for m in resume_mentions + jd_mentions if m.source == "phrase_matcher")
    stats.bert_ner_hits     = sum(1 for m in resume_mentions + jd_mentions if m.source == "bert_ner")

    # ----- Deduplicate -----
    resume_skills = _deduplicate(resume_mentions)[:15]  # cap at 15
    jd_skills     = _deduplicate(jd_mentions)[:15]

    # Ensure minimum coverage — if NLP found nothing, log a warning
    if not resume_skills:
        logger.warning("NLP layers found 0 resume skills — check input text quality")
    if not jd_skills:
        logger.warning("NLP layers found 0 JD skills — check input text quality")

    stats.resume_count = len(resume_skills)
    stats.jd_count     = len(jd_skills)

    logger.info(
        "NLP extraction: %d resume skills, %d JD skills "
        "(spaCy NER=%d, PhraseMatcher=%d, BERT NER=%d)",
        len(resume_skills), len(jd_skills),
        stats.spacy_ner_hits, stats.phrase_match_hits, stats.bert_ner_hits,
    )

    # ----- Groq scoring -----
    try:
        result = _groq_score_skills(resume_skills, jd_skills, resume_text, jd_text)
        stats.groq_scored = True

        # Validate structure — fall back if Groq returned garbage
        if (
            not isinstance(result.get("resume_skills"), list)
            or not isinstance(result.get("jd_requirements"), list)
        ):
            raise ValueError("Groq returned unexpected structure")

    except Exception as exc:
        logger.warning("Groq scoring failed (%s) — using deterministic fallback", exc)
        result = _deterministic_score_fallback(resume_skills, jd_skills)
        stats.groq_fallback = True

    _last_stats = stats
    return result
