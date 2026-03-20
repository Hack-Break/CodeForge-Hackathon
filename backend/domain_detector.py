"""
NeuralPath — Domain Detector & Cross-Domain Generaliser
=========================================================
Classifies a job description into one of 7 supported domains
and returns domain-specific scoring weights and O*NET role codes.

This is the "Secret Winning Feature" — supports ANY job role,
not just Software Engineer.

Supported domains:
  software       → SWE, Backend, Frontend, Full-Stack
  ml             → ML Engineer, Data Scientist, AI Researcher
  cloud          → DevOps, SRE, Platform Engineer, Cloud Architect
  data-eng       → Data Engineer, Analytics Engineer
  security       → Security Engineer, Pen Tester
  product        → Product Manager, Program Manager
  data-analyst   → Data Analyst, BI Developer
  hr             → HR Business Partner, Talent Acquisition
  marketing      → Growth, Performance Marketer
  finance        → Financial Analyst, FP&A
  operations     → Supply Chain, Warehouse, Operations Manager
  general        → Catch-all for unknown roles
"""

from __future__ import annotations
import re
from dataclasses import dataclass, field


# ─────────────────────────────────────────────────────────────────────────────
# Domain profiles
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class DomainProfile:
    domain_id: str
    display_name: str
    keywords: list[str]          # trigger keywords in JD title / text
    primary_skill_tags: list[str]  # skills that are EXTRA important in this domain
    onet_roles: list[str]          # O*NET SOC codes
    difficulty_bias: float = 1.0   # multiplier applied to difficulty scores
    description: str = ""


DOMAIN_PROFILES: list[DomainProfile] = [
    DomainProfile(
        "ml",
        "Machine Learning / AI",
        keywords=[
            "machine learning", "deep learning", "ai engineer", "data scientist",
            "ml engineer", "nlp", "computer vision", "llm", "artificial intelligence",
            "research scientist", "applied scientist", "generative ai", "rag",
        ],
        primary_skill_tags=["pytorch", "tensorflow", "scikit-learn", "transformers", "mlops"],
        onet_roles=["15-2051.00", "15-1221.00"],
        difficulty_bias=1.1,
        description="ML / AI roles with focus on model building and experimentation",
    ),
    DomainProfile(
        "software",
        "Software Engineering",
        keywords=[
            "software engineer", "backend", "frontend", "full stack", "full-stack",
            "developer", "swe", "software development", "web developer", "api",
        ],
        primary_skill_tags=["python", "javascript", "system-design", "docker", "sql"],
        onet_roles=["15-1252.00", "15-1251.00"],
        difficulty_bias=1.0,
        description="General software engineering — backend, frontend, full-stack",
    ),
    DomainProfile(
        "cloud",
        "Cloud / DevOps / Platform",
        keywords=[
            "devops", "sre", "platform engineer", "cloud engineer", "infrastructure",
            "kubernetes", "terraform", "ci/cd", "reliability", "site reliability",
        ],
        primary_skill_tags=["kubernetes", "terraform", "aws", "docker", "observability"],
        onet_roles=["15-1244.00", "15-1231.00"],
        difficulty_bias=1.05,
        description="Cloud infrastructure, DevOps, SRE and platform engineering",
    ),
    DomainProfile(
        "data-eng",
        "Data Engineering",
        keywords=[
            "data engineer", "analytics engineer", "etl", "data pipeline",
            "spark", "kafka", "airflow", "dbt", "data platform", "lakehouse",
        ],
        primary_skill_tags=["spark", "kafka", "airflow", "sql-advanced", "data-warehousing"],
        onet_roles=["15-1243.00"],
        difficulty_bias=1.05,
        description="Data pipeline, warehouse and streaming infrastructure",
    ),
    DomainProfile(
        "security",
        "Cybersecurity",
        keywords=[
            "security engineer", "pen tester", "penetration", "cybersecurity",
            "infosec", "appsec", "devsecops", "red team", "blue team",
        ],
        primary_skill_tags=["security-fundamentals", "cloud-security", "appsec"],
        onet_roles=["15-1212.00"],
        difficulty_bias=1.1,
        description="Application, cloud and infrastructure security",
    ),
    DomainProfile(
        "product",
        "Product Management",
        keywords=[
            "product manager", "product owner", "program manager", "pm",
            "product management", "roadmap", "okr",
        ],
        primary_skill_tags=["product-management", "agile-scrum", "data-analytics"],
        onet_roles=["11-2021.00"],
        difficulty_bias=0.9,
        description="Product strategy, roadmapping and cross-functional leadership",
    ),
    DomainProfile(
        "data-analyst",
        "Data Analyst / BI",
        keywords=[
            "data analyst", "business analyst", "bi developer", "business intelligence",
            "tableau", "power bi", "looker", "analytics", "reporting",
        ],
        primary_skill_tags=["sql-basics", "data-analytics", "data-viz", "excel-advanced"],
        onet_roles=["15-2041.00", "13-1111.00"],
        difficulty_bias=0.9,
        description="Data analysis, reporting and business intelligence",
    ),
    DomainProfile(
        "hr",
        "Human Resources",
        keywords=[
            "hr", "human resources", "talent acquisition", "recruiter",
            "people ops", "people operations", "hrbp", "talent management",
        ],
        primary_skill_tags=["hr-analytics", "excel-advanced", "communication"],
        onet_roles=["13-1071.00", "13-1072.00"],
        difficulty_bias=0.85,
        description="HR, talent management and people operations",
    ),
    DomainProfile(
        "marketing",
        "Marketing / Growth",
        keywords=[
            "marketing", "growth", "performance marketing", "digital marketing",
            "seo", "sem", "demand generation", "brand", "marketing analyst",
        ],
        primary_skill_tags=["marketing-analytics", "data-analytics", "a-b-testing"],
        onet_roles=["13-1161.00"],
        difficulty_bias=0.9,
        description="Marketing analytics, growth and performance campaigns",
    ),
    DomainProfile(
        "finance",
        "Finance / FP&A",
        keywords=[
            "financial analyst", "fp&a", "finance", "investment", "accounting",
            "financial modelling", "dcf", "controller",
        ],
        primary_skill_tags=["financial-analysis", "excel-advanced", "sql-basics"],
        onet_roles=["13-2051.00", "13-2011.00"],
        difficulty_bias=0.95,
        description="Financial analysis, modelling and planning",
    ),
    DomainProfile(
        "operations",
        "Operations / Supply Chain",
        keywords=[
            "operations", "supply chain", "logistics", "warehouse", "inventory",
            "procurement", "ops manager", "plant manager",
        ],
        primary_skill_tags=["supply-chain", "excel-advanced", "project-management"],
        onet_roles=["11-3071.00", "11-9199.01"],
        difficulty_bias=0.85,
        description="Operations management, supply chain and logistics",
    ),
    DomainProfile(
        "general",
        "General / Other",
        keywords=[],
        primary_skill_tags=[],
        onet_roles=[],
        difficulty_bias=1.0,
        description="Catch-all for roles not matching a specific domain",
    ),
]

_DOMAIN_LOOKUP: dict[str, DomainProfile] = {d.domain_id: d for d in DOMAIN_PROFILES}


# ─────────────────────────────────────────────────────────────────────────────
# Detection logic
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class DetectionResult:
    domain_id: str
    display_name: str
    confidence: float           # 0.0–1.0
    matched_keywords: list[str]
    profile: DomainProfile


def detect_domain(jd_text: str, jd_title: str = "") -> DetectionResult:
    """
    Classify a job description into a domain profile.
    Uses keyword scoring — each keyword hit adds to domain score.
    Returns the best matching domain with confidence.
    """
    combined = (jd_title + " " + jd_text).lower()

    scores: dict[str, float] = {}
    matched: dict[str, list[str]] = {}

    for profile in DOMAIN_PROFILES:
        if profile.domain_id == "general":
            continue
        hit_count = 0
        hits = []
        for kw in profile.keywords:
            if re.search(r"\b" + re.escape(kw) + r"\b", combined):
                hit_count += 1
                hits.append(kw)
        # Title match is worth 3× body match
        for kw in profile.keywords:
            if re.search(r"\b" + re.escape(kw) + r"\b", jd_title.lower()):
                hit_count += 2
                if kw not in hits:
                    hits.append(kw + " (title)")
        scores[profile.domain_id] = hit_count
        matched[profile.domain_id] = hits

    if not any(v > 0 for v in scores.values()):
        return DetectionResult(
            domain_id="general",
            display_name="General / Other",
            confidence=0.40,
            matched_keywords=[],
            profile=_DOMAIN_LOOKUP["general"],
        )

    best_domain = max(scores, key=lambda k: scores[k])
    best_score  = scores[best_domain]
    total_hits  = sum(scores.values())
    confidence  = min(0.97, 0.50 + (best_score / max(1, total_hits)) * 0.50)

    return DetectionResult(
        domain_id=best_domain,
        display_name=_DOMAIN_LOOKUP[best_domain].display_name,
        confidence=round(confidence, 3),
        matched_keywords=matched[best_domain],
        profile=_DOMAIN_LOOKUP[best_domain],
    )


def get_domain_profile(domain_id: str) -> DomainProfile:
    return _DOMAIN_LOOKUP.get(domain_id, _DOMAIN_LOOKUP["general"])
