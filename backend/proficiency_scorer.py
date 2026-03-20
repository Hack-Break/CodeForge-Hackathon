"""
NeuralPath — Skill Proficiency Scoring Model
=============================================
Converts raw resume signals into quantitative proficiency scores [0.0 – 1.0].

Scoring formula:
  SkillScore = BaseScore
             + ExperienceYearsBonus
             + ProjectComplexityBonus
             + RecencyBonus
             - RecencyPenalty
             + LeadershipBonus
             + EducationBonus

Each component is bounded and the final score is clamped to [0.05, 0.98].
This prevents both "zero" (you mentioned it → you have something) and
"perfect" (no resume signal should imply mastery).
"""

from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# Scoring constants
# ─────────────────────────────────────────────────────────────────────────────

# Base scores by evidence type
BASE_SCORE = {
    "mentioned":      0.15,  # skill name appears once in resume
    "used":           0.30,  # used in a project or role
    "proficient":     0.55,  # described as proficient / regular use
    "expert":         0.80,  # described as expert / led team using it
}

# Experience years → bonus (logarithmic — diminishing returns after ~5 years)
def _years_bonus(years: float) -> float:
    if years <= 0:
        return 0.0
    return min(0.20, 0.08 * math.log1p(years))

# Project complexity signals
PROJECT_COMPLEXITY = {
    "academic":       0.00,   # university project
    "personal":       0.03,   # personal side-project
    "internship":     0.05,   # internship role
    "production":     0.12,   # shipped to production
    "scale":          0.16,   # production at scale (>1M requests / large dataset)
}

# Recency modifier (years since last used)
def _recency_modifier(years_since: float) -> float:
    if years_since <= 0.5:
        return +0.05   # used very recently
    elif years_since <= 1.0:
        return +0.02
    elif years_since <= 2.0:
        return 0.00
    elif years_since <= 4.0:
        return -0.05   # a bit stale
    elif years_since <= 6.0:
        return -0.10   # significantly stale
    else:
        return -0.15   # likely outdated

# Leadership / ownership bonus
LEADERSHIP_BONUS = {
    "none":     0.00,
    "team":     0.05,   # led a small team using this skill
    "org":      0.08,   # drove adoption across organisation
}

# Education bonus (only for foundational skills)
EDUCATION_BONUS = {
    "none":        0.00,
    "course":      0.03,   # completed relevant online course
    "degree":      0.08,   # degree with this as core subject
    "publication": 0.12,   # published research / open-source contribution
}


# ─────────────────────────────────────────────────────────────────────────────
# Proficiency Signal (input from LLM extractor)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ProficiencySignal:
    skill_name: str
    raw_llm_score: float          # 0.0–1.0 direct from LLM
    years_experience: float = 0.0
    project_complexity: str = "personal"   # see PROJECT_COMPLEXITY keys
    years_since_used: float = 0.0
    leadership_level: str = "none"
    education_level: str = "none"
    is_primary_skill: bool = False  # flagged as core skill in resume


@dataclass
class ScoredSkill:
    skill_name: str
    skill_id: str
    raw_llm_score: float
    computed_score: float           # BKT-calibrated final score
    confidence: float               # how confident we are in this score
    score_breakdown: dict           # component-level breakdown for transparency
    evidence_level: str             # "mentioned" | "used" | "proficient" | "expert"


# ─────────────────────────────────────────────────────────────────────────────
# Core scoring function
# ─────────────────────────────────────────────────────────────────────────────

def compute_proficiency_score(signal: ProficiencySignal) -> tuple[float, dict, float]:
    """
    Compute a calibrated proficiency score from raw resume signals.

    Returns:
        (computed_score, breakdown_dict, confidence)
    """

    # ── Determine base from LLM score ──────────────────────────
    if signal.raw_llm_score >= 0.75:
        evidence = "expert"
    elif signal.raw_llm_score >= 0.50:
        evidence = "proficient"
    elif signal.raw_llm_score >= 0.25:
        evidence = "used"
    else:
        evidence = "mentioned"

    base = BASE_SCORE[evidence]

    # ── Component bonuses ───────────────────────────────────────
    years_bonus     = _years_bonus(signal.years_experience)
    complexity_bonus = PROJECT_COMPLEXITY.get(signal.project_complexity, 0.03)
    recency_mod     = _recency_modifier(signal.years_since_used)
    leadership_bonus = LEADERSHIP_BONUS.get(signal.leadership_level, 0.00)
    education_bonus  = EDUCATION_BONUS.get(signal.education_level, 0.00)
    primary_bonus   = 0.05 if signal.is_primary_skill else 0.00

    # ── Weighted blend with LLM score ──────────────────────────
    # We trust LLM extraction (50%) + computed signals (50%)
    computed = (
        0.50 * signal.raw_llm_score
        + 0.50 * (base + years_bonus + complexity_bonus + recency_mod
                  + leadership_bonus + education_bonus + primary_bonus)
    )

    # ── Clamp to valid range ────────────────────────────────────
    computed = max(0.05, min(0.98, computed))

    # ── Confidence: higher when more signals available ──────────
    n_signals = sum([
        signal.years_experience > 0,
        signal.project_complexity != "personal",
        signal.years_since_used > 0,
        signal.leadership_level != "none",
        signal.education_level != "none",
    ])
    confidence = min(0.95, 0.55 + n_signals * 0.08)

    breakdown = {
        "llm_score":        round(signal.raw_llm_score, 3),
        "base_evidence":    round(base, 3),
        "years_bonus":      round(years_bonus, 3),
        "complexity_bonus": round(complexity_bonus, 3),
        "recency_modifier": round(recency_mod, 3),
        "leadership_bonus": round(leadership_bonus, 3),
        "education_bonus":  round(education_bonus, 3),
        "primary_bonus":    round(primary_bonus, 3),
        "final_score":      round(computed, 3),
    }

    return round(computed, 3), breakdown, round(confidence, 3)


# ─────────────────────────────────────────────────────────────────────────────
# Batch scorer — called from the main pipeline
# ─────────────────────────────────────────────────────────────────────────────

def score_resume_skills(
    resume_skills: list[dict],
    skill_resolver,          # callable: str → Optional[str]
) -> dict[str, ScoredSkill]:
    """
    Score all skills extracted from a resume.
    `skill_resolver` maps free-form skill names to canonical graph IDs.

    Returns dict: skill_id → ScoredSkill
    """
    scored: dict[str, ScoredSkill] = {}

    for raw in resume_skills:
        name = (
            raw.get("skill")
            or raw.get("skill_name")
            or raw.get("name")
            or "unknown"
        )
        skill_id = skill_resolver(name) or _slugify(name)

        signal = ProficiencySignal(
            skill_name=name,
            raw_llm_score=float(raw.get("proficiency", 0.3)),
            years_experience=float(raw.get("years", 0.0)),
            project_complexity=_infer_complexity(raw),
            years_since_used=float(raw.get("years_since_used", 0.0)),
            leadership_level=_infer_leadership(raw),
            education_level=_infer_education(raw),
            is_primary_skill=bool(raw.get("is_primary", False)),
        )

        computed, breakdown, confidence = compute_proficiency_score(signal)

        scored[skill_id] = ScoredSkill(
            skill_name=name,
            skill_id=skill_id,
            raw_llm_score=signal.raw_llm_score,
            computed_score=computed,
            confidence=confidence,
            score_breakdown=breakdown,
            evidence_level=(
                "expert" if computed >= 0.75 else
                "proficient" if computed >= 0.50 else
                "used" if computed >= 0.25 else
                "mentioned"
            ),
        )

    return scored


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _infer_complexity(raw: dict) -> str:
    """Infer project complexity from LLM-provided hints."""
    hint = str(raw.get("context", "") + raw.get("project_type", "")).lower()
    if any(k in hint for k in ["production", "deployed", "shipped", "enterprise"]):
        if any(k in hint for k in ["scale", "million", "large", "high-traffic"]):
            return "scale"
        return "production"
    if any(k in hint for k in ["internship", "intern"]):
        return "internship"
    if any(k in hint for k in ["academic", "university", "thesis", "coursework"]):
        return "academic"
    return "personal"


def _infer_leadership(raw: dict) -> str:
    hint = str(raw.get("context", "") + raw.get("role", "")).lower()
    if any(k in hint for k in ["led", "lead", "managed team", "architected", "drove adoption"]):
        if any(k in hint for k in ["organisation", "company", "org-wide"]):
            return "org"
        return "team"
    return "none"


def _infer_education(raw: dict) -> str:
    hint = str(raw.get("education", "") + raw.get("certification", "")).lower()
    if any(k in hint for k in ["published", "paper", "research", "open source"]):
        return "publication"
    if any(k in hint for k in ["degree", "bachelor", "master", "phd"]):
        return "degree"
    if any(k in hint for k in ["course", "certificate", "coursera", "udemy", "certification"]):
        return "course"
    return "none"


def _slugify(text: str) -> str:
    import re
    text = str(text).lower().strip()
    text = re.sub(r"[^a-z0-9\s\-]", "", text)
    text = re.sub(r"\s+", "-", text)
    return text[:50]
