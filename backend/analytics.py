"""
NeuralPath — Analytics & Visualisation Data Builders
======================================================
Produces structured data for two key UI visualisations:

  1. Skill Radar Chart  — per-domain proficiency vs requirement
  2. Roadmap Timeline   — ordered weekly/phase-based learning plan
  3. Time Saved Summary — traditional vs adaptive comparison card

All outputs are JSON-serialisable dicts ready for the API response.
"""

from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Any

from .optimizer import PathPlan, PathStep
from .gap_analyzer import GapResult


# ─────────────────────────────────────────────────────────────────────────────
# 1. Skill Radar Chart
# ─────────────────────────────────────────────────────────────────────────────

RADAR_AXES = [
    "Foundations",
    "Classical ML",
    "Deep Learning",
    "Cloud / DevOps",
    "Data Engineering",
    "NLP / LLM",
    "MLOps / Production",
    "Software Engineering",
    "Security",
    "Product / Analytics",
]

# Map skill domain + id → radar axis
_DOMAIN_AXIS_MAP: dict[str, str] = {
    "foundations":       "Foundations",
    "ml":                "Classical ML",
    "cloud":             "Cloud / DevOps",
    "data-eng":          "Data Engineering",
    "security":          "Security",
    "product":           "Product / Analytics",
    "general":           "Product / Analytics",
    "software":          "Software Engineering",
}

_SKILL_AXIS_OVERRIDE: dict[str, str] = {
    "deep-learning-fundamentals": "Deep Learning",
    "pytorch":                    "Deep Learning",
    "tensorflow":                 "Deep Learning",
    "cnn":                        "Deep Learning",
    "rnn-lstm":                   "Deep Learning",
    "transformers":               "Deep Learning",
    "nlp-classical":              "NLP / LLM",
    "nlp-transformers":           "NLP / LLM",
    "llm-fundamentals":           "NLP / LLM",
    "llm-fine-tuning":            "NLP / LLM",
    "rag":                        "NLP / LLM",
    "langchain":                  "NLP / LLM",
    "mlops-fundamentals":         "MLOps / Production",
    "model-serving":              "MLOps / Production",
    "ml-monitoring":              "MLOps / Production",
    "feature-store":              "MLOps / Production",
    "math-foundations":           "Foundations",
    "statistics-ml":              "Foundations",
    "numpy-pandas":               "Foundations",
    "data-preprocessing":         "Foundations",
    "classical-ml":               "Classical ML",
    "model-evaluation":           "Classical ML",
    "gradient-boosting":          "Classical ML",
    "sklearn-advanced":           "Classical ML",
    "time-series-ml":             "Classical ML",
    "reinforcement-learning":     "Deep Learning",
    "computer-vision":            "Deep Learning",
    "generative-ai":              "Deep Learning",
}


def build_radar_data(
    gap_map: dict[str, GapResult],
    domain_map: dict[str, str],    # skill_id → domain
) -> dict[str, Any]:
    """
    Build radar chart data with per-axis current vs required scores.

    Returns:
    {
      "axes": [...],
      "current":  [0.4, 0.6, ...],   # per axis — user's level
      "required": [0.7, 0.8, ...],   # per axis — JD target
    }
    """
    axis_current:  dict[str, list[float]] = {ax: [] for ax in RADAR_AXES}
    axis_required: dict[str, list[float]] = {ax: [] for ax in RADAR_AXES}

    for skill_id, gap in gap_map.items():
        # Determine axis
        axis = _SKILL_AXIS_OVERRIDE.get(skill_id)
        if axis is None:
            domain = domain_map.get(skill_id, "general")
            axis = _DOMAIN_AXIS_MAP.get(domain, "Software Engineering")

        axis_current[axis].append(gap.proficiency_current)
        axis_required[axis].append(gap.proficiency_required)

    def avg(vals: list[float]) -> float:
        return round(sum(vals) / len(vals), 3) if vals else 0.0

    current_scores  = [avg(axis_current[ax])  for ax in RADAR_AXES]
    required_scores = [avg(axis_required[ax]) for ax in RADAR_AXES]

    # Filter to axes that have data
    active_indices = [i for i, (c, r) in enumerate(zip(current_scores, required_scores)) if c > 0 or r > 0]
    active_axes    = [RADAR_AXES[i] for i in active_indices]
    active_current = [current_scores[i]  for i in active_indices]
    active_required = [required_scores[i] for i in active_indices]

    return {
        "axes":     active_axes,
        "current":  active_current,
        "required": active_required,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 2. Roadmap Timeline
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TimelinePhase:
    phase: int
    label: str
    weeks: str
    modules: list[dict]
    phase_hours: float
    domain_focus: str


def build_roadmap_timeline(plan: PathPlan, hours_per_week: float = 10.0) -> dict[str, Any]:
    """
    Convert a PathPlan into a phase-based learning timeline.

    Phases:
      Phase 1: Foundations (seq 0–20)     — Prerequisites
      Phase 2: Core Skills (seq 21–35)    — Primary gap modules
      Phase 3: Advanced (seq 36–50)       — Specialisation
      Phase 4: Production (seq 51+)       — MLOps / deployment

    Each phase shows:
      - Week range
      - Module list with hours
      - Domain focus
    """
    active = [s for s in plan.pathway if s.action != "SKIP"]

    if not active:
        return {"phases": [], "total_weeks": 0, "hours_per_week": hours_per_week}

    # Group modules into phases based on difficulty and domain
    def phase_for(step: PathStep) -> int:
        if step.difficulty <= 1:
            return 1   # Foundations
        if step.difficulty == 2:
            return 2   # Core
        if step.difficulty <= 3:
            return 3   # Intermediate
        return 4       # Advanced / Expert

    phases: dict[int, list[PathStep]] = {1: [], 2: [], 3: [], 4: []}
    for step in active:
        phases[phase_for(step)].append(step)

    # Remove empty phases
    phase_labels = {
        1: "Foundations & Prerequisites",
        2: "Core Skills",
        3: "Intermediate / Specialisation",
        4: "Advanced & Production",
    }

    result_phases = []
    cumulative_weeks = 0.0

    for p_num in [1, 2, 3, 4]:
        steps = phases[p_num]
        if not steps:
            continue

        phase_hours = sum(s.estimated_hours for s in steps)
        phase_weeks = math.ceil(phase_hours / hours_per_week)

        start_week = int(cumulative_weeks) + 1
        end_week   = int(cumulative_weeks) + phase_weeks
        cumulative_weeks += phase_weeks

        # Dominant domain in this phase
        domain_counts: dict[str, int] = {}
        for s in steps:
            domain_counts[s.domain] = domain_counts.get(s.domain, 0) + 1
        domain_focus = max(domain_counts, key=lambda k: domain_counts[k])

        result_phases.append({
            "phase":        p_num,
            "label":        phase_labels[p_num],
            "weeks":        f"Week {start_week}–{end_week}",
            "week_start":   start_week,
            "week_end":     end_week,
            "phase_hours":  round(phase_hours, 1),
            "domain_focus": domain_focus,
            "modules": [
                {
                    "module_id":   s.module_id,
                    "module_name": s.module_name,
                    "action":      s.action,
                    "hours":       s.estimated_hours,
                    "difficulty":  s.difficulty,
                    "confidence":  s.confidence,
                }
                for s in steps
            ],
        })

    total_weeks = int(cumulative_weeks)

    return {
        "phases":          result_phases,
        "total_weeks":     total_weeks,
        "total_hours":     round(sum(s.estimated_hours for s in active), 1),
        "hours_per_week":  hours_per_week,
        "estimated_completion": f"{total_weeks} weeks at {hours_per_week}h/week",
    }


# ─────────────────────────────────────────────────────────────────────────────
# 3. Time Saved Summary
# ─────────────────────────────────────────────────────────────────────────────

def build_time_saved_summary(plan: PathPlan) -> dict[str, Any]:
    """
    Produce the "Time Saved" metrics card.

    Output:
    {
      "traditional_hours": 120,
      "adaptive_hours": 68,
      "hours_saved": 52,
      "time_saved_pct": 43.3,
      "modules_skipped": 4,
      "modules_fast_tracked": 3,
      "label": "43% faster than traditional onboarding",
      "breakdown": [...]
    }
    """
    pathway = plan.pathway
    required    = [s for s in pathway if s.action == "REQUIRED"]
    fast_track  = [s for s in pathway if s.action == "FAST_TRACK"]
    skipped     = [s for s in pathway if s.action == "SKIP"]

    adapt_h = sum(s.estimated_hours   for s in required + fast_track)
    trad_h  = sum(s.traditional_hours for s in required + fast_track)
    skip_saved = sum(s.traditional_hours for s in skipped)
    total_trad = trad_h + skip_saved
    total_saved = total_trad - adapt_h

    pct = round((total_saved / total_trad * 100) if total_trad > 0 else 0.0, 1)

    label = (
        f"{pct:.0f}% faster than traditional onboarding"
        if pct > 0
        else "Personalised pathway — no redundant training"
    )

    breakdown = []
    for s in sorted(pathway, key=lambda x: x.hours_saved, reverse=True)[:8]:
        if s.action == "SKIP":
            saving_label = f"Skipped entirely — {s.traditional_hours}h saved"
        elif s.action == "FAST_TRACK":
            saving_label = f"Fast-tracked — {s.hours_saved}h saved vs full module"
        else:
            saving_label = f"Full module — {s.estimated_hours}h"
        breakdown.append({
            "module_name":  s.module_name,
            "action":       s.action,
            "adaptive_h":  s.estimated_hours,
            "traditional_h": s.traditional_hours,
            "saved_h":      s.hours_saved,
            "label":        saving_label,
        })

    return {
        "traditional_hours":    round(total_trad, 1),
        "adaptive_hours":       round(adapt_h, 1),
        "hours_saved":          round(total_saved, 1),
        "time_saved_pct":       pct,
        "modules_skipped":      len(skipped),
        "modules_fast_tracked": len(fast_track),
        "modules_required":     len(required),
        "label":                label,
        "breakdown":            breakdown,
    }
