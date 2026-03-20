"""
NeuralPath — ML/DL Dedicated Pathway Engine
============================================
Generates a structured, opinionated ML/DL learning roadmap
from a candidate's current skill level and target specialisation.

Unlike the generic /analyze endpoint (which is JD-driven),
this endpoint uses a curated curriculum graph with 5 specialisation tracks:

  Track A: Classical ML → Production (Data Scientist path)
  Track B: Deep Learning → Computer Vision
  Track C: Deep Learning → NLP / LLMs / Agents
  Track D: Deep Learning → MLOps / Platform
  Track E: Reinforcement Learning (Research path)

The engine:
  1. Assesses the candidate's current ML level (0–5)
  2. Selects the appropriate entry point in the curriculum
  3. Computes which modules to SKIP / FAST_TRACK / REQUIRE
  4. Returns an ordered, dependency-safe roadmap with hours
"""

from __future__ import annotations
import logging
from dataclasses import dataclass, field
from typing import Literal

from .knowledge_graph import KNOWLEDGE_GRAPH, SKILL_LOOKUP, resolve_skill_id
from .proficiency_scorer import score_resume_skills, ProficiencySignal, compute_proficiency_score
from .gap_analyzer import GapResult, GAP_SKIP_THRESHOLD, GAP_FAST_THRESHOLD, BKT_SLIP_FACTOR
from .optimizer import PathStep, PathPlan, optimize_learning_path, _adaptive_hours, _traditional_hours

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# ML/DL Curriculum: ordered modules per track
# ─────────────────────────────────────────────────────────────────────────────

MLTrack = Literal["classical", "computer-vision", "nlp-llm", "mlops", "rl"]


@dataclass
class CurriculumModule:
    skill_id: str
    required_proficiency: float     # what level the track requires
    importance: str                 # "critical" | "important" | "nice-to-have"
    track: list[str]                # which tracks include this module
    sequence: int                   # ordering within the track (lower = earlier)


# Full ML/DL curriculum — every module tagged with which tracks need it
ML_CURRICULUM: list[CurriculumModule] = [

    # ── Tier 0: Universal foundations (ALL tracks) ────────────
    CurriculumModule("math-foundations",            0.60, "critical",     ["classical", "computer-vision", "nlp-llm", "mlops", "rl"], 10),
    CurriculumModule("statistics-ml",               0.65, "critical",     ["classical", "computer-vision", "nlp-llm", "mlops", "rl"], 11),
    CurriculumModule("python-basics",               0.70, "critical",     ["classical", "computer-vision", "nlp-llm", "mlops", "rl"], 12),
    CurriculumModule("numpy-pandas",                0.70, "critical",     ["classical", "computer-vision", "nlp-llm", "mlops", "rl"], 13),
    CurriculumModule("data-viz",                    0.55, "important",    ["classical", "computer-vision", "nlp-llm", "mlops", "rl"], 14),
    CurriculumModule("data-preprocessing",          0.70, "critical",     ["classical", "computer-vision", "nlp-llm", "mlops", "rl"], 15),

    # ── Tier 1: Classical ML (ALL tracks need this) ───────────
    CurriculumModule("classical-ml",                0.70, "critical",     ["classical", "computer-vision", "nlp-llm", "mlops", "rl"], 20),
    CurriculumModule("model-evaluation",            0.70, "critical",     ["classical", "computer-vision", "nlp-llm", "mlops", "rl"], 21),
    CurriculumModule("gradient-boosting",           0.65, "important",    ["classical", "mlops"],                                     22),

    # ── Tier 2: Deep Learning Core (DL tracks) ────────────────
    CurriculumModule("deep-learning-fundamentals",  0.75, "critical",     ["computer-vision", "nlp-llm", "mlops", "rl"],              30),
    CurriculumModule("pytorch",                     0.75, "critical",     ["computer-vision", "nlp-llm", "rl"],                       31),
    CurriculumModule("tensorflow",                  0.55, "important",    ["computer-vision", "mlops"],                               32),

    # ── Tier 3A: Computer Vision track ───────────────────────
    CurriculumModule("cnn",                         0.80, "critical",     ["computer-vision"],                                        40),
    CurriculumModule("computer-vision",             0.75, "critical",     ["computer-vision"],                                        41),
    CurriculumModule("generative-ai",               0.65, "important",    ["computer-vision"],                                        42),

    # ── Tier 3B: NLP / LLM track ─────────────────────────────
    CurriculumModule("rnn-lstm",                    0.65, "important",    ["nlp-llm"],                                                40),
    CurriculumModule("transformers",                0.80, "critical",     ["nlp-llm"],                                                41),
    CurriculumModule("nlp-classical",               0.65, "important",    ["nlp-llm"],                                                42),
    CurriculumModule("nlp-transformers",            0.80, "critical",     ["nlp-llm"],                                                43),
    CurriculumModule("llm-fundamentals",            0.75, "critical",     ["nlp-llm"],                                                44),
    CurriculumModule("llm-fine-tuning",             0.70, "critical",     ["nlp-llm"],                                                45),
    CurriculumModule("rag",                         0.70, "critical",     ["nlp-llm"],                                                46),
    CurriculumModule("langchain",                   0.65, "important",    ["nlp-llm"],                                                47),

    # ── Tier 3C: Classical ML production (classical track) ───
    CurriculumModule("sklearn-advanced",            0.65, "important",    ["classical"],                                              40),
    CurriculumModule("time-series-ml",              0.60, "important",    ["classical"],                                              41),

    # ── Tier 4: MLOps (ALL production tracks) ────────────────
    CurriculumModule("mlops-fundamentals",          0.70, "critical",     ["classical", "computer-vision", "nlp-llm", "mlops"],       50),
    CurriculumModule("model-serving",               0.70, "critical",     ["classical", "computer-vision", "nlp-llm", "mlops"],       51),
    CurriculumModule("ml-monitoring",               0.60, "important",    ["mlops"],                                                  52),
    CurriculumModule("feature-store",               0.60, "important",    ["mlops"],                                                  53),
    CurriculumModule("docker",                      0.65, "important",    ["classical", "computer-vision", "nlp-llm", "mlops"],       54),

    # ── Tier 4: RL track ─────────────────────────────────────
    CurriculumModule("reinforcement-learning",      0.80, "critical",     ["rl"],                                                     50),
]

# Build lookup: skill_id → CurriculumModule
_CURRICULUM_LOOKUP: dict[str, CurriculumModule] = {m.skill_id: m for m in ML_CURRICULUM}


# ─────────────────────────────────────────────────────────────────────────────
# Level detection
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class MLLevelAssessment:
    level: int                          # 0–5
    label: str                          # "Beginner" ... "Expert"
    foundation_score: float             # avg score on Tier-0 skills
    dl_score: float                     # avg score on DL fundamentals
    specialisation_score: float         # avg score on advanced skills
    strongest_area: str
    weakest_area: str
    recommended_entry_sequence: int     # which sequence number to start from


def assess_ml_level(scored_skills: dict) -> MLLevelAssessment:
    """
    Assess the candidate's overall ML/DL level (0–5)
    from their scored skills.
    """
    tier0_ids   = ["math-foundations", "statistics-ml", "python-basics", "numpy-pandas", "data-preprocessing"]
    tier1_ids   = ["classical-ml", "model-evaluation", "gradient-boosting"]
    tier2_ids   = ["deep-learning-fundamentals", "pytorch", "tensorflow"]
    tier3_ids   = ["cnn", "transformers", "nlp-transformers", "rnn-lstm", "reinforcement-learning"]
    tier4_ids   = ["mlops-fundamentals", "model-serving", "llm-fundamentals", "llm-fine-tuning"]

    def avg_score(ids: list[str]) -> float:
        scores = [
            scored_skills[sid].computed_score
            for sid in ids
            if sid in scored_skills
        ]
        return sum(scores) / len(scores) if scores else 0.0

    foundation_score     = avg_score(tier0_ids)
    classical_score      = avg_score(tier1_ids)
    dl_score             = avg_score(tier2_ids)
    specialisation_score = avg_score(tier3_ids + tier4_ids)

    # Determine overall level
    avg = (foundation_score + classical_score + dl_score + specialisation_score) / 4

    if avg >= 0.80:
        level, label = 5, "Expert"
        entry_seq = 40
    elif avg >= 0.65:
        level, label = 4, "Advanced"
        entry_seq = 30
    elif avg >= 0.45:
        level, label = 3, "Intermediate"
        entry_seq = 20
    elif avg >= 0.25:
        level, label = 2, "Beginner–Intermediate"
        entry_seq = 10
    elif avg >= 0.10:
        level, label = 1, "Beginner"
        entry_seq = 10
    else:
        level, label = 0, "No ML Background"
        entry_seq = 10

    # Strongest / weakest
    area_scores = {
        "Foundations":     foundation_score,
        "Classical ML":    classical_score,
        "Deep Learning":   dl_score,
        "Advanced/MLOps":  specialisation_score,
    }
    strongest = max(area_scores, key=lambda k: area_scores[k])
    weakest   = min(area_scores, key=lambda k: area_scores[k])

    return MLLevelAssessment(
        level=level,
        label=label,
        foundation_score=round(foundation_score, 3),
        dl_score=round(dl_score, 3),
        specialisation_score=round(specialisation_score, 3),
        strongest_area=strongest,
        weakest_area=weakest,
        recommended_entry_sequence=entry_seq,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Track selector
# ─────────────────────────────────────────────────────────────────────────────

TRACK_DESCRIPTIONS = {
    "classical":       "Classical ML → Production (Data Scientist / Analyst path)",
    "computer-vision": "Deep Learning → Computer Vision (CV Engineer path)",
    "nlp-llm":         "Deep Learning → NLP / LLMs / Agents (AI Engineer path)",
    "mlops":           "MLOps & ML Platform Engineer (Production ML path)",
    "rl":              "Reinforcement Learning (Research Scientist path)",
}

def infer_track_from_jd(jd_text: str, skill_map: dict) -> MLTrack:
    """
    Infer the best ML track from JD keywords and required skills.
    """
    jd_lower = jd_text.lower()

    track_signals: dict[str, int] = {
        "nlp-llm":         0,
        "computer-vision": 0,
        "mlops":           0,
        "rl":              0,
        "classical":       0,
    }

    nlp_kws  = ["nlp", "llm", "language model", "bert", "gpt", "transformers", "rag",
                "chatbot", "text", "natural language", "fine-tuning", "langchain"]
    cv_kws   = ["computer vision", "image", "cnn", "yolo", "opencv", "object detection",
                "segmentation", "video", "visual"]
    ops_kws  = ["mlops", "mlflow", "kubeflow", "serving", "deployment", "pipeline",
                "platform", "monitoring", "feature store", "drift"]
    rl_kws   = ["reinforcement", "reward", "policy", "agent", "ppo", "dqn", "gym",
                "simulation", "game"]
    classical = ["data scientist", "analytics", "prediction", "forecasting", "regression",
                 "classification", "tabular", "sklearn", "xgboost"]

    for kw in nlp_kws:
        if kw in jd_lower: track_signals["nlp-llm"] += 2
    for kw in cv_kws:
        if kw in jd_lower: track_signals["computer-vision"] += 2
    for kw in ops_kws:
        if kw in jd_lower: track_signals["mlops"] += 2
    for kw in rl_kws:
        if kw in jd_lower: track_signals["rl"] += 3
    for kw in classical:
        if kw in jd_lower: track_signals["classical"] += 1

    # Also check required skills
    for req in skill_map.get("jd_requirements", []):
        skill = req.get("skill", "").lower()
        if any(k in skill for k in ["nlp", "bert", "llm", "transformers", "rag"]):
            track_signals["nlp-llm"] += 3
        elif any(k in skill for k in ["cnn", "vision", "yolo", "image"]):
            track_signals["computer-vision"] += 3
        elif any(k in skill for k in ["mlops", "serving", "monitoring", "pipeline"]):
            track_signals["mlops"] += 3
        elif any(k in skill for k in ["reinforcement", "rl", "ppo"]):
            track_signals["rl"] += 3

    best = max(track_signals, key=lambda k: track_signals[k])
    return best if track_signals[best] > 0 else "classical"


# ─────────────────────────────────────────────────────────────────────────────
# ML/DL pathway builder
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class MLPathwayResponse:
    track: str
    track_description: str
    level_assessment: MLLevelAssessment
    plan: PathPlan
    curriculum_modules_total: int
    modules_in_track: int


def build_mldl_pathway(
    resume_skills: list[dict],
    jd_text: str,
    jd_requirements: list[dict],
    force_track: str | None = None,
) -> MLPathwayResponse:
    """
    Build a dedicated ML/DL learning pathway.

    Steps:
      1. Score all ML-relevant skills from resume
      2. Assess current ML level (0–5)
      3. Detect target track from JD (or use force_track)
      4. Filter curriculum to track modules
      5. Compute SKIP / FAST_TRACK / REQUIRED for each module
      6. Run Dijkstra on the resulting gap_map
      7. Return ordered plan
    """

    # ── 1. Score resume skills against ML curriculum ───────────
    all_ml_skill_ids = list(_CURRICULUM_LOOKUP.keys())

    # Build a full skill map with zero scores for anything not in resume
    from .embedder import match_skills_to_onet
    matched = match_skills_to_onet({
        "resume_skills":   resume_skills,
        "jd_requirements": jd_requirements,
    })

    scored_skills = score_resume_skills(
        matched["resume_skills"],
        skill_resolver=resolve_skill_id,
    )

    # Fill in zeros for ML skills not in resume
    for skill_id in all_ml_skill_ids:
        if skill_id not in scored_skills:
            from .proficiency_scorer import ScoredSkill
            scored_skills[skill_id] = ScoredSkill(
                skill_name=SKILL_LOOKUP[skill_id].name if skill_id in SKILL_LOOKUP else skill_id,
                skill_id=skill_id,
                raw_llm_score=0.0,
                computed_score=0.0,
                confidence=0.55,
                score_breakdown={},
                evidence_level="mentioned",
            )

    # ── 2. Assess level ─────────────────────────────────────────
    level_assessment = assess_ml_level(scored_skills)
    logger.info(f"ML level: {level_assessment.label} (L{level_assessment.level}), entry at seq {level_assessment.recommended_entry_sequence}")

    # ── 3. Select track ─────────────────────────────────────────
    track = force_track or infer_track_from_jd(jd_text, {
        "resume_skills":   resume_skills,
        "jd_requirements": jd_requirements,
    })
    logger.info(f"ML track selected: {track}")

    # ── 4. Filter curriculum to this track ─────────────────────
    track_modules = [
        m for m in ML_CURRICULUM
        if track in m.track
    ]
    track_modules.sort(key=lambda m: m.sequence)

    logger.info(f"Track '{track}' has {len(track_modules)} modules")

    # ── 5. Build gap_map from curriculum ───────────────────────
    gap_map: dict[str, GapResult] = {}

    for module in track_modules:
        sid     = module.skill_id
        current = scored_skills.get(sid)
        current_score = current.computed_score if current else 0.0
        required      = module.required_proficiency

        raw_gap = max(0.0, required - current_score)
        adj_gap = raw_gap * (2.0 - BKT_SLIP_FACTOR) if 0 < current_score < required else raw_gap

        if adj_gap <= GAP_SKIP_THRESHOLD:
            action = "SKIP"
        elif adj_gap <= GAP_FAST_THRESHOLD:
            action = "FAST_TRACK"
        else:
            action = "REQUIRED"

        node_data  = KNOWLEDGE_GRAPH.nodes.get(sid, {})
        skill_name = node_data.get("name") or (SKILL_LOOKUP[sid].name if sid in SKILL_LOOKUP else sid)

        gap_map[sid] = GapResult(
            skill_id=sid,
            skill_name=skill_name,
            proficiency_current=round(current_score, 3),
            proficiency_required=round(required, 3),
            raw_gap=round(raw_gap, 3),
            adjusted_gap=round(adj_gap, 3),
            action=action,
            importance=module.importance,
        )

    # ── 6. Optimise path ────────────────────────────────────────
    # For ML/DL always use DP to ensure complete, ordered curriculum
    plan = optimize_learning_path(gap_map, scored_skills, algorithm="dp")

    # ── 7. Re-sort by curriculum sequence (not just gap severity) ─
    seq_map = {m.skill_id: m.sequence for m in track_modules}
    plan.pathway.sort(key=lambda s: (
        0 if s.action == "REQUIRED" else (1 if s.action == "FAST_TRACK" else 2),
        seq_map.get(s.module_id, 99),
    ))

    return MLPathwayResponse(
        track=track,
        track_description=TRACK_DESCRIPTIONS.get(track, track),
        level_assessment=level_assessment,
        plan=plan,
        curriculum_modules_total=len(ML_CURRICULUM),
        modules_in_track=len(track_modules),
    )
