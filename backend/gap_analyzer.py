"""
NeuralPath — Gap Analyzer (BKT-inspired)
Classifies each skill as SKIP / FAST_TRACK / REQUIRED based on
the delta between current proficiency and JD requirements.
"""
import os
from dataclasses import dataclass
from typing import Literal

GAP_SKIP_THRESHOLD = float(os.getenv("GAP_SKIP_THRESHOLD", "0.10"))
GAP_FAST_THRESHOLD = float(os.getenv("GAP_FAST_THRESHOLD", "0.30"))
BKT_SLIP_FACTOR    = float(os.getenv("BKT_SLIP_FACTOR",    "0.85"))

ActionType = Literal["SKIP", "FAST_TRACK", "REQUIRED"]


@dataclass
class GapResult:
    skill_id:             str
    skill_name:           str
    proficiency_current:  float
    proficiency_required: float
    raw_gap:              float
    adjusted_gap:         float
    action:               ActionType
    importance:           str


def compute_gap_map(matched_skills: dict) -> dict[str, GapResult]:
    """
    For each JD requirement, compute the gap vs. resume proficiency.
    Applies BKT slip-factor adjustment for partial knowledge.
    """
    gap_map: dict[str, GapResult] = {}

    jd_reqs = {
        s.get("onet_id", s.get("skill", "")): s
        for s in matched_skills.get("jd_requirements", [])
    }
    res_skills = {
        s.get("onet_id", s.get("skill", "")): s
        for s in matched_skills.get("resume_skills", [])
    }

    for onet_id, req in jd_reqs.items():
        if not onet_id:
            continue

        current  = max(0.0, min(1.0, float(res_skills.get(onet_id, {}).get("proficiency", 0.0))))
        required = max(0.0, min(1.0, float(req.get("required_level", 0.7))))

        raw_gap = max(0.0, required - current)

        # BKT slip: partial knowledge may be overconfident
        adj_gap = raw_gap * (2.0 - BKT_SLIP_FACTOR) if 0 < current < required else raw_gap

        if adj_gap <= GAP_SKIP_THRESHOLD:
            action: ActionType = "SKIP"
        elif adj_gap <= GAP_FAST_THRESHOLD:
            action = "FAST_TRACK"
        else:
            action = "REQUIRED"

        skill_name = req.get("skill_name") or req.get("skill") or onet_id

        gap_map[onet_id] = GapResult(
            skill_id=onet_id,
            skill_name=skill_name,
            proficiency_current=round(current, 3),
            proficiency_required=round(required, 3),
            raw_gap=round(raw_gap, 3),
            adjusted_gap=round(adj_gap, 3),
            action=action,
            importance=req.get("importance", "important"),
        )

    return gap_map
