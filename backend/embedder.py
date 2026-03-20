"""
NeuralPath — Embedder / O*NET Skill Matcher
Maps extracted skill names to stable O*NET-style IDs via slug normalization.
Handles any key format Claude might return gracefully.

In production this would use pgvector + text-embedding-3-small for
semantic similarity. For the current release, deterministic slug matching
gives clean, reliable results without an embeddings API dependency.
"""
import re


def match_skills_to_onet(skill_map: dict) -> dict:
    """
    Maps each skill name to a stable slug-based O*NET ID.
    Enriches each entry with normalized onet_id, skill_name, and proficiency.
    """
    result: dict = {
        "resume_skills":   [],
        "jd_requirements": [],
    }

    for skill in skill_map.get("resume_skills", []):
        name = (
            skill.get("skill")
            or skill.get("skill_name")
            or skill.get("name")
            or "unknown"
        )
        proficiency = _clamp(float(skill.get("proficiency", 0.3)))

        enriched = dict(skill)
        enriched["onet_id"]    = _slugify(name)
        enriched["skill_name"] = name
        enriched["proficiency"] = proficiency
        result["resume_skills"].append(enriched)

    for req in skill_map.get("jd_requirements", []):
        name = (
            req.get("skill")
            or req.get("skill_name")
            or req.get("name")
            or "unknown"
        )
        required_level = _clamp(float(req.get("required_level", 0.7)))

        enriched = dict(req)
        enriched["onet_id"]       = _slugify(name)
        enriched["skill_name"]    = name
        enriched["required_level"] = required_level
        result["jd_requirements"].append(enriched)

    return result


def _slugify(text: str) -> str:
    """Convert a skill name to a stable, lowercase slug."""
    text = str(text).lower().strip()
    text = re.sub(r"[^a-z0-9\s\-]", "", text)
    text = re.sub(r"\s+", "-", text)
    text = re.sub(r"-+", "-", text).strip("-")
    return text[:50]


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))
