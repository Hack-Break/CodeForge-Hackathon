"""
NeuralPath — Reasoning Trace Engine
=====================================
Generates transparent chain-of-thought reasoning for every
SKIP / FAST_TRACK / REQUIRED decision.

Each trace shows:
  1. WHY the module was recommended / skipped
  2. WHAT evidence from the resume drove the decision
  3. WHICH JD requirement it satisfies
  4. WHAT prerequisite dependencies exist
  5. HOW CONFIDENT the system is

Groq (Llama 3.3 70B) enriches traces for non-trivial plans.
Deterministic fallbacks are always available for speed / offline use.
"""

from __future__ import annotations
import os
import logging
from dataclasses import dataclass

from groq import Groq

from .optimizer import PathStep, PathPlan
from .knowledge_graph import KNOWLEDGE_GRAPH

logger = logging.getLogger(__name__)

MODEL = "llama-3.3-70b-versatile"

_client: Groq | None = None


def _get_client() -> Groq:
    global _client
    if _client is None:
        key = os.environ.get("GROQ_API_KEY", "")
        if not key or key in ("dummy", ""):
            raise RuntimeError(
                "GROQ_API_KEY not set. "
                "Get a free key at https://console.groq.com/"
            )
        _client = Groq(api_key=key)
    return _client


# ─────────────────────────────────────────────────────────────────────────────
# Deterministic trace builder (zero API cost, always available)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ReasoningTrace:
    module_name:     str
    action:          str
    why_recommended: str
    jd_alignment:    str
    resume_evidence: str
    dependency_note: str
    confidence_note: str
    full_trace:      str


def _det_trace(step: PathStep, gap_map: dict) -> ReasoningTrace:
    """Build a deterministic reasoning trace from gap data alone."""
    cur  = f"{step.proficiency_current:.0%}"
    req  = f"{step.proficiency_required:.0%}"
    diff_label = {1: "Beginner", 2: "Beginner–Intermediate", 3: "Intermediate",
                  4: "Advanced",  5: "Expert"}.get(step.difficulty, "Intermediate")
    conf_pct = f"{step.confidence:.0%}"

    if step.action == "REQUIRED":
        why = (
            f"Significant knowledge gap detected: your current proficiency ({cur}) "
            f"is well below the role requirement ({req}). Full module required."
        )
    elif step.action == "FAST_TRACK":
        why = (
            f"Partial knowledge detected: your current proficiency ({cur}) is close "
            f"to the requirement ({req}). An accelerated module covers only the delta."
        )
    else:
        why = (
            f"You already meet or exceed this requirement: your proficiency ({cur}) "
            f"satisfies the role benchmark ({req}). Module skipped."
        )

    gap = gap_map.get(step.module_id)
    importance   = getattr(gap, "importance", "important") if gap else "important"
    jd_alignment = (
        f"This skill is marked '{importance}' in the job description. "
        f"Gap score: {step.gap_score:.2f}."
    )

    breakdown    = step.score_breakdown or {}
    llm_score    = breakdown.get("llm_score", 0)
    years_bonus  = breakdown.get("years_bonus", 0)
    resume_evidence = (
        f"Resume analysis gave a raw LLM proficiency score of {llm_score:.2f}. "
        + (f"Experience signal added {years_bonus:.2f}. " if years_bonus > 0
           else "No multi-year experience detected. ")
        + f"After BKT adjustment: {step.proficiency_current:.2f}."
    )

    if step.prerequisites:
        prereq_names = []
        for pid in step.prerequisites[:3]:
            pdata = KNOWLEDGE_GRAPH.nodes.get(pid, {})
            prereq_names.append(pdata.get("name", pid))
        dep_note = f"Requires completing: {', '.join(prereq_names)} first."
    else:
        dep_note = "No prerequisite dependencies in this plan."

    conf_note = (
        f"Recommendation confidence: {conf_pct}. "
        + ("High confidence — clear gap evidence from multiple resume signals."
           if step.confidence >= 0.80
           else "Moderate confidence — limited signals; recommendation based on JD analysis.")
    )

    full = (
        f"[{step.action}] {step.module_name} ({diff_label}, {step.estimated_hours}h)\n"
        f"→ {why}\n"
        f"→ JD: {jd_alignment}\n"
        f"→ Evidence: {resume_evidence}\n"
        f"→ Dependencies: {dep_note}\n"
        f"→ Confidence: {conf_note}"
    )

    return ReasoningTrace(
        module_name=step.module_name,
        action=step.action,
        why_recommended=why,
        jd_alignment=jd_alignment,
        resume_evidence=resume_evidence,
        dependency_note=dep_note,
        confidence_note=conf_note,
        full_trace=full,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Groq-powered batch trace enrichment
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are explaining a personalized training recommendation to a new hire.
For each numbered module, write exactly TWO sentences:
  Sentence 1: WHY this action (REQUIRED / FAST_TRACK / SKIP) was chosen, referencing the exact proficiency numbers.
  Sentence 2: WHAT learning this module will unlock or why it is already covered.

Rules:
- Reference actual numbers (current %, required %, gap score).
- Mention the job description importance level.
- Do NOT invent skills or requirements not in the data.
- Keep each pair under 50 words total.
- Number responses exactly like: 1. sentence1 sentence2\
"""


def _build_batch_prompt(steps: list[PathStep]) -> str:
    items = []
    for i, s in enumerate(steps, 1):
        items.append(
            f"[{i}]\n"
            f"Module: {s.module_name}\n"
            f"Action: {s.action}\n"
            f"Current: {s.proficiency_current:.0%}  Required: {s.proficiency_required:.0%}\n"
            f"Gap score: {s.gap_score:.2f}  Confidence: {s.confidence:.2f}\n"
            f"Estimated hours: {s.estimated_hours}h"
        )
    return "\n\n---\n\n".join(items)


def _parse_numbered_responses(raw: str, n: int) -> dict[int, str]:
    reasons: dict[int, str] = {}
    for line in raw.split("\n"):
        line = line.strip()
        if line and line[0].isdigit() and ". " in line:
            idx, _, text = line.partition(". ")
            try:
                k = int(idx)
                if 1 <= k <= n:
                    reasons[k - 1] = text.strip()
            except ValueError:
                pass
    return reasons


def enrich_traces_with_claude(
    plan: PathPlan,
    gap_map: dict,
    use_llm: bool = True,
) -> PathPlan:
    """
    Attach reasoning traces to every PathStep in the plan.
    Deterministic traces are built first (always), then Groq enriches
    the REQUIRED and FAST_TRACK modules if use_llm=True.
    """
    # Build deterministic traces for all steps first
    for step in plan.pathway:
        trace = _det_trace(step, gap_map)
        step.reason = trace.why_recommended

    if not use_llm:
        return plan

    # Groq enrichment — REQUIRED and FAST_TRACK only (saves tokens)
    enrichable = [s for s in plan.pathway if s.action != "SKIP"]
    if not enrichable:
        return plan

    try:
        client = _get_client()
        prompt = _build_batch_prompt(enrichable)

        response = client.chat.completions.create(
            model=MODEL,
            max_tokens=2000,
            temperature=0,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": prompt},
            ],
        )

        raw     = response.choices[0].message.content.strip()
        reasons = _parse_numbered_responses(raw, len(enrichable))

        for i, step in enumerate(enrichable):
            if i in reasons:
                step.reason = reasons[i]

        logger.info(f"Groq enriched {len(reasons)}/{len(enrichable)} traces")

    except Exception as e:
        logger.warning(f"Groq trace enrichment failed — using deterministic fallbacks: {e}")

    return plan
