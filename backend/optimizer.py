"""
NeuralPath — Adaptive Learning Path Optimizer
==============================================
Implements 3 planning algorithms on the Knowledge Graph:

  1. Dijkstra  — shortest path (minimise total learning hours)
  2. A*        — heuristic-guided (prioritise critical JD skills)
  3. DP        — full coverage (ensures no dependency is missed)

The optimizer selects the best algorithm based on the size and
connectivity of the skill gap, then returns a fully ordered
PathPlan with time savings and confidence metrics.
"""

from __future__ import annotations
import math
import heapq
import logging
from dataclasses import dataclass, field
from typing import Optional

import networkx as nx

from .knowledge_graph import KNOWLEDGE_GRAPH, SKILL_LOOKUP, get_prerequisite_chain
from .gap_analyzer import GapResult

logger = logging.getLogger(__name__)

# Virtual graph nodes
START_NODE = "__START__"
END_NODE   = "__END__"

# Difficulty multiplier map (graph node difficulty 1–5 → hour multiplier)
DIFFICULTY_MULTIPLIER = {1: 0.75, 2: 0.90, 3: 1.00, 4: 1.20, 5: 1.45}

# Traditional (non-adaptive) onboarding assumes 100% of hours for all REQUIRED skills
TRADITIONAL_OVERHEAD = 1.0


# ─────────────────────────────────────────────────────────────────────────────
# Data models
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PathStep:
    module_id:            str
    module_name:          str
    action:               str           # "REQUIRED" | "FAST_TRACK" | "SKIP"
    domain:               str
    difficulty:           int           # 1–5
    gap_score:            float         # adjusted gap [0,1]
    proficiency_current:  float
    proficiency_required: float
    estimated_hours:      float
    traditional_hours:    float         # what it would cost without adaptation
    hours_saved:          float         # traditional − adaptive
    confidence:           float         # recommendation confidence [0,1]
    prerequisites:        list[str] = field(default_factory=list)
    reason:               str = ""
    score_breakdown:      dict = field(default_factory=dict)


@dataclass
class PathPlan:
    pathway:              list[PathStep]
    algorithm_used:       str           # "dijkstra" | "astar" | "dp"
    total_adaptive_hours: float
    total_traditional_hours: float
    total_hours_saved:    float
    time_saved_pct:       float
    competency_coverage:  float         # % of JD requirements addressed
    overall_confidence:   float
    domain_breakdown:     dict[str, int]  # domain → module count
    summary: dict = field(default_factory=dict)


# ─────────────────────────────────────────────────────────────────────────────
# Hours estimation
# ─────────────────────────────────────────────────────────────────────────────

def _adaptive_hours(gap: GapResult, difficulty: int) -> float:
    """
    Hours adjusted for:
      - gap severity (bigger gap = more time)
      - difficulty multiplier
      - fast-track reduction (partial knowledge → 60% of full module)
    """
    if gap.action == "SKIP":
        return 0.0

    # Get base hours from knowledge graph if available
    node_data = KNOWLEDGE_GRAPH.nodes.get(gap.skill_id, {})
    base = node_data.get("base_hours") or max(4.0, gap.adjusted_gap * 28)

    mult = DIFFICULTY_MULTIPLIER.get(difficulty, 1.0)
    hours = base * mult

    if gap.action == "FAST_TRACK":
        hours *= 0.60   # 40% time saving for partial knowledge

    if gap.importance == "critical":
        hours *= 1.10   # critical skills get a slight buffer

    return round(min(hours, 32.0), 1)   # cap at 32h per module


def _traditional_hours(gap: GapResult, difficulty: int) -> float:
    """
    Full-module hours as if no prior knowledge existed.
    SKIP modules also get a value — a static curriculum would include them;
    the adaptive saving comes from skipping them entirely.
    """
    node_data = KNOWLEDGE_GRAPH.nodes.get(gap.skill_id, {})
    base = node_data.get("base_hours") or max(4.0, gap.proficiency_required * 28)
    mult = DIFFICULTY_MULTIPLIER.get(difficulty, 1.0)
    return round(min(base * mult, 32.0), 1)


def _confidence(gap: GapResult, scored_skill) -> float:
    """
    Recommendation confidence combines:
      - gap certainty (how precise is the gap score?)
      - scoring confidence (how many resume signals did we have?)
    """
    scoring_conf = getattr(scored_skill, "confidence", 0.70) if scored_skill else 0.70
    gap_certainty = 0.90 if gap.importance == "critical" else 0.75
    return round((scoring_conf + gap_certainty) / 2, 3)


# ─────────────────────────────────────────────────────────────────────────────
# Prerequisite expansion
# ─────────────────────────────────────────────────────────────────────────────

def _expand_with_prerequisites(
    gap_map: dict[str, GapResult],
    scored_skills: dict,
) -> dict[str, GapResult]:
    """
    For every REQUIRED / FAST_TRACK skill, walk up the knowledge graph
    and inject any unmet prerequisites as REQUIRED (if user has no score)
    or SKIP (if already met).

    This ensures the plan is always learnable — no orphaned advanced modules.
    """
    expanded = dict(gap_map)

    for skill_id, gap in list(gap_map.items()):
        if gap.action == "SKIP":
            continue

        chain = get_prerequisite_chain(skill_id)

        for prereq_id in chain:
            if prereq_id == skill_id or prereq_id in expanded:
                continue

            # Does the user already have this prerequisite?
            current = scored_skills.get(prereq_id)
            current_score = getattr(current, "computed_score", 0.0) if current else 0.0
            prereq_node = KNOWLEDGE_GRAPH.nodes.get(prereq_id, {})
            required_level = max(0.4, prereq_node.get("difficulty", 2) * 0.15)

            if current_score >= required_level - 0.05:
                action = "SKIP"
            elif current_score >= required_level - 0.25:
                action = "FAST_TRACK"
            else:
                action = "REQUIRED"

            raw_gap = max(0.0, required_level - current_score)
            adj_gap = raw_gap * 1.05 if 0 < current_score < required_level else raw_gap

            from .gap_analyzer import GapResult as GR
            expanded[prereq_id] = GR(
                skill_id=prereq_id,
                skill_name=prereq_node.get("name", prereq_id),
                proficiency_current=round(current_score, 3),
                proficiency_required=round(required_level, 3),
                raw_gap=round(raw_gap, 3),
                adjusted_gap=round(adj_gap, 3),
                action=action,
                importance="important",
            )

    return expanded


# ─────────────────────────────────────────────────────────────────────────────
# Build adaptive subgraph
# ─────────────────────────────────────────────────────────────────────────────

def _build_adaptive_graph(gap_map: dict[str, GapResult]) -> nx.DiGraph:
    """
    Build a weighted DAG containing only the modules in the gap map.
    Edges come from the master knowledge graph (prerequisite chains).
    """
    G = nx.DiGraph()
    G.add_node(START_NODE)
    G.add_node(END_NODE)

    actionable = {sid: gap for sid, gap in gap_map.items() if gap.action != "SKIP"}

    for sid, gap in actionable.items():
        node_data = KNOWLEDGE_GRAPH.nodes.get(sid, {})
        difficulty = node_data.get("difficulty", 2)
        hours = _adaptive_hours(gap, difficulty)

        G.add_node(
            sid,
            gap=gap,
            difficulty=difficulty,
            estimated_hours=hours,
            weight=hours,
            domain=node_data.get("domain", "general"),
        )

    # Add edges from master graph where both endpoints exist in our subgraph
    for u, v in KNOWLEDGE_GRAPH.edges():
        if u in G and v in G:
            G.add_edge(u, v, weight=G.nodes[v]["weight"])

    # Wire to START / END
    for sid in actionable:
        if G.has_node(sid) and not any(
            p for p in G.predecessors(sid) if p != START_NODE
        ):
            G.add_edge(START_NODE, sid, weight=0)
        if G.has_node(sid) and not any(
            s for s in G.successors(sid) if s != END_NODE
        ):
            G.add_edge(sid, END_NODE, weight=0)

    return G


# ─────────────────────────────────────────────────────────────────────────────
# Algorithm 1: Dijkstra — minimum total learning hours
# ─────────────────────────────────────────────────────────────────────────────

def _dijkstra_path(G: nx.DiGraph) -> list[str]:
    try:
        return nx.dijkstra_path(G, START_NODE, END_NODE, weight="weight")
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return []


# ─────────────────────────────────────────────────────────────────────────────
# Algorithm 2: A* — heuristic prioritises critical JD gaps
# ─────────────────────────────────────────────────────────────────────────────

def _astar_path(G: nx.DiGraph, gap_map: dict[str, GapResult]) -> list[str]:
    """
    A* with a heuristic that rewards covering critical skills faster.
    h(node) = 1 / (gap_score + 0.01) for REQUIRED critical nodes
    """
    def heuristic(node, target):
        if node in (START_NODE, END_NODE):
            return 0.0
        gap = gap_map.get(node)
        if gap is None:
            return 0.0
        # Lower heuristic for high-gap, critical nodes → A* prioritises them
        if gap.importance == "critical" and gap.action == "REQUIRED":
            return -gap.adjusted_gap * 5   # negative → priority boost
        return 0.0

    try:
        return nx.astar_path(G, START_NODE, END_NODE, heuristic=heuristic, weight="weight")
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return _dijkstra_path(G)


# ─────────────────────────────────────────────────────────────────────────────
# Algorithm 3: Dynamic Programming — full coverage path
# ─────────────────────────────────────────────────────────────────────────────

def _dp_full_coverage(G: nx.DiGraph, gap_map: dict[str, GapResult]) -> list[str]:
    """
    DP approach: topological sort with priority weighting.
    Ensures 100% of REQUIRED nodes are included, then orders them
    to minimise redundant learning (prerequisite always before dependent).
    """
    try:
        topo = [n for n in nx.topological_sort(G) if n not in (START_NODE, END_NODE)]
    except nx.NetworkXUnfeasible:
        topo = [n for n in G.nodes if n not in (START_NODE, END_NODE)]

    # Sort within topological layers: REQUIRED critical first, then REQUIRED, then FAST_TRACK
    def priority(node_id):
        gap = gap_map.get(node_id)
        if gap is None:
            return (3, 0)
        if gap.action == "REQUIRED" and gap.importance == "critical":
            return (0, -gap.adjusted_gap)
        if gap.action == "REQUIRED":
            return (1, -gap.adjusted_gap)
        return (2, -gap.adjusted_gap)

    return sorted(topo, key=priority)


# ─────────────────────────────────────────────────────────────────────────────
# Main optimizer
# ─────────────────────────────────────────────────────────────────────────────

def optimize_learning_path(
    gap_map: dict[str, GapResult],
    scored_skills: dict,
    algorithm: str = "auto",
) -> PathPlan:
    """
    Main entry point for path optimization.

    Parameters:
        gap_map       : skill_id → GapResult from gap_analyzer
        scored_skills : skill_id → ScoredSkill from proficiency_scorer
        algorithm     : "dijkstra" | "astar" | "dp" | "auto"

    Returns PathPlan with full ordered pathway + metrics.
    """

    # ── Step 1: Expand with prerequisite chain ──────────────────
    expanded_gap_map = _expand_with_prerequisites(gap_map, scored_skills)

    # ── Step 2: Build adaptive subgraph ────────────────────────
    G = _build_adaptive_graph(expanded_gap_map)
    n_required = sum(1 for g in expanded_gap_map.values() if g.action == "REQUIRED")

    # ── Step 3: Select algorithm ────────────────────────────────
    if algorithm == "auto":
        if n_required <= 3:
            algo = "dijkstra"       # small graph — fastest path
        elif n_required <= 8:
            algo = "astar"          # medium — prioritise critical skills
        else:
            algo = "dp"             # large — full coverage guaranteed
    else:
        algo = algorithm

    logger.info(f"PathOptimizer: {n_required} required modules, using {algo.upper()}")

    # ── Step 4: Find ordered path ───────────────────────────────
    if algo == "dijkstra":
        ordered_ids = _dijkstra_path(G)
    elif algo == "astar":
        ordered_ids = _astar_path(G, expanded_gap_map)
    else:
        ordered_ids = _dp_full_coverage(G, expanded_gap_map)

    # Remove virtual nodes
    ordered_ids = [n for n in ordered_ids if n not in (START_NODE, END_NODE)]

    # ── Step 5: Ensure ALL actionable modules are included ──────
    # (Dijkstra/A* may miss parallel branches)
    seen = set(ordered_ids)
    for sid, gap in expanded_gap_map.items():
        if gap.action != "SKIP" and sid not in seen:
            ordered_ids.append(sid)
            seen.add(sid)

    # ── Step 6: Build PathStep objects ──────────────────────────
    steps: list[PathStep] = []
    skip_steps: list[PathStep] = []

    for sid in ordered_ids:
        gap = expanded_gap_map.get(sid)
        if gap is None:
            continue

        node_data = KNOWLEDGE_GRAPH.nodes.get(sid, {})
        difficulty = node_data.get("difficulty", 2)
        domain = node_data.get("domain", "general")

        scored = scored_skills.get(sid)
        adapt_hours = _adaptive_hours(gap, difficulty)
        trad_hours  = _traditional_hours(gap, difficulty)
        conf        = _confidence(gap, scored)

        prereqs = [
            p for p in G.predecessors(sid)
            if p not in (START_NODE, END_NODE)
        ] if G.has_node(sid) else []

        breakdown = getattr(scored, "score_breakdown", {}) if scored else {}

        steps.append(PathStep(
            module_id=sid,
            module_name=gap.skill_name,
            action=gap.action,
            domain=domain,
            difficulty=difficulty,
            gap_score=gap.adjusted_gap,
            proficiency_current=gap.proficiency_current,
            proficiency_required=gap.proficiency_required,
            estimated_hours=adapt_hours,
            traditional_hours=trad_hours,
            hours_saved=max(0.0, round(trad_hours - adapt_hours, 1)),
            confidence=conf,
            prerequisites=prereqs,
            score_breakdown=breakdown,
        ))

    # ── Step 7: Add SKIP steps ───────────────────────────────────
    for sid, gap in expanded_gap_map.items():
        if gap.action == "SKIP" and sid not in {s.module_id for s in steps}:
            node_data = KNOWLEDGE_GRAPH.nodes.get(sid, {})
            cur  = f"{gap.proficiency_current:.0%}"
            req  = f"{gap.proficiency_required:.0%}"
            skip_steps.append(PathStep(
                module_id=sid,
                module_name=gap.skill_name,
                action="SKIP",
                domain=node_data.get("domain", "general"),
                difficulty=node_data.get("difficulty", 2),
                gap_score=0.0,
                proficiency_current=gap.proficiency_current,
                proficiency_required=gap.proficiency_required,
                estimated_hours=0.0,
                traditional_hours=_traditional_hours(gap, node_data.get("difficulty", 2)),
                hours_saved=max(0.0, _traditional_hours(gap, node_data.get("difficulty", 2))),
                confidence=0.92,
                prerequisites=[],
                reason=f"Already proficient ({cur} ≥ required {req}) — module skipped.",
            ))

    # ── Step 8: Sort final pathway respecting prerequisites ──────
    # Use topological sort on the adaptive subgraph, then sort within
    # each "topological layer" by priority (REQUIRED critical first).
    try:
        topo_order = [
            n for n in nx.topological_sort(G)
            if n not in (START_NODE, END_NODE) and n in {s.module_id for s in steps}
        ]
    except nx.NetworkXUnfeasible:
        topo_order = [s.module_id for s in steps]

    # Build a lookup for fast ordering
    topo_pos = {mid: i for i, mid in enumerate(topo_order)}

    # Sort: first by topological position (prerequisite safety),
    # then by priority within same position
    def _sort_key(s: PathStep):
        pos = topo_pos.get(s.module_id, 999)
        action_tier = 0 if s.action == "REQUIRED" else (1 if s.action == "FAST_TRACK" else 2)
        return (pos, action_tier, -s.gap_score)

    steps.sort(key=_sort_key)

    full_pathway = steps + skip_steps

    # ── Step 9: Compute plan-level metrics ───────────────────────
    # Traditional = what it would cost to learn ONLY the original gap_map skills
    # (not the injected prerequisites) without any adaptation.
    original_ids      = set(gap_map.keys())
    adaptive_total    = sum(s.estimated_hours for s in steps)
    traditional_total = sum(
        s.traditional_hours for s in steps
        if s.module_id in original_ids
    )
    # Add traditional cost for expanded prereqs (user would still need them)
    traditional_total += sum(
        s.traditional_hours for s in steps
        if s.module_id not in original_ids
    )
    hours_saved      = max(0.0, traditional_total - adaptive_total)
    time_saved_pct   = max(0.0, (hours_saved / traditional_total * 100) if traditional_total > 0 else 0.0)

    n_jd_total = len([g for g in expanded_gap_map.values() if g.action != "SKIP" or True])
    n_addressed = len([g for g in expanded_gap_map.values()])
    coverage = min(1.0, n_addressed / max(1, n_jd_total))

    avg_conf = (sum(s.confidence for s in steps) / len(steps)) if steps else 0.0

    domain_breakdown: dict[str, int] = {}
    for s in full_pathway:
        domain_breakdown[s.domain] = domain_breakdown.get(s.domain, 0) + 1

    required_count   = sum(1 for s in full_pathway if s.action == "REQUIRED")
    fast_track_count = sum(1 for s in full_pathway if s.action == "FAST_TRACK")
    skip_count       = sum(1 for s in full_pathway if s.action == "SKIP")

    plan = PathPlan(
        pathway=full_pathway,
        algorithm_used=algo,
        total_adaptive_hours=round(adaptive_total, 1),
        total_traditional_hours=round(traditional_total, 1),
        total_hours_saved=round(hours_saved, 1),
        time_saved_pct=round(time_saved_pct, 1),
        competency_coverage=round(coverage, 3),
        overall_confidence=round(avg_conf, 3),
        domain_breakdown=domain_breakdown,
        summary={
            "total_modules":   len(full_pathway),
            "required":        required_count,
            "fast_track":      fast_track_count,
            "skipped":         skip_count,
            "estimated_hours": round(adaptive_total, 1),
            "traditional_hours": round(traditional_total, 1),
            "hours_saved":     round(hours_saved, 1),
            "time_saved_pct":  round(time_saved_pct, 1),
            "algorithm":       algo,
            "coverage":        round(coverage * 100, 1),
        },
    )

    logger.info(
        f"PathPlan complete: {required_count} req, {fast_track_count} fast, {skip_count} skip. "
        f"Adaptive={round(adaptive_total,1)}h vs Traditional={round(traditional_total,1)}h "
        f"(saved {round(time_saved_pct,1)}%)"
    )

    return plan
