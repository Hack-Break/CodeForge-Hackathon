# """
# NeuralPath — AI-Adaptive Onboarding Engine  v2.0
# FastAPI Backend — Research-Grade Adaptive Engine

# Endpoints:
#   GET  /api/health              → system health + graph stats
#   GET  /api/nlp/status          → NLP extraction pipeline diagnostics
#   GET  /api/graph/stats         → full knowledge graph metadata
#   GET  /api/graph/domains       → all supported domains (O*NET)
#   GET  /api/cache/stats         → cache hit/miss metrics
#   POST /api/analyze             → core adaptive pathway (any domain)
#   POST /api/pathway/mldl        → dedicated ML/DL curriculum pathway
#   POST /api/compare             → side-by-side scenario comparison
#   GET  /api/analytics/radar     → skill radar chart data
#   GET  /api/analytics/timeline  → roadmap timeline data
#   GET  /api/analytics/savings   → time saved summary card
# """

# import os
# import logging
# from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Query
# from fastapi.staticfiles import StaticFiles
# from fastapi.middleware.cors import CORSMiddleware

# from .parser             import extract_text
# from .nlp_extractor      import extract_skills, get_extraction_stats
# from .embedder           import match_skills_to_onet
# from .gap_analyzer       import (
#     compute_gap_map, GapResult,
#     GAP_SKIP_THRESHOLD, GAP_FAST_THRESHOLD, BKT_SLIP_FACTOR
# )
# from .proficiency_scorer import score_resume_skills
# from .optimizer          import optimize_learning_path, PathPlan
# from .reasoning          import enrich_traces_with_claude
# from .domain_detector    import detect_domain, DOMAIN_PROFILES
# from .knowledge_graph    import KNOWLEDGE_GRAPH, SKILL_NODES, resolve_skill_id
# from .ml_pathway         import build_mldl_pathway, TRACK_DESCRIPTIONS
# from .analytics          import build_radar_data, build_roadmap_timeline, build_time_saved_summary
# from .cache              import skill_cache, plan_cache, make_skill_key, make_plan_key
# from .models             import (
#     AnalyzeResponse, CompareResponse, HealthResponse,
#     GraphStatsResponse, SummaryModel, DomainInfoModel,
#     PathStepModel, ScoreBreakdownModel, CompareMetricsModel,
# )

# logging.basicConfig(
#     level=logging.INFO,
#     format="%(asctime)s [%(levelname)s] %(name)s — %(message)s"
# )
# logger = logging.getLogger(__name__)


# # ─────────────────────────────────────────────────────────────────────────────
# # App
# # ─────────────────────────────────────────────────────────────────────────────

# app = FastAPI(
#     title="NeuralPath API",
#     description=(
#         "**Research-grade AI-Adaptive Onboarding Engine**\n\n"
#         "- **73-node Skill Knowledge Graph** with prerequisite dependency chains\n"
#         "- **BKT Proficiency Scoring** — `years × complexity × recency × leadership`\n"
#         "- **3 Path Algorithms** — Dijkstra / A* / DP (auto-selected)\n"
#         "- **11 Domains** — ML, SWE, DevOps, Data Eng, Security, Product, HR, Finance …\n"
#         "- **Dedicated ML/DL Pathway** — 5 specialisation tracks\n"
#         "- **Scenario Comparison** — Fresher vs Senior side-by-side\n"
#     ),
#     version="3.0.0",
#     docs_url="/api/docs",
#     redoc_url="/api/redoc",
# )

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_methods=["*"],
#     allow_headers=["*"],
# )


# # ─────────────────────────────────────────────────────────────────────────────
# # Pipeline helpers
# # ─────────────────────────────────────────────────────────────────────────────

# async def _run_pipeline(
#     resume_text: str,
#     jd_text: str,
#     algorithm: str = "auto",
#     use_llm_traces: bool = True,
#     use_cache: bool = True,
# ) -> tuple[PathPlan, object, dict, dict]:
#     """
#     Full adaptive engine pipeline with caching.

#     Returns: (plan, domain_result, gap_map, scored_skills)
#     """
#     # ── Cache check ─────────────────────────────────────────────
#     plan_key = make_plan_key(resume_text, jd_text, algorithm)
#     if use_cache:
#         cached = plan_cache.get(plan_key)
#         if cached:
#             logger.info("Cache HIT — returning cached plan")
#             return cached

#     # ── 1. Domain detection ─────────────────────────────────────
#     domain_result = detect_domain(jd_text)
#     logger.info(f"Domain: {domain_result.display_name} ({domain_result.confidence:.0%})")

#     # ── 2. Skill extraction (with cache) ────────────────────────
#     skill_key = make_skill_key(resume_text, jd_text)
#     skill_map = skill_cache.get(skill_key)

#     if skill_map is None:
#         skill_map = extract_skills(resume_text, jd_text)
#         skill_cache.set(skill_key, skill_map)
#         logger.info("Skills extracted via NLP pipeline (cache MISS)")
#     else:
#         logger.info("Skills from cache (cache HIT)")

#     if not skill_map.get("jd_requirements"):
#         raise HTTPException(422, "Could not extract skills from the JD. Try a more detailed description.")

#     logger.info(
#         f"Skills: {len(skill_map.get('resume_skills', []))} resume, "
#         f"{len(skill_map.get('jd_requirements', []))} JD"
#     )

#     # ── 3. O*NET matching ───────────────────────────────────────
#     matched = match_skills_to_onet(skill_map)

#     # ── 4a. Quantitative proficiency scoring ────────────────────
#     scored_skills = score_resume_skills(
#         matched.get("resume_skills", []),
#         skill_resolver=resolve_skill_id,
#     )

#     # ── 4b. BKT gap analysis ────────────────────────────────────
#     gap_map = compute_gap_map(matched)

#     if not gap_map:
#         raise HTTPException(422, "Could not compute skill gaps. Check documents and retry.")

#     # Override gap_map with higher-quality computed scores
#     for skill_id, scored in scored_skills.items():
#         if skill_id in gap_map:
#             old      = gap_map[skill_id]
#             current  = scored.computed_score
#             required = old.proficiency_required
#             raw_gap  = max(0.0, required - current)
#             adj_gap  = raw_gap * (2.0 - BKT_SLIP_FACTOR) if 0 < current < required else raw_gap

#             if adj_gap <= GAP_SKIP_THRESHOLD:    action = "SKIP"
#             elif adj_gap <= GAP_FAST_THRESHOLD:  action = "FAST_TRACK"
#             else:                                action = "REQUIRED"

#             gap_map[skill_id] = GapResult(
#                 skill_id=old.skill_id,
#                 skill_name=old.skill_name,
#                 proficiency_current=round(current, 3),
#                 proficiency_required=old.proficiency_required,
#                 raw_gap=round(raw_gap, 3),
#                 adjusted_gap=round(adj_gap, 3),
#                 action=action,
#                 importance=old.importance,
#             )

#     # ── 5 & 6. Graph expansion + path optimisation ──────────────
#     plan = optimize_learning_path(gap_map, scored_skills, algorithm=algorithm)

#     # ── 7. Reasoning traces ─────────────────────────────────────
#     plan = enrich_traces_with_claude(plan, gap_map, use_llm=use_llm_traces)

#     result = (plan, domain_result, gap_map, scored_skills)

#     if use_cache:
#         plan_cache.set(plan_key, result)

#     return result


# def _build_step_model(step) -> PathStepModel:
#     bd = step.score_breakdown or {}
#     return PathStepModel(
#         module_id=step.module_id,
#         module_name=step.module_name,
#         action=step.action,
#         domain=step.domain,
#         difficulty=step.difficulty,
#         gap_score=step.gap_score,
#         proficiency_current=step.proficiency_current,
#         proficiency_required=step.proficiency_required,
#         estimated_hours=step.estimated_hours,
#         traditional_hours=step.traditional_hours,
#         hours_saved=step.hours_saved,
#         confidence=step.confidence,
#         prerequisites=step.prerequisites,
#         reason=step.reason,
#         score_breakdown=ScoreBreakdownModel(**bd) if bd else ScoreBreakdownModel(),
#     )


# def _plan_to_response(plan: PathPlan, domain_result) -> AnalyzeResponse:
#     s = plan.summary
#     return AnalyzeResponse(
#         summary=SummaryModel(
#             total_modules=s["total_modules"],
#             required=s["required"],
#             fast_track=s["fast_track"],
#             skipped=s["skipped"],
#             estimated_hours=s["estimated_hours"],
#             traditional_hours=s["traditional_hours"],
#             hours_saved=s["hours_saved"],
#             time_saved_pct=s["time_saved_pct"],
#             algorithm=s["algorithm"],
#             coverage=s["coverage"],
#         ),
#         pathway=[_build_step_model(step) for step in plan.pathway],
#         domain_info=DomainInfoModel(
#             domain_id=domain_result.domain_id,
#             display_name=domain_result.display_name,
#             confidence=domain_result.confidence,
#             matched_keywords=domain_result.matched_keywords,
#             onet_roles=domain_result.profile.onet_roles,
#         ),
#         graph_stats={
#             "total_nodes":             KNOWLEDGE_GRAPH.number_of_nodes(),
#             "total_edges":             KNOWLEDGE_GRAPH.number_of_edges(),
#             "algorithm_used":          plan.algorithm_used,
#             "competency_coverage_pct": round(plan.competency_coverage * 100, 1),
#             "overall_confidence":      plan.overall_confidence,
#         },
#     )


# # ─────────────────────────────────────────────────────────────────────────────
# # System endpoints
# # ─────────────────────────────────────────────────────────────────────────────

# @app.get("/api/health", response_model=HealthResponse, tags=["System"])
# async def health():
#     """Health check — model info and knowledge graph statistics."""
#     return HealthResponse(
#         status="ok",
#         model="llama-3.3-70b-versatile",
#         version="3.0.0",
#         graph_nodes=KNOWLEDGE_GRAPH.number_of_nodes(),
#         graph_edges=KNOWLEDGE_GRAPH.number_of_edges(),
#     )


# @app.get("/api/cache/stats", tags=["System"])
# async def cache_stats():
#     """Cache hit / miss metrics."""
#     return {
#         "skill_cache": skill_cache.stats(),
#         "plan_cache":  plan_cache.stats(),
#     }


# @app.delete("/api/cache", tags=["System"])
# async def clear_cache():
#     """Clear all caches (admin use)."""
#     s = skill_cache.clear()
#     p = plan_cache.clear()
#     return {"cleared": {"skill_cache": s, "plan_cache": p}}


# # ─────────────────────────────────────────────────────────────────────────────
# # Knowledge Graph endpoints
# # ─────────────────────────────────────────────────────────────────────────────

# @app.get("/api/nlp/status", tags=["System"])
# async def nlp_status():
#     """
#     NLP extraction pipeline diagnostics.
#     Returns hit counts per layer and model identifiers from the most recent
#     extract_skills() call.
#     """
#     return get_extraction_stats()


# @app.get("/api/graph/stats", response_model=GraphStatsResponse, tags=["Knowledge Graph"])
# async def graph_stats():
#     """Full knowledge graph metadata — all nodes, edges, domains, difficulty distribution."""
#     domains: dict[str, int] = {}
#     difficulties: list[int] = []

#     for _, data in KNOWLEDGE_GRAPH.nodes(data=True):
#         d = data.get("domain", "general")
#         domains[d] = domains.get(d, 0) + 1
#         difficulties.append(data.get("difficulty", 2))

#     avg_diff = sum(difficulties) / len(difficulties) if difficulties else 0.0

#     return GraphStatsResponse(
#         total_nodes=KNOWLEDGE_GRAPH.number_of_nodes(),
#         total_edges=KNOWLEDGE_GRAPH.number_of_edges(),
#         domains=domains,
#         avg_difficulty=round(avg_diff, 2),
#         skill_list=[
#             {
#                 "id":            s.id,
#                 "name":          s.name,
#                 "domain":        s.domain,
#                 "difficulty":    s.difficulty,
#                 "base_hours":    s.base_hours,
#                 "tags":          s.tags[:5],
#                 "prerequisites": s.prerequisites,
#             }
#             for s in SKILL_NODES
#         ],
#     )


# @app.get("/api/graph/domains", tags=["Knowledge Graph"])
# async def list_domains():
#     """All 11 supported job domains for cross-domain generalisation (O*NET-aligned)."""
#     return {
#         "domains": [
#             {
#                 "id":           p.domain_id,
#                 "name":         p.display_name,
#                 "description":  p.description,
#                 "onet_roles":   p.onet_roles,
#                 "primary_tags": p.primary_skill_tags,
#             }
#             for p in DOMAIN_PROFILES
#             if p.domain_id != "general"
#         ]
#     }


# # ─────────────────────────────────────────────────────────────────────────────
# # Core adaptive engine
# # ─────────────────────────────────────────────────────────────────────────────

# @app.post("/api/analyze", response_model=AnalyzeResponse, tags=["Core Engine"])
# async def analyze(
#     resume:    UploadFile = File(...,  description="Resume (PDF / DOCX / TXT)"),
#     jd:        UploadFile = File(None, description="Job description file (optional)"),
#     jd_text:   str        = Form(None, description="Job description as plain text"),
#     algorithm: str        = Form("auto", description="Path algorithm: auto | dijkstra | astar | dp"),
# ):
#     """
#     **Core endpoint** — Full adaptive engine pipeline for ANY job domain.

#     ### Pipeline
#     1. Domain detection (11 domains via O*NET keyword scoring)
#     2. Claude Sonnet 4 extracts structured skills from resume + JD
#     3. Quantitative proficiency scoring (BKT model):
#        `SkillScore = 0.5 × LLM + 0.5 × (Base + YearsBonus + ComplexityBonus + RecencyMod + LeadershipBonus)`
#     4. Gap analysis with BKT slip-factor adjustment
#     5. Prerequisite chain expansion from the 73-node Knowledge Graph
#     6. Path optimisation (auto-selects Dijkstra / A* / DP)
#     7. Claude reasoning traces per module

#     ### Response includes
#     - `pathway` — ordered modules with `estimated_hours`, `traditional_hours`, `hours_saved`, `confidence`
#     - `summary` — time_saved_pct, algorithm used, coverage %
#     - `domain_info` — detected domain with confidence and O*NET codes
#     - `graph_stats` — knowledge graph metadata
#     """
#     if jd is None and not jd_text:
#         raise HTTPException(400, "Provide either a JD file or jd_text.")
#     if algorithm not in ("auto", "dijkstra", "astar", "dp"):
#         raise HTTPException(400, "algorithm must be one of: auto, dijkstra, astar, dp")

#     try:
#         resume_text = await extract_text(resume)
#         jd_raw      = await extract_text(jd) if jd else jd_text

#         if len(resume_text.strip()) < 50:
#             raise HTTPException(422, "Resume appears empty or unreadable.")
#         if len(jd_raw.strip()) < 30:
#             raise HTTPException(422, "Job description too short.")

#         plan, domain_result, _, _ = await _run_pipeline(
#             resume_text, jd_raw, algorithm=algorithm
#         )
#         return _plan_to_response(plan, domain_result)

#     except HTTPException:
#         raise
#     except Exception as e:
#         logger.exception("Analysis pipeline failed")
#         raise HTTPException(500, f"Analysis failed: {str(e)}")


# # ─────────────────────────────────────────────────────────────────────────────
# # ML/DL Dedicated Pathway
# # ─────────────────────────────────────────────────────────────────────────────

# @app.post("/api/pathway/mldl", tags=["ML/DL Pathway"])
# async def mldl_pathway(
#     resume:      UploadFile = File(...,  description="Resume (PDF / DOCX / TXT)"),
#     jd:          UploadFile = File(None, description="ML/DL job description (optional)"),
#     jd_text:     str        = Form(None, description="JD as plain text"),
#     track:       str        = Form(None, description="Force track: classical | computer-vision | nlp-llm | mlops | rl"),
#     hours_per_week: float   = Form(10.0, description="Available study hours per week"),
# ):
#     """
#     **Dedicated ML/DL Pathway** — Curated curriculum for 5 specialisation tracks.

#     ### Tracks
#     | Track | Description |
#     |-------|-------------|
#     | `classical` | Classical ML → Production (Data Scientist path) |
#     | `computer-vision` | Deep Learning → CV (CV Engineer path) |
#     | `nlp-llm` | Deep Learning → NLP / LLMs / Agents (AI Engineer path) |
#     | `mlops` | MLOps & ML Platform (Production ML path) |
#     | `rl` | Reinforcement Learning (Research Scientist path) |

#     ### Assessment
#     Assesses candidate's current ML level (0–5) and returns:
#     - Entry point in the curriculum (skips known foundations)
#     - Ordered, dependency-safe module sequence
#     - Phase-based timeline (Foundation → Core → Advanced → Production)
#     - Level label: Beginner / Intermediate / Advanced / Expert

#     ### Response
#     - `track` — selected or detected track
#     - `level_assessment` — current ML level (0–5) with strongest/weakest areas
#     - `summary` — hours, time saved, modules
#     - `pathway` — full ordered module list
#     - `timeline` — week-by-week learning phases
#     - `time_saved` — vs traditional curriculum
#     """
#     if jd is None and not jd_text:
#         jd_text = "ML Engineer with expertise in Python, machine learning, deep learning, and model deployment."

#     valid_tracks = (None, "classical", "computer-vision", "nlp-llm", "mlops", "rl")
#     if track not in valid_tracks:
#         raise HTTPException(400, f"track must be one of: {', '.join(t for t in valid_tracks if t)}")

#     try:
#         resume_text = await extract_text(resume)
#         jd_raw      = await extract_text(jd) if jd else jd_text

#         if len(resume_text.strip()) < 30:
#             raise HTTPException(422, "Resume appears empty or unreadable.")

#         # Extract skills first (cached)
#         skill_key = make_skill_key(resume_text, jd_raw)
#         skill_map = skill_cache.get(skill_key)
#         if skill_map is None:
#             skill_map = extract_skills(resume_text, jd_raw)
#             skill_cache.set(skill_key, skill_map)

#         matched = match_skills_to_onet(skill_map)

#         # Build ML/DL pathway
#         result = build_mldl_pathway(
#             resume_skills=matched.get("resume_skills", []),
#             jd_text=jd_raw,
#             jd_requirements=matched.get("jd_requirements", []),
#             force_track=track,
#         )

#         # Enrich traces
#         result.plan = enrich_traces_with_claude(result.plan, {}, use_llm=True)

#         # Build timeline and savings
#         timeline   = build_roadmap_timeline(result.plan, hours_per_week=hours_per_week)
#         time_saved = build_time_saved_summary(result.plan)

#         # Domain result placeholder for ML
#         from .domain_detector import DetectionResult, get_domain_profile
#         domain_result = DetectionResult(
#             domain_id="ml",
#             display_name="Machine Learning / AI",
#             confidence=0.98,
#             matched_keywords=[result.track],
#             profile=get_domain_profile("ml"),
#         )

#         base_response = _plan_to_response(result.plan, domain_result)

#         return {
#             "track":             result.track,
#             "track_description": result.track_description,
#             "all_tracks":        TRACK_DESCRIPTIONS,
#             "level_assessment": {
#                 "level":               result.level_assessment.level,
#                 "label":               result.level_assessment.label,
#                 "foundation_score":    result.level_assessment.foundation_score,
#                 "dl_score":            result.level_assessment.dl_score,
#                 "specialisation_score": result.level_assessment.specialisation_score,
#                 "strongest_area":      result.level_assessment.strongest_area,
#                 "weakest_area":        result.level_assessment.weakest_area,
#                 "entry_sequence":      result.level_assessment.recommended_entry_sequence,
#             },
#             "curriculum_stats": {
#                 "total_curriculum_modules": result.curriculum_modules_total,
#                 "modules_in_track":         result.modules_in_track,
#             },
#             "summary":   base_response.summary,
#             "pathway":   base_response.pathway,
#             "domain_info": base_response.domain_info,
#             "graph_stats": base_response.graph_stats,
#             "timeline":   timeline,
#             "time_saved": time_saved,
#         }

#     except HTTPException:
#         raise
#     except Exception as e:
#         logger.exception("ML/DL pathway failed")
#         raise HTTPException(500, f"ML/DL pathway failed: {str(e)}")


# # ─────────────────────────────────────────────────────────────────────────────
# # Scenario Comparison
# # ─────────────────────────────────────────────────────────────────────────────

# @app.post("/api/compare", response_model=CompareResponse, tags=["Scenario Comparison"])
# async def compare_scenarios(
#     resume_a: UploadFile = File(...,  description="Resume A (e.g. fresher)"),
#     resume_b: UploadFile = File(...,  description="Resume B (e.g. senior)"),
#     jd:       UploadFile = File(None, description="Shared job description (file)"),
#     jd_text:  str        = Form(None, description="Shared job description (text)"),
# ):
#     """
#     **Side-by-side scenario comparison** — the demo feature judges love.

#     Upload two resumes against the same JD to compare:
#     - Full learning pathway for each candidate
#     - Hours required vs saved per person
#     - Modules required / skipped delta
#     - Domain coverage comparison

#     **Classic demo:** Fresher Resume A vs Senior Resume B for the same ML Engineer JD.
#     Shows concretely how the adaptive engine personalises differently.
#     """
#     if jd is None and not jd_text:
#         raise HTTPException(400, "Provide a JD file or jd_text.")

#     try:
#         text_a = await extract_text(resume_a)
#         text_b = await extract_text(resume_b)
#         jd_raw = (await jd.read()).decode("utf-8", errors="ignore") if jd else jd_text

#         plan_a, dr_a, _, _ = await _run_pipeline(text_a, jd_raw, use_llm_traces=False)
#         plan_b, dr_b, _, _ = await _run_pipeline(text_b, jd_raw, use_llm_traces=False)

#         def _metrics(plan: PathPlan) -> CompareMetricsModel:
#             s = plan.summary
#             return CompareMetricsModel(
#                 total_modules=s["total_modules"],
#                 required_count=s["required"],
#                 fast_track_count=s["fast_track"],
#                 skip_count=s["skipped"],
#                 estimated_hours=s["estimated_hours"],
#                 traditional_hours=s["traditional_hours"],
#                 time_saved_pct=s["time_saved_pct"],
#                 overall_confidence=plan.overall_confidence,
#                 domain_breakdown=plan.domain_breakdown,
#             )

#         diff = {
#             "hours_delta":       round(plan_a.total_adaptive_hours - plan_b.total_adaptive_hours, 1),
#             "required_delta":    plan_a.summary["required"] - plan_b.summary["required"],
#             "skip_delta":        plan_b.summary["skipped"] - plan_a.summary["skipped"],
#             "time_saved_pct_a":  plan_a.summary["time_saved_pct"],
#             "time_saved_pct_b":  plan_b.summary["time_saved_pct"],
#             "domain_a":          dr_a.display_name,
#             "domain_b":          dr_b.display_name,
#             "more_experienced":  "B" if plan_b.summary["skipped"] > plan_a.summary["skipped"] else "A",
#             "interpretation": (
#                 f"Candidate B needs {abs(plan_b.summary['required'] - plan_a.summary['required'])} "
#                 f"{'fewer' if plan_b.summary['required'] < plan_a.summary['required'] else 'more'} "
#                 f"required modules than Candidate A."
#             ),
#         }

#         return CompareResponse(
#             scenario_a=_metrics(plan_a),
#             scenario_b=_metrics(plan_b),
#             pathway_a=[_build_step_model(s) for s in plan_a.pathway],
#             pathway_b=[_build_step_model(s) for s in plan_b.pathway],
#             diff=diff,
#         )

#     except HTTPException:
#         raise
#     except Exception as e:
#         logger.exception("Comparison pipeline failed")
#         raise HTTPException(500, f"Comparison failed: {str(e)}")


# # ─────────────────────────────────────────────────────────────────────────────
# # Analytics endpoints
# # ─────────────────────────────────────────────────────────────────────────────

# @app.post("/api/analytics/radar", tags=["Analytics"])
# async def radar_chart(
#     resume:  UploadFile = File(...,  description="Resume"),
#     jd:      UploadFile = File(None, description="Job description (file)"),
#     jd_text: str        = Form(None, description="Job description (text)"),
# ):
#     """
#     **Skill Radar Chart data** — per-axis current vs required proficiency.

#     Axes: Foundations, Classical ML, Deep Learning, Cloud/DevOps,
#     Data Engineering, NLP/LLM, MLOps/Production, Software Engineering,
#     Security, Product/Analytics.

#     Returns numeric arrays ready to drop into Chart.js / Recharts radar.
#     """
#     if jd is None and not jd_text:
#         raise HTTPException(400, "Provide a JD file or jd_text.")
#     try:
#         resume_text = await extract_text(resume)
#         jd_raw      = await extract_text(jd) if jd else jd_text

#         _, _, gap_map, _ = await _run_pipeline(
#             resume_text, jd_raw, use_llm_traces=False
#         )

#         domain_map = {}
#         for _, data in KNOWLEDGE_GRAPH.nodes(data=True):
#             pass  # done inline in analytics
#         for sid in gap_map:
#             node_data = KNOWLEDGE_GRAPH.nodes.get(sid, {})
#             domain_map[sid] = node_data.get("domain", "general")

#         radar = build_radar_data(gap_map, domain_map)
#         return radar

#     except HTTPException:
#         raise
#     except Exception as e:
#         logger.exception("Radar chart failed")
#         raise HTTPException(500, str(e))


# @app.post("/api/analytics/timeline", tags=["Analytics"])
# async def roadmap_timeline(
#     resume:         UploadFile = File(...),
#     jd:             UploadFile = File(None),
#     jd_text:        str        = Form(None),
#     hours_per_week: float      = Form(10.0, description="Study hours per week"),
# ):
#     """
#     **Roadmap Timeline** — phase-based weekly learning plan.

#     Breaks the pathway into 4 phases:
#     1. Foundations & Prerequisites
#     2. Core Skills
#     3. Intermediate / Specialisation
#     4. Advanced & Production

#     Returns week ranges, hours per phase, and module lists.
#     """
#     if jd is None and not jd_text:
#         raise HTTPException(400, "Provide a JD file or jd_text.")
#     try:
#         resume_text = await extract_text(resume)
#         jd_raw      = await extract_text(jd) if jd else jd_text

#         plan, _, _, _ = await _run_pipeline(resume_text, jd_raw, use_llm_traces=False)
#         return build_roadmap_timeline(plan, hours_per_week=hours_per_week)

#     except HTTPException:
#         raise
#     except Exception as e:
#         logger.exception("Timeline failed")
#         raise HTTPException(500, str(e))


# @app.post("/api/analytics/savings", tags=["Analytics"])
# async def time_savings(
#     resume:  UploadFile = File(...),
#     jd:      UploadFile = File(None),
#     jd_text: str        = Form(None),
# ):
#     """
#     **Time Saved Summary** — adaptive vs traditional onboarding comparison.

#     Returns:
#     - `traditional_hours` — what a standard curriculum would take
#     - `adaptive_hours`    — personalised time after SKIP + FAST_TRACK
#     - `hours_saved`       — absolute saving
#     - `time_saved_pct`    — percentage faster
#     - `breakdown`         — per-module savings table
#     - `label`             — human-readable summary e.g. "43% faster"
#     """
#     if jd is None and not jd_text:
#         raise HTTPException(400, "Provide a JD file or jd_text.")
#     try:
#         resume_text = await extract_text(resume)
#         jd_raw      = await extract_text(jd) if jd else jd_text

#         plan, _, _, _ = await _run_pipeline(resume_text, jd_raw, use_llm_traces=False)
#         return build_time_saved_summary(plan)

#     except HTTPException:
#         raise
#     except Exception as e:
#         logger.exception("Savings endpoint failed")
#         raise HTTPException(500, str(e))


# # ─────────────────────────────────────────────────────────────────────────────
# # Transparency & Validation endpoints
# # ─────────────────────────────────────────────────────────────────────────────

# @app.get("/api/transparency/disclosure", tags=["Transparency"])
# async def transparency_disclosure():
#     """
#     **Full Transparency Disclosure** — datasets, models, citations, originality statement.

#     Returns every dataset, pre-trained model, and open-source library used,
#     with exact citations, how each is used, what NeuralPath built originally
#     on top of it, and validation metrics measured against each dataset.

#     Per hackathon rules: all datasets and models explicitly cited here.
#     """
#     from .dataset_registry import get_full_disclosure
#     return get_full_disclosure()


# @app.get("/api/transparency/metrics", tags=["Transparency"])
# async def validation_metrics():
#     """
#     **Internal Validation Metrics** — all 8 metrics that validate the engine's efficiency.

#     Runs the full validation suite and returns:

#     | Metric | Dataset | Method |
#     |--------|---------|--------|
#     | Skill Resolver Accuracy | Resume Dataset (Kaggle) | Exact match on 20 fixtures |
#     | Domain Detection F1 | Jobs Dataset (Kaggle) | Macro-F1 across 11 domains |
#     | Gap Classification Precision | Synthetic ground truth | BKT action match |
#     | Proficiency Scorer Monotonicity | Synthetic fixtures | Bracket ordering |
#     | Prerequisite Coverage | Knowledge Graph | No dangling references |
#     | Time Savings Efficiency | Synthetic profiles | Adaptive vs traditional |
#     | Pathway Topological Validity | Synthetic chains | Prereq ordering |
#     | BKT Slip-Factor Sensitivity | Synthetic pairs | adj_gap > raw_gap |
#     """
#     from .validation import run_all_validations
#     return run_all_validations()


# @app.get("/api/transparency/algorithms", tags=["Transparency"])
# async def algorithm_deep_dive():
#     """
#     **Algorithm Deep Dive** — mathematical specification of every original algorithm.

#     Documents:
#     - BKT Proficiency Scoring formula with all component weights
#     - Gap Classification thresholds and slip-factor derivation
#     - Path optimisation: Dijkstra, A* heuristic, DP priority function
#     - Prerequisite expansion algorithm
#     - Domain detection scoring function
#     - ML level assessment formula
#     """
#     return _ALGORITHM_DOCS


# # Algorithm documentation — machine-readable deep-dive
# _ALGORITHM_DOCS = {
#     "title": "NeuralPath — Adaptive Logic Algorithm Specification",
#     "version": "3.0.0",
#     "originality_note": (
#         "All algorithms below are original implementations by the NeuralPath team. "
#         "Pre-trained models (Claude Sonnet 4, NetworkX) are used only as tools "
#         "— the decision logic is entirely our code."
#     ),

#     "algorithms": [

#         {
#             "name": "BKT Proficiency Scoring Model",
#             "file": "backend/proficiency_scorer.py",
#             "purpose": "Convert raw resume signals into a calibrated proficiency score [0.05, 0.98]",
#             "formula": {
#                 "equation": "SkillScore = 0.5 × LLM_score + 0.5 × SignalScore",
#                 "signal_score": "Base + YearsBonus + ComplexityBonus + RecencyModifier + LeadershipBonus + EducationBonus + PrimaryBonus",
#                 "components": {
#                     "Base": {
#                         "mentioned":  0.15,
#                         "used":       0.30,
#                         "proficient": 0.55,
#                         "expert":     0.80,
#                         "derived_from": "LLM score bracket (≥0.75=expert, ≥0.50=proficient, ≥0.25=used, else=mentioned)",
#                     },
#                     "YearsBonus": {
#                         "formula":    "min(0.20, 0.08 × log(1 + years))",
#                         "rationale":  "Logarithmic — diminishing returns after ~5 years",
#                         "max_value":  0.20,
#                     },
#                     "ComplexityBonus": {
#                         "academic":    0.00,
#                         "personal":    0.03,
#                         "internship":  0.05,
#                         "production":  0.12,
#                         "scale":       0.16,
#                     },
#                     "RecencyModifier": {
#                         "≤0.5yr":   +0.05,
#                         "≤1yr":     +0.02,
#                         "≤2yr":      0.00,
#                         "≤4yr":     -0.05,
#                         "≤6yr":     -0.10,
#                         ">6yr":     -0.15,
#                     },
#                     "LeadershipBonus": {"none": 0.00, "team": 0.05, "org": 0.08},
#                     "EducationBonus":  {"none": 0.00, "course": 0.03, "degree": 0.08, "publication": 0.12},
#                     "PrimaryBonus":    {"is_primary_skill": 0.05, "default": 0.00},
#                 },
#                 "clamp": "max(0.05, min(0.98, computed))",
#                 "confidence": "min(0.95, 0.55 + n_signals × 0.08) where n_signals = count of non-default signals",
#             },
#             "inspiration": "Bayesian Knowledge Tracing (Corbett & Anderson, 1994) — weights are original",
#         },

#         {
#             "name": "BKT Gap Classification",
#             "file": "backend/gap_analyzer.py",
#             "purpose": "Classify each skill as SKIP / FAST_TRACK / REQUIRED",
#             "formula": {
#                 "raw_gap":      "max(0.0, required_proficiency − current_proficiency)",
#                 "adjusted_gap": "raw_gap × (2.0 − BKT_SLIP_FACTOR)  if  0 < current < required  else  raw_gap",
#                 "slip_factor":  "BKT_SLIP_FACTOR = 0.85 (tunable via env var)",
#                 "rationale":    "Partial knowledge is often overconfident — widening the gap by ×1.15 accounts for this bias",
#                 "classification": {
#                     "SKIP":       "adjusted_gap ≤ 0.10",
#                     "FAST_TRACK": "0.10 < adjusted_gap ≤ 0.30",
#                     "REQUIRED":   "adjusted_gap > 0.30",
#                 },
#                 "thresholds_tunable": True,
#                 "env_vars": {
#                     "GAP_SKIP_THRESHOLD": 0.10,
#                     "GAP_FAST_THRESHOLD": 0.30,
#                     "BKT_SLIP_FACTOR":    0.85,
#                 },
#             },
#             "inspiration": "BKT slip parameter concept; threshold values are original",
#         },

#         {
#             "name": "Adaptive Path Optimizer — Algorithm Auto-Selection",
#             "file": "backend/optimizer.py",
#             "purpose": "Select the best path algorithm based on problem size",
#             "selection_logic": {
#                 "≤3 required modules":  "Dijkstra  — minimum total hours",
#                 "4–8 required modules": "A*        — prioritise critical JD skills",
#                 "9+ required modules":  "DP        — full coverage guaranteed",
#             },
#         },

#         {
#             "name": "Dijkstra Path Algorithm",
#             "file": "backend/optimizer.py :: _dijkstra_path()",
#             "purpose": "Find the minimum-total-hours learning path",
#             "implementation": {
#                 "library":    "NetworkX nx.dijkstra_path()",
#                 "graph":      "Dynamically built adaptive DAG — original",
#                 "edge_weight": "estimated_hours × DIFFICULTY_MULTIPLIER[difficulty]",
#                 "virtual_nodes": "START_NODE and END_NODE added to support single-source shortest path",
#                 "original_work": "Graph construction, edge weighting, virtual node wiring",
#             },
#         },

#         {
#             "name": "A* Path Algorithm with Custom Heuristic",
#             "file": "backend/optimizer.py :: _astar_path()",
#             "purpose": "Prioritise critical JD skills — find high-importance path faster",
#             "implementation": {
#                 "library":   "NetworkX nx.astar_path()",
#                 "heuristic": {
#                     "function":   "h(node) = −gap_score × 5  if  action==REQUIRED and importance==critical  else  0.0",
#                     "rationale":  "Negative heuristic boosts priority for high-gap critical nodes — A* explores them first",
#                     "originality": "Heuristic function is 100% original — NetworkX provides only the A* skeleton",
#                 },
#             },
#         },

#         {
#             "name": "DP Full-Coverage Algorithm",
#             "file": "backend/optimizer.py :: _dp_full_coverage()",
#             "purpose": "Guarantee 100% required module coverage with dependency ordering",
#             "implementation": {
#                 "base":        "NetworkX nx.topological_sort() for dependency ordering",
#                 "priority_fn": {
#                     "function": "priority(node) = (action_tier, −gap_score)",
#                     "tiers":    {"REQUIRED + critical": 0, "REQUIRED": 1, "FAST_TRACK": 2},
#                     "rationale": "Stable sort within topological layers: critical gaps first, then by severity",
#                     "originality": "Dual-key priority function is original — guarantees both order and coverage",
#                 },
#                 "guarantee": "All REQUIRED and FAST_TRACK nodes included, no module before its prerequisites",
#             },
#         },

#         {
#             "name": "Prerequisite Chain Expansion",
#             "file": "backend/optimizer.py :: _expand_with_prerequisites()",
#             "purpose": "Automatically inject missing prerequisite modules into the gap_map",
#             "algorithm": {
#                 "steps": [
#                     "1. For each REQUIRED/FAST_TRACK skill, call get_prerequisite_chain(skill_id)",
#                     "2. get_prerequisite_chain() returns all ancestors in topological order via nx.ancestors()",
#                     "3. For each ancestor not already in gap_map:",
#                     "   a. Check scored_skills for current proficiency",
#                     "   b. Set required_level = max(0.4, difficulty × 0.15)",
#                     "   c. Classify as SKIP/FAST_TRACK/REQUIRED via BKT",
#                     "   d. Insert into expanded_gap_map",
#                     "4. Ensures the plan is always learnable — no orphaned advanced modules",
#                 ],
#                 "originality": "Entire algorithm is original — no library provides this",
#             },
#         },

#         {
#             "name": "Domain Detection",
#             "file": "backend/domain_detector.py :: detect_domain()",
#             "purpose": "Classify JD into 1 of 11 supported domains",
#             "algorithm": {
#                 "for_each_domain": "score += 1 per body keyword match; score += 3 per title keyword match",
#                 "selection":       "domain_id = argmax(scores)",
#                 "confidence":      "min(0.97, 0.50 + (best_score / total_hits) × 0.50)",
#                 "fallback":        "Return 'general' if all scores are 0",
#                 "originality":     "Keyword lists, scoring weights, and confidence formula are original",
#             },
#         },

#         {
#             "name": "ML Level Assessment",
#             "file": "backend/ml_pathway.py :: assess_ml_level()",
#             "purpose": "Score candidate's overall ML/DL level (0–5) from scored skills",
#             "formula": {
#                 "foundation_score":      "avg(math, statistics, python, numpy, preprocessing)",
#                 "classical_score":       "avg(classical_ml, model_evaluation, gradient_boosting)",
#                 "dl_score":              "avg(deep_learning_fundamentals, pytorch, tensorflow)",
#                 "specialisation_score":  "avg(cnn, transformers, nlp_transformers, rl, mlops, serving)",
#                 "overall_avg":           "(foundation + classical + dl + specialisation) / 4",
#                 "levels": {
#                     "≥0.80": {"level": 5, "label": "Expert",                "entry_seq": 40},
#                     "≥0.65": {"level": 4, "label": "Advanced",              "entry_seq": 30},
#                     "≥0.45": {"level": 3, "label": "Intermediate",          "entry_seq": 20},
#                     "≥0.25": {"level": 2, "label": "Beginner–Intermediate", "entry_seq": 10},
#                     "≥0.10": {"level": 1, "label": "Beginner",              "entry_seq": 10},
#                     "<0.10": {"level": 0, "label": "No ML Background",      "entry_seq": 10},
#                 },
#                 "originality": "Assessment formula, tier thresholds, and entry_seq mapping are original",
#             },
#         },
#     ],

#     "knowledge_graph_schema": {
#         "description": "73-node directed acyclic graph (DAG) of skills with prerequisite edges",
#         "node_attributes": {
#             "id":           "Canonical slug identifier (e.g., 'pytorch')",
#             "name":         "Display name",
#             "description":  "Plain English description",
#             "domain":       "software | ml | cloud | data-eng | security | product | general",
#             "difficulty":   "Integer 1–5 (beginner → expert)",
#             "base_hours":   "Estimated learning time from zero knowledge",
#             "tags":         "Aliases for fuzzy matching",
#             "onet_codes":   "O*NET SOC codes (O*NET 28.3)",
#         },
#         "edge_attributes": {
#             "weight":       "prerequisite node's difficulty score",
#             "relationship": "prerequisite (always)",
#         },
#         "originality": (
#             "Graph schema, all 73 nodes, all 65 edges, difficulty scores, "
#             "base_hours estimates — all original. O*NET used only as a reference taxonomy."
#         ),
#     },
# }


# # ─────────────────────────────────────────────────────────────────────────────
# # Serve React frontend (production build)
# # ─────────────────────────────────────────────────────────────────────────────
# if os.path.exists("frontend/dist"):
#     app.mount("/", StaticFiles(directory="frontend/dist", html=True), name="frontend")
#     logger.info("Serving React frontend from frontend/dist")


"""
NeuralPath — AI-Adaptive Onboarding Engine  v2.0
FastAPI Backend — Research-Grade Adaptive Engine

Endpoints:
  GET  /api/health              → system health + graph stats
  GET  /api/graph/stats         → full knowledge graph metadata
  GET  /api/graph/domains       → all supported domains (O*NET)
  GET  /api/cache/stats         → cache hit/miss metrics
  POST /api/analyze             → core adaptive pathway (any domain)
  POST /api/pathway/mldl        → dedicated ML/DL curriculum pathway
  POST /api/compare             → side-by-side scenario comparison
  GET  /api/analytics/radar     → skill radar chart data
  GET  /api/analytics/timeline  → roadmap timeline data
  GET  /api/analytics/savings   → time saved summary card
"""

import os
import logging
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Query
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from .parser             import extract_text
from .skill_extractor    import extract_skills
from .embedder           import match_skills_to_onet
from .gap_analyzer       import (
    compute_gap_map, GapResult,
    GAP_SKIP_THRESHOLD, GAP_FAST_THRESHOLD, BKT_SLIP_FACTOR
)
from .proficiency_scorer import score_resume_skills
from .optimizer          import optimize_learning_path, PathPlan
from .reasoning          import enrich_traces_with_claude
from .domain_detector    import detect_domain, DOMAIN_PROFILES
from .knowledge_graph    import KNOWLEDGE_GRAPH, SKILL_NODES, resolve_skill_id
from .ml_pathway         import build_mldl_pathway, TRACK_DESCRIPTIONS
from .analytics          import build_radar_data, build_roadmap_timeline, build_time_saved_summary
from .cache              import skill_cache, plan_cache, make_skill_key, make_plan_key
from .models             import (
    AnalyzeResponse, CompareResponse, HealthResponse,
    GraphStatsResponse, SummaryModel, DomainInfoModel,
    PathStepModel, ScoreBreakdownModel, CompareMetricsModel,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s"
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# App
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="NeuralPath API",
    description=(
        "**Research-grade AI-Adaptive Onboarding Engine**\n\n"
        "- **73-node Skill Knowledge Graph** with prerequisite dependency chains\n"
        "- **BKT Proficiency Scoring** — `years × complexity × recency × leadership`\n"
        "- **3 Path Algorithms** — Dijkstra / A* / DP (auto-selected)\n"
        "- **11 Domains** — ML, SWE, DevOps, Data Eng, Security, Product, HR, Finance …\n"
        "- **Dedicated ML/DL Pathway** — 5 specialisation tracks\n"
        "- **Scenario Comparison** — Fresher vs Senior side-by-side\n"
    ),
    version="2.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline helpers
# ─────────────────────────────────────────────────────────────────────────────

async def _run_pipeline(
    resume_text: str,
    jd_text: str,
    algorithm: str = "auto",
    use_llm_traces: bool = True,
    use_cache: bool = True,
) -> tuple[PathPlan, object, dict, dict]:
    """
    Full adaptive engine pipeline with caching.

    Returns: (plan, domain_result, gap_map, scored_skills)
    """
    # ── Cache check ─────────────────────────────────────────────
    plan_key = make_plan_key(resume_text, jd_text, algorithm)
    if use_cache:
        cached = plan_cache.get(plan_key)
        if cached:
            logger.info("Cache HIT — returning cached plan")
            return cached

    # ── 1. Domain detection ─────────────────────────────────────
    domain_result = detect_domain(jd_text)
    logger.info(f"Domain: {domain_result.display_name} ({domain_result.confidence:.0%})")

    # ── 2. Skill extraction (with cache) ────────────────────────
    skill_key = make_skill_key(resume_text, jd_text)
    skill_map = skill_cache.get(skill_key)

    if skill_map is None:
        skill_map = extract_skills(resume_text, jd_text)
        skill_cache.set(skill_key, skill_map)
        logger.info("Skills extracted via Claude (cache MISS)")
    else:
        logger.info("Skills from cache (cache HIT)")

    if not skill_map.get("jd_requirements"):
        raise HTTPException(422, "Could not extract skills from the JD. Try a more detailed description.")

    logger.info(
        f"Skills: {len(skill_map.get('resume_skills', []))} resume, "
        f"{len(skill_map.get('jd_requirements', []))} JD"
    )

    # ── 3. O*NET matching ───────────────────────────────────────
    matched = match_skills_to_onet(skill_map)

    # ── 4a. Quantitative proficiency scoring ────────────────────
    scored_skills = score_resume_skills(
        matched.get("resume_skills", []),
        skill_resolver=resolve_skill_id,
    )

    # ── 4b. BKT gap analysis ────────────────────────────────────
    gap_map = compute_gap_map(matched)

    if not gap_map:
        raise HTTPException(422, "Could not compute skill gaps. Check documents and retry.")

    # Override gap_map with higher-quality computed scores
    for skill_id, scored in scored_skills.items():
        if skill_id in gap_map:
            old      = gap_map[skill_id]
            current  = scored.computed_score
            required = old.proficiency_required
            raw_gap  = max(0.0, required - current)
            adj_gap  = raw_gap * (2.0 - BKT_SLIP_FACTOR) if 0 < current < required else raw_gap

            if adj_gap <= GAP_SKIP_THRESHOLD:    action = "SKIP"
            elif adj_gap <= GAP_FAST_THRESHOLD:  action = "FAST_TRACK"
            else:                                action = "REQUIRED"

            gap_map[skill_id] = GapResult(
                skill_id=old.skill_id,
                skill_name=old.skill_name,
                proficiency_current=round(current, 3),
                proficiency_required=old.proficiency_required,
                raw_gap=round(raw_gap, 3),
                adjusted_gap=round(adj_gap, 3),
                action=action,
                importance=old.importance,
            )

    # ── 5 & 6. Graph expansion + path optimisation ──────────────
    plan = optimize_learning_path(gap_map, scored_skills, algorithm=algorithm)

    # ── 7. Reasoning traces ─────────────────────────────────────
    plan = enrich_traces_with_claude(plan, gap_map, use_llm=use_llm_traces)

    result = (plan, domain_result, gap_map, scored_skills, skill_map)

    if use_cache:
        plan_cache.set(plan_key, result)

    return result


def _build_step_model(step) -> PathStepModel:
    bd = step.score_breakdown or {}
    return PathStepModel(
        module_id=step.module_id,
        module_name=step.module_name,
        action=step.action,
        domain=step.domain,
        difficulty=step.difficulty,
        gap_score=step.gap_score,
        proficiency_current=step.proficiency_current,
        proficiency_required=step.proficiency_required,
        estimated_hours=step.estimated_hours,
        traditional_hours=step.traditional_hours,
        hours_saved=step.hours_saved,
        confidence=step.confidence,
        prerequisites=step.prerequisites,
        reason=step.reason,
        score_breakdown=ScoreBreakdownModel(**bd) if bd else ScoreBreakdownModel(),
    )


def _plan_to_response(plan: PathPlan, domain_result, skill_map: dict | None = None) -> AnalyzeResponse:
    s = plan.summary
    # Surface raw NLP-extracted skills for the evaluation script
    extracted_skills = [
        {"skill": sk.get("skill", sk.get("skill_name", "")),
         "proficiency": sk.get("proficiency", 0.3)}
        for sk in (skill_map or {}).get("resume_skills", [])
        if sk.get("skill") or sk.get("skill_name")
    ]
    return AnalyzeResponse(
        summary=SummaryModel(
            total_modules=s["total_modules"],
            required=s["required"],
            fast_track=s["fast_track"],
            skipped=s["skipped"],
            estimated_hours=s["estimated_hours"],
            traditional_hours=s["traditional_hours"],
            hours_saved=s["hours_saved"],
            time_saved_pct=s["time_saved_pct"],
            algorithm=s["algorithm"],
            coverage=s["coverage"],
        ),
        pathway=[_build_step_model(step) for step in plan.pathway],
        domain_info=DomainInfoModel(
            domain_id=domain_result.domain_id,
            display_name=domain_result.display_name,
            confidence=domain_result.confidence,
            matched_keywords=domain_result.matched_keywords,
            onet_roles=domain_result.profile.onet_roles,
        ),
        graph_stats={
            "total_nodes":             KNOWLEDGE_GRAPH.number_of_nodes(),
            "total_edges":             KNOWLEDGE_GRAPH.number_of_edges(),
            "algorithm_used":          plan.algorithm_used,
            "competency_coverage_pct": round(plan.competency_coverage * 100, 1),
            "overall_confidence":      plan.overall_confidence,
        },
        extracted_skills=extracted_skills,
        extraction_meta=(skill_map or {}).get("extraction_meta", {}),
    )


# ─────────────────────────────────────────────────────────────────────────────
# System endpoints
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/api/health", response_model=HealthResponse, tags=["System"])
async def health():
    """Health check — model info and knowledge graph statistics."""
    return HealthResponse(
        status="ok",
        model="llama-3.3-70b-versatile",
        version="2.0.0",
        graph_nodes=KNOWLEDGE_GRAPH.number_of_nodes(),
        graph_edges=KNOWLEDGE_GRAPH.number_of_edges(),
    )


@app.get("/api/nlp/status", tags=["NLP Pipeline"])
async def nlp_pipeline_status():
    """
    **NLP Pipeline Status** — shows which models are loaded.

    Reports:
    - spaCy model loaded + lexicon size
    - BERT NER model loaded
    - Groq API key present
    """
    from .skill_extractor import _models, _SKILL_LEXICON, BERT_MODEL_NAME, GROQ_MODEL
    return {
        "pipeline_layers": [
            {
                "layer": 1,
                "name": "spaCy NER + PhraseMatcher",
                "model": "en_core_web_sm",
                "lexicon_terms": len(_SKILL_LEXICON),
                "loaded": _models.spacy_nlp is not None,
            },
            {
                "layer": 2,
                "name": "BERT NER",
                "model": BERT_MODEL_NAME,
                "loaded": bool(_models.bert_pipe),
            },
            {
                "layer": 3,
                "name": "Merge + deduplicate",
                "model": "rule-based",
                "loaded": True,
            },
            {
                "layer": 4,
                "name": "Groq scoring",
                "model": GROQ_MODEL,
                "loaded": bool(os.environ.get("GROQ_API_KEY", "")),
            },
        ],
        "note": "Models for layers 1+2 are lazy-loaded on first request (pre-downloaded in Docker build).",
    }


@app.get("/api/cache/stats", tags=["System"])
async def cache_stats():
    """Cache hit / miss metrics."""
    return {
        "skill_cache": skill_cache.stats(),
        "plan_cache":  plan_cache.stats(),
    }


@app.delete("/api/cache", tags=["System"])
async def clear_cache():
    """Clear all caches (admin use)."""
    s = skill_cache.clear()
    p = plan_cache.clear()
    return {"cleared": {"skill_cache": s, "plan_cache": p}}


# ─────────────────────────────────────────────────────────────────────────────
# Knowledge Graph endpoints
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/api/graph/stats", response_model=GraphStatsResponse, tags=["Knowledge Graph"])
async def graph_stats():
    """Full knowledge graph metadata — all nodes, edges, domains, difficulty distribution."""
    domains: dict[str, int] = {}
    difficulties: list[int] = []

    for _, data in KNOWLEDGE_GRAPH.nodes(data=True):
        d = data.get("domain", "general")
        domains[d] = domains.get(d, 0) + 1
        difficulties.append(data.get("difficulty", 2))

    avg_diff = sum(difficulties) / len(difficulties) if difficulties else 0.0

    return GraphStatsResponse(
        total_nodes=KNOWLEDGE_GRAPH.number_of_nodes(),
        total_edges=KNOWLEDGE_GRAPH.number_of_edges(),
        domains=domains,
        avg_difficulty=round(avg_diff, 2),
        skill_list=[
            {
                "id":            s.id,
                "name":          s.name,
                "domain":        s.domain,
                "difficulty":    s.difficulty,
                "base_hours":    s.base_hours,
                "tags":          s.tags[:5],
                "prerequisites": s.prerequisites,
            }
            for s in SKILL_NODES
        ],
    )


@app.get("/api/graph/domains", tags=["Knowledge Graph"])
async def list_domains():
    """All 11 supported job domains for cross-domain generalisation (O*NET-aligned)."""
    return {
        "domains": [
            {
                "id":           p.domain_id,
                "name":         p.display_name,
                "description":  p.description,
                "onet_roles":   p.onet_roles,
                "primary_tags": p.primary_skill_tags,
            }
            for p in DOMAIN_PROFILES
            if p.domain_id != "general"
        ]
    }


# ─────────────────────────────────────────────────────────────────────────────
# Core adaptive engine
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/api/analyze", response_model=AnalyzeResponse, tags=["Core Engine"])
async def analyze(
    resume:    UploadFile = File(...,  description="Resume (PDF / DOCX / TXT)"),
    jd:        UploadFile = File(None, description="Job description file (optional)"),
    jd_text:   str        = Form(None, description="Job description as plain text"),
    algorithm: str        = Form("auto", description="Path algorithm: auto | dijkstra | astar | dp"),
):
    """
    **Core endpoint** — Full adaptive engine pipeline for ANY job domain.

    ### Pipeline
    1. Domain detection (11 domains via O*NET keyword scoring)
    2. Claude Sonnet 4 extracts structured skills from resume + JD
    3. Quantitative proficiency scoring (BKT model):
       `SkillScore = 0.5 × LLM + 0.5 × (Base + YearsBonus + ComplexityBonus + RecencyMod + LeadershipBonus)`
    4. Gap analysis with BKT slip-factor adjustment
    5. Prerequisite chain expansion from the 73-node Knowledge Graph
    6. Path optimisation (auto-selects Dijkstra / A* / DP)
    7. Claude reasoning traces per module

    ### Response includes
    - `pathway` — ordered modules with `estimated_hours`, `traditional_hours`, `hours_saved`, `confidence`
    - `summary` — time_saved_pct, algorithm used, coverage %
    - `domain_info` — detected domain with confidence and O*NET codes
    - `graph_stats` — knowledge graph metadata
    """
    if jd is None and not jd_text:
        raise HTTPException(400, "Provide either a JD file or jd_text.")
    if algorithm not in ("auto", "dijkstra", "astar", "dp"):
        raise HTTPException(400, "algorithm must be one of: auto, dijkstra, astar, dp")

    try:
        resume_text = await extract_text(resume)
        jd_raw      = await extract_text(jd) if jd else jd_text

        if len(resume_text.strip()) < 50:
            raise HTTPException(422, "Resume appears empty or unreadable.")
        if len(jd_raw.strip()) < 30:
            raise HTTPException(422, "Job description too short.")

        plan, domain_result, _, _, skill_map = await _run_pipeline(
            resume_text, jd_raw, algorithm=algorithm
        )
        return _plan_to_response(plan, domain_result, skill_map)

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Analysis pipeline failed")
        raise HTTPException(500, f"Analysis failed: {str(e)}")


# ─────────────────────────────────────────────────────────────────────────────
# ML/DL Dedicated Pathway
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/api/pathway/mldl", tags=["ML/DL Pathway"])
async def mldl_pathway(
    resume:      UploadFile = File(...,  description="Resume (PDF / DOCX / TXT)"),
    jd:          UploadFile = File(None, description="ML/DL job description (optional)"),
    jd_text:     str        = Form(None, description="JD as plain text"),
    track:       str        = Form(None, description="Force track: classical | computer-vision | nlp-llm | mlops | rl"),
    hours_per_week: float   = Form(10.0, description="Available study hours per week"),
):
    """
    **Dedicated ML/DL Pathway** — Curated curriculum for 5 specialisation tracks.

    ### Tracks
    | Track | Description |
    |-------|-------------|
    | `classical` | Classical ML → Production (Data Scientist path) |
    | `computer-vision` | Deep Learning → CV (CV Engineer path) |
    | `nlp-llm` | Deep Learning → NLP / LLMs / Agents (AI Engineer path) |
    | `mlops` | MLOps & ML Platform (Production ML path) |
    | `rl` | Reinforcement Learning (Research Scientist path) |

    ### Assessment
    Assesses candidate's current ML level (0–5) and returns:
    - Entry point in the curriculum (skips known foundations)
    - Ordered, dependency-safe module sequence
    - Phase-based timeline (Foundation → Core → Advanced → Production)
    - Level label: Beginner / Intermediate / Advanced / Expert

    ### Response
    - `track` — selected or detected track
    - `level_assessment` — current ML level (0–5) with strongest/weakest areas
    - `summary` — hours, time saved, modules
    - `pathway` — full ordered module list
    - `timeline` — week-by-week learning phases
    - `time_saved` — vs traditional curriculum
    """
    if jd is None and not jd_text:
        jd_text = "ML Engineer with expertise in Python, machine learning, deep learning, and model deployment."

    valid_tracks = (None, "classical", "computer-vision", "nlp-llm", "mlops", "rl")
    if track not in valid_tracks:
        raise HTTPException(400, f"track must be one of: {', '.join(t for t in valid_tracks if t)}")

    try:
        resume_text = await extract_text(resume)
        jd_raw      = await extract_text(jd) if jd else jd_text

        if len(resume_text.strip()) < 30:
            raise HTTPException(422, "Resume appears empty or unreadable.")

        # Extract skills first (cached)
        skill_key = make_skill_key(resume_text, jd_raw)
        skill_map = skill_cache.get(skill_key)
        if skill_map is None:
            skill_map = extract_skills(resume_text, jd_raw)
            skill_cache.set(skill_key, skill_map)

        matched = match_skills_to_onet(skill_map)

        # Build ML/DL pathway
        result = build_mldl_pathway(
            resume_skills=matched.get("resume_skills", []),
            jd_text=jd_raw,
            jd_requirements=matched.get("jd_requirements", []),
            force_track=track,
        )

        # Enrich traces
        result.plan = enrich_traces_with_claude(result.plan, {}, use_llm=True)

        # Build timeline and savings
        timeline   = build_roadmap_timeline(result.plan, hours_per_week=hours_per_week)
        time_saved = build_time_saved_summary(result.plan)

        # Domain result placeholder for ML
        from .domain_detector import DetectionResult, get_domain_profile
        domain_result = DetectionResult(
            domain_id="ml",
            display_name="Machine Learning / AI",
            confidence=0.98,
            matched_keywords=[result.track],
            profile=get_domain_profile("ml"),
        )

        base_response = _plan_to_response(result.plan, domain_result)

        return {
            "track":             result.track,
            "track_description": result.track_description,
            "all_tracks":        TRACK_DESCRIPTIONS,
            "level_assessment": {
                "level":               result.level_assessment.level,
                "label":               result.level_assessment.label,
                "foundation_score":    result.level_assessment.foundation_score,
                "dl_score":            result.level_assessment.dl_score,
                "specialisation_score": result.level_assessment.specialisation_score,
                "strongest_area":      result.level_assessment.strongest_area,
                "weakest_area":        result.level_assessment.weakest_area,
                "entry_sequence":      result.level_assessment.recommended_entry_sequence,
            },
            "curriculum_stats": {
                "total_curriculum_modules": result.curriculum_modules_total,
                "modules_in_track":         result.modules_in_track,
            },
            "summary":   base_response.summary,
            "pathway":   base_response.pathway,
            "domain_info": base_response.domain_info,
            "graph_stats": base_response.graph_stats,
            "timeline":   timeline,
            "time_saved": time_saved,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("ML/DL pathway failed")
        raise HTTPException(500, f"ML/DL pathway failed: {str(e)}")


# ─────────────────────────────────────────────────────────────────────────────
# Scenario Comparison
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/api/compare", response_model=CompareResponse, tags=["Scenario Comparison"])
async def compare_scenarios(
    resume_a: UploadFile = File(...,  description="Resume A (e.g. fresher)"),
    resume_b: UploadFile = File(...,  description="Resume B (e.g. senior)"),
    jd:       UploadFile = File(None, description="Shared job description (file)"),
    jd_text:  str        = Form(None, description="Shared job description (text)"),
):
    """
    **Side-by-side scenario comparison** — the demo feature judges love.

    Upload two resumes against the same JD to compare:
    - Full learning pathway for each candidate
    - Hours required vs saved per person
    - Modules required / skipped delta
    - Domain coverage comparison

    **Classic demo:** Fresher Resume A vs Senior Resume B for the same ML Engineer JD.
    Shows concretely how the adaptive engine personalises differently.
    """
    if jd is None and not jd_text:
        raise HTTPException(400, "Provide a JD file or jd_text.")

    try:
        text_a = await extract_text(resume_a)
        text_b = await extract_text(resume_b)
        jd_raw = (await jd.read()).decode("utf-8", errors="ignore") if jd else jd_text

        plan_a, dr_a, _, _, _ = await _run_pipeline(text_a, jd_raw, use_llm_traces=False)
        plan_b, dr_b, _, _, _ = await _run_pipeline(text_b, jd_raw, use_llm_traces=False)

        def _metrics(plan: PathPlan) -> CompareMetricsModel:
            s = plan.summary
            return CompareMetricsModel(
                total_modules=s["total_modules"],
                required_count=s["required"],
                fast_track_count=s["fast_track"],
                skip_count=s["skipped"],
                estimated_hours=s["estimated_hours"],
                traditional_hours=s["traditional_hours"],
                time_saved_pct=s["time_saved_pct"],
                overall_confidence=plan.overall_confidence,
                domain_breakdown=plan.domain_breakdown,
            )

        diff = {
            "hours_delta":       round(plan_a.total_adaptive_hours - plan_b.total_adaptive_hours, 1),
            "required_delta":    plan_a.summary["required"] - plan_b.summary["required"],
            "skip_delta":        plan_b.summary["skipped"] - plan_a.summary["skipped"],
            "time_saved_pct_a":  plan_a.summary["time_saved_pct"],
            "time_saved_pct_b":  plan_b.summary["time_saved_pct"],
            "domain_a":          dr_a.display_name,
            "domain_b":          dr_b.display_name,
            "more_experienced":  "B" if plan_b.summary["skipped"] > plan_a.summary["skipped"] else "A",
            "interpretation": (
                f"Candidate B needs {abs(plan_b.summary['required'] - plan_a.summary['required'])} "
                f"{'fewer' if plan_b.summary['required'] < plan_a.summary['required'] else 'more'} "
                f"required modules than Candidate A."
            ),
        }

        return CompareResponse(
            scenario_a=_metrics(plan_a),
            scenario_b=_metrics(plan_b),
            pathway_a=[_build_step_model(s) for s in plan_a.pathway],
            pathway_b=[_build_step_model(s) for s in plan_b.pathway],
            diff=diff,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Comparison pipeline failed")
        raise HTTPException(500, f"Comparison failed: {str(e)}")


# ─────────────────────────────────────────────────────────────────────────────
# Analytics endpoints
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/api/analytics/radar", tags=["Analytics"])
async def radar_chart(
    resume:  UploadFile = File(...,  description="Resume"),
    jd:      UploadFile = File(None, description="Job description (file)"),
    jd_text: str        = Form(None, description="Job description (text)"),
):
    """
    **Skill Radar Chart data** — per-axis current vs required proficiency.

    Axes: Foundations, Classical ML, Deep Learning, Cloud/DevOps,
    Data Engineering, NLP/LLM, MLOps/Production, Software Engineering,
    Security, Product/Analytics.

    Returns numeric arrays ready to drop into Chart.js / Recharts radar.
    """
    if jd is None and not jd_text:
        raise HTTPException(400, "Provide a JD file or jd_text.")
    try:
        resume_text = await extract_text(resume)
        jd_raw      = await extract_text(jd) if jd else jd_text

        _, _, gap_map, _, _ = await _run_pipeline(
            resume_text, jd_raw, use_llm_traces=False
        )

        domain_map = {}
        for _, data in KNOWLEDGE_GRAPH.nodes(data=True):
            pass  # done inline in analytics
        for sid in gap_map:
            node_data = KNOWLEDGE_GRAPH.nodes.get(sid, {})
            domain_map[sid] = node_data.get("domain", "general")

        radar = build_radar_data(gap_map, domain_map)
        return radar

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Radar chart failed")
        raise HTTPException(500, str(e))


@app.post("/api/analytics/timeline", tags=["Analytics"])
async def roadmap_timeline(
    resume:         UploadFile = File(...),
    jd:             UploadFile = File(None),
    jd_text:        str        = Form(None),
    hours_per_week: float      = Form(10.0, description="Study hours per week"),
):
    """
    **Roadmap Timeline** — phase-based weekly learning plan.

    Breaks the pathway into 4 phases:
    1. Foundations & Prerequisites
    2. Core Skills
    3. Intermediate / Specialisation
    4. Advanced & Production

    Returns week ranges, hours per phase, and module lists.
    """
    if jd is None and not jd_text:
        raise HTTPException(400, "Provide a JD file or jd_text.")
    try:
        resume_text = await extract_text(resume)
        jd_raw      = await extract_text(jd) if jd else jd_text

        plan, _, _, _, _ = await _run_pipeline(resume_text, jd_raw, use_llm_traces=False)
        return build_roadmap_timeline(plan, hours_per_week=hours_per_week)

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Timeline failed")
        raise HTTPException(500, str(e))


@app.post("/api/analytics/savings", tags=["Analytics"])
async def time_savings(
    resume:  UploadFile = File(...),
    jd:      UploadFile = File(None),
    jd_text: str        = Form(None),
):
    """
    **Time Saved Summary** — adaptive vs traditional onboarding comparison.

    Returns:
    - `traditional_hours` — what a standard curriculum would take
    - `adaptive_hours`    — personalised time after SKIP + FAST_TRACK
    - `hours_saved`       — absolute saving
    - `time_saved_pct`    — percentage faster
    - `breakdown`         — per-module savings table
    - `label`             — human-readable summary e.g. "43% faster"
    """
    if jd is None and not jd_text:
        raise HTTPException(400, "Provide a JD file or jd_text.")
    try:
        resume_text = await extract_text(resume)
        jd_raw      = await extract_text(jd) if jd else jd_text

        plan, _, _, _, _ = await _run_pipeline(resume_text, jd_raw, use_llm_traces=False)
        return build_time_saved_summary(plan)

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Savings endpoint failed")
        raise HTTPException(500, str(e))


# ─────────────────────────────────────────────────────────────────────────────
# Transparency & Validation endpoints
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/api/transparency/disclosure", tags=["Transparency"])
async def transparency_disclosure():
    """
    **Full Transparency Disclosure** — datasets, models, citations, originality statement.

    Returns every dataset, pre-trained model, and open-source library used,
    with exact citations, how each is used, what NeuralPath built originally
    on top of it, and validation metrics measured against each dataset.

    Per hackathon rules: all datasets and models explicitly cited here.
    """
    from .dataset_registry import get_full_disclosure
    return get_full_disclosure()


@app.get("/api/transparency/metrics", tags=["Transparency"])
async def validation_metrics():
    """
    **Internal Validation Metrics** — all 8 metrics that validate the engine's efficiency.

    Runs the full validation suite and returns:

    | Metric | Dataset | Method |
    |--------|---------|--------|
    | Skill Resolver Accuracy | Resume Dataset (Kaggle) | Exact match on 20 fixtures |
    | Domain Detection F1 | Jobs Dataset (Kaggle) | Macro-F1 across 11 domains |
    | Gap Classification Precision | Synthetic ground truth | BKT action match |
    | Proficiency Scorer Monotonicity | Synthetic fixtures | Bracket ordering |
    | Prerequisite Coverage | Knowledge Graph | No dangling references |
    | Time Savings Efficiency | Synthetic profiles | Adaptive vs traditional |
    | Pathway Topological Validity | Synthetic chains | Prereq ordering |
    | BKT Slip-Factor Sensitivity | Synthetic pairs | adj_gap > raw_gap |
    """
    from .validation import run_all_validations
    return run_all_validations()


@app.get("/api/transparency/algorithms", tags=["Transparency"])
async def algorithm_deep_dive():
    """
    **Algorithm Deep Dive** — mathematical specification of every original algorithm.

    Documents:
    - BKT Proficiency Scoring formula with all component weights
    - Gap Classification thresholds and slip-factor derivation
    - Path optimisation: Dijkstra, A* heuristic, DP priority function
    - Prerequisite expansion algorithm
    - Domain detection scoring function
    - ML level assessment formula
    """
    return _ALGORITHM_DOCS


# Algorithm documentation — machine-readable deep-dive
_ALGORITHM_DOCS = {
    "title": "NeuralPath — Adaptive Logic Algorithm Specification",
    "version": "2.0.0",
    "originality_note": (
        "All algorithms below are original implementations by the NeuralPath team. "
        "Pre-trained models (Claude Sonnet 4, NetworkX) are used only as tools "
        "— the decision logic is entirely our code."
    ),

    "algorithms": [

        {
            "name": "BKT Proficiency Scoring Model",
            "file": "backend/proficiency_scorer.py",
            "purpose": "Convert raw resume signals into a calibrated proficiency score [0.05, 0.98]",
            "formula": {
                "equation": "SkillScore = 0.5 × LLM_score + 0.5 × SignalScore",
                "signal_score": "Base + YearsBonus + ComplexityBonus + RecencyModifier + LeadershipBonus + EducationBonus + PrimaryBonus",
                "components": {
                    "Base": {
                        "mentioned":  0.15,
                        "used":       0.30,
                        "proficient": 0.55,
                        "expert":     0.80,
                        "derived_from": "LLM score bracket (≥0.75=expert, ≥0.50=proficient, ≥0.25=used, else=mentioned)",
                    },
                    "YearsBonus": {
                        "formula":    "min(0.20, 0.08 × log(1 + years))",
                        "rationale":  "Logarithmic — diminishing returns after ~5 years",
                        "max_value":  0.20,
                    },
                    "ComplexityBonus": {
                        "academic":    0.00,
                        "personal":    0.03,
                        "internship":  0.05,
                        "production":  0.12,
                        "scale":       0.16,
                    },
                    "RecencyModifier": {
                        "≤0.5yr":   +0.05,
                        "≤1yr":     +0.02,
                        "≤2yr":      0.00,
                        "≤4yr":     -0.05,
                        "≤6yr":     -0.10,
                        ">6yr":     -0.15,
                    },
                    "LeadershipBonus": {"none": 0.00, "team": 0.05, "org": 0.08},
                    "EducationBonus":  {"none": 0.00, "course": 0.03, "degree": 0.08, "publication": 0.12},
                    "PrimaryBonus":    {"is_primary_skill": 0.05, "default": 0.00},
                },
                "clamp": "max(0.05, min(0.98, computed))",
                "confidence": "min(0.95, 0.55 + n_signals × 0.08) where n_signals = count of non-default signals",
            },
            "inspiration": "Bayesian Knowledge Tracing (Corbett & Anderson, 1994) — weights are original",
        },

        {
            "name": "BKT Gap Classification",
            "file": "backend/gap_analyzer.py",
            "purpose": "Classify each skill as SKIP / FAST_TRACK / REQUIRED",
            "formula": {
                "raw_gap":      "max(0.0, required_proficiency − current_proficiency)",
                "adjusted_gap": "raw_gap × (2.0 − BKT_SLIP_FACTOR)  if  0 < current < required  else  raw_gap",
                "slip_factor":  "BKT_SLIP_FACTOR = 0.85 (tunable via env var)",
                "rationale":    "Partial knowledge is often overconfident — widening the gap by ×1.15 accounts for this bias",
                "classification": {
                    "SKIP":       "adjusted_gap ≤ 0.10",
                    "FAST_TRACK": "0.10 < adjusted_gap ≤ 0.30",
                    "REQUIRED":   "adjusted_gap > 0.30",
                },
                "thresholds_tunable": True,
                "env_vars": {
                    "GAP_SKIP_THRESHOLD": 0.10,
                    "GAP_FAST_THRESHOLD": 0.30,
                    "BKT_SLIP_FACTOR":    0.85,
                },
            },
            "inspiration": "BKT slip parameter concept; threshold values are original",
        },

        {
            "name": "Adaptive Path Optimizer — Algorithm Auto-Selection",
            "file": "backend/optimizer.py",
            "purpose": "Select the best path algorithm based on problem size",
            "selection_logic": {
                "≤3 required modules":  "Dijkstra  — minimum total hours",
                "4–8 required modules": "A*        — prioritise critical JD skills",
                "9+ required modules":  "DP        — full coverage guaranteed",
            },
        },

        {
            "name": "Dijkstra Path Algorithm",
            "file": "backend/optimizer.py :: _dijkstra_path()",
            "purpose": "Find the minimum-total-hours learning path",
            "implementation": {
                "library":    "NetworkX nx.dijkstra_path()",
                "graph":      "Dynamically built adaptive DAG — original",
                "edge_weight": "estimated_hours × DIFFICULTY_MULTIPLIER[difficulty]",
                "virtual_nodes": "START_NODE and END_NODE added to support single-source shortest path",
                "original_work": "Graph construction, edge weighting, virtual node wiring",
            },
        },

        {
            "name": "A* Path Algorithm with Custom Heuristic",
            "file": "backend/optimizer.py :: _astar_path()",
            "purpose": "Prioritise critical JD skills — find high-importance path faster",
            "implementation": {
                "library":   "NetworkX nx.astar_path()",
                "heuristic": {
                    "function":   "h(node) = −gap_score × 5  if  action==REQUIRED and importance==critical  else  0.0",
                    "rationale":  "Negative heuristic boosts priority for high-gap critical nodes — A* explores them first",
                    "originality": "Heuristic function is 100% original — NetworkX provides only the A* skeleton",
                },
            },
        },

        {
            "name": "DP Full-Coverage Algorithm",
            "file": "backend/optimizer.py :: _dp_full_coverage()",
            "purpose": "Guarantee 100% required module coverage with dependency ordering",
            "implementation": {
                "base":        "NetworkX nx.topological_sort() for dependency ordering",
                "priority_fn": {
                    "function": "priority(node) = (action_tier, −gap_score)",
                    "tiers":    {"REQUIRED + critical": 0, "REQUIRED": 1, "FAST_TRACK": 2},
                    "rationale": "Stable sort within topological layers: critical gaps first, then by severity",
                    "originality": "Dual-key priority function is original — guarantees both order and coverage",
                },
                "guarantee": "All REQUIRED and FAST_TRACK nodes included, no module before its prerequisites",
            },
        },

        {
            "name": "Prerequisite Chain Expansion",
            "file": "backend/optimizer.py :: _expand_with_prerequisites()",
            "purpose": "Automatically inject missing prerequisite modules into the gap_map",
            "algorithm": {
                "steps": [
                    "1. For each REQUIRED/FAST_TRACK skill, call get_prerequisite_chain(skill_id)",
                    "2. get_prerequisite_chain() returns all ancestors in topological order via nx.ancestors()",
                    "3. For each ancestor not already in gap_map:",
                    "   a. Check scored_skills for current proficiency",
                    "   b. Set required_level = max(0.4, difficulty × 0.15)",
                    "   c. Classify as SKIP/FAST_TRACK/REQUIRED via BKT",
                    "   d. Insert into expanded_gap_map",
                    "4. Ensures the plan is always learnable — no orphaned advanced modules",
                ],
                "originality": "Entire algorithm is original — no library provides this",
            },
        },

        {
            "name": "Domain Detection",
            "file": "backend/domain_detector.py :: detect_domain()",
            "purpose": "Classify JD into 1 of 11 supported domains",
            "algorithm": {
                "for_each_domain": "score += 1 per body keyword match; score += 3 per title keyword match",
                "selection":       "domain_id = argmax(scores)",
                "confidence":      "min(0.97, 0.50 + (best_score / total_hits) × 0.50)",
                "fallback":        "Return 'general' if all scores are 0",
                "originality":     "Keyword lists, scoring weights, and confidence formula are original",
            },
        },

        {
            "name": "ML Level Assessment",
            "file": "backend/ml_pathway.py :: assess_ml_level()",
            "purpose": "Score candidate's overall ML/DL level (0–5) from scored skills",
            "formula": {
                "foundation_score":      "avg(math, statistics, python, numpy, preprocessing)",
                "classical_score":       "avg(classical_ml, model_evaluation, gradient_boosting)",
                "dl_score":              "avg(deep_learning_fundamentals, pytorch, tensorflow)",
                "specialisation_score":  "avg(cnn, transformers, nlp_transformers, rl, mlops, serving)",
                "overall_avg":           "(foundation + classical + dl + specialisation) / 4",
                "levels": {
                    "≥0.80": {"level": 5, "label": "Expert",                "entry_seq": 40},
                    "≥0.65": {"level": 4, "label": "Advanced",              "entry_seq": 30},
                    "≥0.45": {"level": 3, "label": "Intermediate",          "entry_seq": 20},
                    "≥0.25": {"level": 2, "label": "Beginner–Intermediate", "entry_seq": 10},
                    "≥0.10": {"level": 1, "label": "Beginner",              "entry_seq": 10},
                    "<0.10": {"level": 0, "label": "No ML Background",      "entry_seq": 10},
                },
                "originality": "Assessment formula, tier thresholds, and entry_seq mapping are original",
            },
        },
    ],

    "knowledge_graph_schema": {
        "description": "73-node directed acyclic graph (DAG) of skills with prerequisite edges",
        "node_attributes": {
            "id":           "Canonical slug identifier (e.g., 'pytorch')",
            "name":         "Display name",
            "description":  "Plain English description",
            "domain":       "software | ml | cloud | data-eng | security | product | general",
            "difficulty":   "Integer 1–5 (beginner → expert)",
            "base_hours":   "Estimated learning time from zero knowledge",
            "tags":         "Aliases for fuzzy matching",
            "onet_codes":   "O*NET SOC codes (O*NET 28.3)",
        },
        "edge_attributes": {
            "weight":       "prerequisite node's difficulty score",
            "relationship": "prerequisite (always)",
        },
        "originality": (
            "Graph schema, all 73 nodes, all 65 edges, difficulty scores, "
            "base_hours estimates — all original. O*NET used only as a reference taxonomy."
        ),
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# Serve React frontend (production build)
# ─────────────────────────────────────────────────────────────────────────────
if os.path.exists("frontend/dist"):
    app.mount("/", StaticFiles(directory="frontend/dist", html=True), name="frontend")
    logger.info("Serving React frontend from frontend/dist")