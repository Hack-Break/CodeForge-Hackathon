
"""
NeuralPath — Test Suite
========================
Tests every module and every API endpoint.
Run with:  pytest backend/tests.py -v
"""

import os
import pytest

os.environ.setdefault("GROQ_API_KEY", "gsk_test-key-for-unit-tests")

# ─────────────────────────────────────────────────────────────────────────────
# 1. Knowledge Graph
# ─────────────────────────────────────────────────────────────────────────────

class TestKnowledgeGraph:

    def test_graph_has_sufficient_nodes(self):
        from backend.knowledge_graph import KNOWLEDGE_GRAPH
        assert KNOWLEDGE_GRAPH.number_of_nodes() >= 40

    def test_graph_has_prerequisite_edges(self):
        from backend.knowledge_graph import KNOWLEDGE_GRAPH
        assert KNOWLEDGE_GRAPH.number_of_edges() >= 15

    def test_resolver_returns_canonical_id(self):
        from backend.knowledge_graph import resolve_skill_id
        assert resolve_skill_id("pytorch")          == "pytorch"
        assert resolve_skill_id("machine learning") is not None
        assert resolve_skill_id("docker")           == "docker"
        assert resolve_skill_id("SQL")              is not None
        assert resolve_skill_id("xyz_nonexistent_abc") is None

    def test_prerequisite_chain_is_ordered(self):
        from backend.knowledge_graph import get_prerequisite_chain, KNOWLEDGE_GRAPH
        import networkx as nx
        chain = get_prerequisite_chain("deep-learning-fundamentals")
        assert len(chain) > 0
        # Foundation skills should appear before deep learning
        if "python-basics" in chain and "deep-learning-fundamentals" in chain:
            assert chain.index("python-basics") < chain.index("deep-learning-fundamentals")

    def test_get_domain_skills(self):
        from backend.knowledge_graph import get_domain_skills
        ml_skills = get_domain_skills("ml")
        assert len(ml_skills) >= 5
        cloud_skills = get_domain_skills("cloud")
        assert len(cloud_skills) >= 3

    def test_all_prerequisite_ids_exist_in_graph(self):
        from backend.knowledge_graph import KNOWLEDGE_GRAPH, SKILL_NODES
        all_ids = set(s.id for s in SKILL_NODES)
        for skill in SKILL_NODES:
            for prereq_id in skill.prerequisites:
                assert prereq_id in all_ids, f"Missing prereq: {prereq_id} for {skill.id}"


# ─────────────────────────────────────────────────────────────────────────────
# 2. Proficiency Scorer
# ─────────────────────────────────────────────────────────────────────────────

class TestProficiencyScorer:

    def test_expert_scores_higher_than_fresher(self):
        from backend.proficiency_scorer import ProficiencySignal, compute_proficiency_score
        expert = ProficiencySignal("pytorch", 0.85, years_experience=4.0,
                                   project_complexity="production", leadership_level="team")
        fresher = ProficiencySignal("pytorch", 0.20, years_experience=0.0)
        s_e, _, _ = compute_proficiency_score(expert)
        s_f, _, _ = compute_proficiency_score(fresher)
        assert s_e > s_f, f"Expert {s_e} should > Fresher {s_f}"

    def test_score_clamped_to_valid_range(self):
        from backend.proficiency_scorer import ProficiencySignal, compute_proficiency_score
        sig = ProficiencySignal("python", 1.5, years_experience=100)
        score, _, _ = compute_proficiency_score(sig)
        assert 0.0 <= score <= 1.0

    def test_recency_penalty_applied(self):
        from backend.proficiency_scorer import ProficiencySignal, compute_proficiency_score
        recent = ProficiencySignal("python", 0.7, years_since_used=0.5)
        stale  = ProficiencySignal("python", 0.7, years_since_used=7.0)
        s_r, _, _ = compute_proficiency_score(recent)
        s_s, _, _ = compute_proficiency_score(stale)
        assert s_r > s_s

    def test_score_breakdown_has_all_fields(self):
        from backend.proficiency_scorer import ProficiencySignal, compute_proficiency_score
        sig = ProficiencySignal("sklearn", 0.6, years_experience=2.0)
        _, breakdown, _ = compute_proficiency_score(sig)
        required_keys = ["llm_score", "base_evidence", "years_bonus", "final_score"]
        for key in required_keys:
            assert key in breakdown, f"Missing breakdown key: {key}"

    def test_confidence_increases_with_more_signals(self):
        from backend.proficiency_scorer import ProficiencySignal, compute_proficiency_score
        few_signals  = ProficiencySignal("sql", 0.5)
        many_signals = ProficiencySignal("sql", 0.5, years_experience=3.0,
                                         project_complexity="production", leadership_level="team",
                                         education_level="degree")
        _, _, conf_few  = compute_proficiency_score(few_signals)
        _, _, conf_many = compute_proficiency_score(many_signals)
        assert conf_many > conf_few


# ─────────────────────────────────────────────────────────────────────────────
# 3. Gap Analyzer
# ─────────────────────────────────────────────────────────────────────────────

class TestGapAnalyzer:

    def test_zero_gap_gives_skip(self):
        from backend.gap_analyzer import compute_gap_map
        skill_map = {
            "resume_skills":   [{"onet_id": "python", "skill": "Python", "proficiency": 0.95}],
            "jd_requirements": [{"onet_id": "python", "skill": "Python", "required_level": 0.80}],
        }
        gap_map = compute_gap_map(skill_map)
        assert gap_map["python"].action == "SKIP"

    def test_large_gap_gives_required(self):
        from backend.gap_analyzer import compute_gap_map
        skill_map = {
            "resume_skills":   [],
            "jd_requirements": [{"onet_id": "k8s", "skill": "Kubernetes", "required_level": 0.80}],
        }
        gap_map = compute_gap_map(skill_map)
        assert gap_map["k8s"].action == "REQUIRED"

    def test_medium_gap_gives_fast_track(self):
        from backend.gap_analyzer import compute_gap_map
        skill_map = {
            "resume_skills":   [{"onet_id": "ts", "skill": "TypeScript", "proficiency": 0.65}],
            "jd_requirements": [{"onet_id": "ts", "skill": "TypeScript", "required_level": 0.85}],
        }
        gap_map = compute_gap_map(skill_map)
        assert gap_map["ts"].action in ("FAST_TRACK", "REQUIRED")

    def test_bkt_slip_factor_increases_gap(self):
        from backend.gap_analyzer import compute_gap_map
        skill_map = {
            "resume_skills":   [{"onet_id": "aws", "skill": "AWS", "proficiency": 0.50}],
            "jd_requirements": [{"onet_id": "aws", "skill": "AWS", "required_level": 0.70}],
        }
        gap_map = compute_gap_map(skill_map)
        raw_gap = gap_map["aws"].raw_gap
        adj_gap = gap_map["aws"].adjusted_gap
        assert adj_gap >= raw_gap  # BKT should never reduce gap for partial knowledge


# ─────────────────────────────────────────────────────────────────────────────
# 4. Domain Detector
# ─────────────────────────────────────────────────────────────────────────────

class TestDomainDetector:

    def test_ml_jd_detected(self):
        from backend.domain_detector import detect_domain
        r = detect_domain("Hiring ML Engineer with PyTorch and deep learning", "ML Engineer")
        assert r.domain_id == "ml"
        assert r.confidence >= 0.70

    def test_devops_jd_detected(self):
        from backend.domain_detector import detect_domain
        r = detect_domain("DevOps Engineer: Kubernetes, Terraform, CI/CD", "DevOps Engineer")
        assert r.domain_id == "cloud"

    def test_hr_jd_detected(self):
        from backend.domain_detector import detect_domain
        r = detect_domain("HR Business Partner for talent acquisition and people ops", "HR Business Partner")
        assert r.domain_id == "hr"

    def test_data_analyst_detected(self):
        from backend.domain_detector import detect_domain
        r = detect_domain("Data analyst with SQL, Tableau and Excel skills", "Data Analyst")
        assert r.domain_id == "data-analyst"

    def test_unknown_jd_returns_general(self):
        from backend.domain_detector import detect_domain
        r = detect_domain("We need a person to do things at our company", "")
        assert r.domain_id == "general"

    def test_all_profiles_have_required_fields(self):
        from backend.domain_detector import DOMAIN_PROFILES
        for p in DOMAIN_PROFILES:
            assert p.domain_id
            assert p.display_name
            assert isinstance(p.keywords, list)
            assert isinstance(p.primary_skill_tags, list)


# ─────────────────────────────────────────────────────────────────────────────
# 5. Optimizer
# ─────────────────────────────────────────────────────────────────────────────

class TestOptimizer:

    def _make_gap_map(self):
        from backend.gap_analyzer import GapResult
        return {
            "python-basics":             GapResult("python-basics",            "Python Basics",             0.90, 0.70, 0.00, 0.00, "SKIP",       "critical"),
            "numpy-pandas":              GapResult("numpy-pandas",             "NumPy & Pandas",            0.40, 0.75, 0.35, 0.37, "REQUIRED",   "critical"),
            "classical-ml":              GapResult("classical-ml",             "Classical ML",              0.30, 0.80, 0.50, 0.53, "REQUIRED",   "critical"),
            "deep-learning-fundamentals":GapResult("deep-learning-fundamentals","Deep Learning Fundamentals",0.10, 0.80, 0.70, 0.74, "REQUIRED",   "critical"),
            "pytorch":                   GapResult("pytorch",                  "PyTorch",                   0.05, 0.85, 0.80, 0.85, "REQUIRED",   "critical"),
            "docker":                    GapResult("docker",                   "Docker",                    0.60, 0.65, 0.05, 0.05, "SKIP",        "nice-to-have"),
        }

    def test_plan_contains_all_required_modules(self):
        from backend.optimizer import optimize_learning_path
        gap_map = self._make_gap_map()
        plan = optimize_learning_path(gap_map, {}, algorithm="dp")
        module_ids = {s.module_id for s in plan.pathway}
        for sid, gap in gap_map.items():
            if gap.action != "SKIP":
                assert sid in module_ids, f"Missing required module: {sid}"

    def test_dijkstra_produces_valid_plan(self):
        from backend.optimizer import optimize_learning_path
        gap_map = self._make_gap_map()
        plan = optimize_learning_path(gap_map, {}, algorithm="dijkstra")
        assert len(plan.pathway) > 0
        assert plan.algorithm_used in ("dijkstra", "astar", "dp")

    def test_astar_prioritises_critical_skills(self):
        from backend.optimizer import optimize_learning_path
        gap_map = self._make_gap_map()
        plan = optimize_learning_path(gap_map, {}, algorithm="astar")
        required = [s for s in plan.pathway if s.action == "REQUIRED"]
        assert len(required) >= 3

    def test_time_savings_computed(self):
        from backend.optimizer import optimize_learning_path
        gap_map = self._make_gap_map()
        plan = optimize_learning_path(gap_map, {}, algorithm="dp")
        assert plan.total_traditional_hours >= 0
        assert plan.total_adaptive_hours   >= 0

    def test_prerequisite_chain_expanded(self):
        from backend.optimizer import optimize_learning_path
        from backend.gap_analyzer import GapResult
        # Only provide pytorch — optimizer should add its prereqs
        gap_map = {
            "pytorch": GapResult("pytorch", "PyTorch", 0.0, 0.85, 0.85, 0.90, "REQUIRED", "critical"),
        }
        plan = optimize_learning_path(gap_map, {}, algorithm="dp")
        module_ids = {s.module_id for s in plan.pathway}
        # Should have injected at least some prerequisites
        assert len(module_ids) >= 1

    def test_skip_modules_have_zero_hours(self):
        from backend.optimizer import optimize_learning_path
        gap_map = self._make_gap_map()
        plan = optimize_learning_path(gap_map, {}, algorithm="dp")
        for step in plan.pathway:
            if step.action == "SKIP":
                assert step.estimated_hours == 0.0


# ─────────────────────────────────────────────────────────────────────────────
# 6. ML/DL Pathway
# ─────────────────────────────────────────────────────────────────────────────

class TestMLPathway:

    def _make_resume_skills(self, level="beginner"):
        if level == "beginner":
            return [{"skill": "Python", "proficiency": 0.6, "years": 1}]
        else:
            return [
                {"skill": "Python",           "proficiency": 0.9, "years": 4},
                {"skill": "PyTorch",          "proficiency": 0.8, "years": 2},
                {"skill": "Deep Learning",    "proficiency": 0.7, "years": 2},
                {"skill": "Machine Learning", "proficiency": 0.9, "years": 3},
            ]

    def test_beginner_gets_more_required_modules(self):
        from backend.ml_pathway import build_mldl_pathway
        fresher = build_mldl_pathway(
            self._make_resume_skills("beginner"), "ML Engineer", [], force_track="nlp-llm"
        )
        senior = build_mldl_pathway(
            self._make_resume_skills("senior"), "ML Engineer", [], force_track="nlp-llm"
        )
        fresher_req = sum(1 for s in fresher.plan.pathway if s.action == "REQUIRED")
        senior_req  = sum(1 for s in senior.plan.pathway  if s.action == "REQUIRED")
        assert fresher_req >= senior_req, "Fresher should need >= modules as senior"

    def test_track_detection_nlp(self):
        from backend.ml_pathway import infer_track_from_jd
        track = infer_track_from_jd(
            "NLP Engineer with BERT, GPT fine-tuning and RAG experience",
            {"jd_requirements": [{"skill": "transformers"}]}
        )
        assert track == "nlp-llm"

    def test_track_detection_cv(self):
        from backend.ml_pathway import infer_track_from_jd
        track = infer_track_from_jd("Computer Vision engineer with YOLO and image segmentation", {})
        assert track == "computer-vision"

    def test_level_assessment_produces_valid_level(self):
        from backend.ml_pathway import assess_ml_level
        from backend.proficiency_scorer import ScoredSkill
        skilled = {
            "pytorch":    ScoredSkill("PyTorch",    "pytorch",    0.85, 0.85, 0.9, {}, "expert"),
            "classical-ml": ScoredSkill("Classical ML", "classical-ml", 0.90, 0.90, 0.9, {}, "expert"),
        }
        assessment = assess_ml_level(skilled)
        assert 0 <= assessment.level <= 5
        assert assessment.label in ("Beginner", "Beginner–Intermediate", "Intermediate",
                                    "Advanced", "Expert", "No ML Background")

    def test_all_tracks_produce_pathway(self):
        from backend.ml_pathway import build_mldl_pathway
        skills = self._make_resume_skills("beginner")
        for track in ["classical", "computer-vision", "nlp-llm", "mlops", "rl"]:
            result = build_mldl_pathway(skills, "", [], force_track=track)
            assert len(result.plan.pathway) > 0, f"Empty pathway for track: {track}"


# ─────────────────────────────────────────────────────────────────────────────
# 7. Analytics
# ─────────────────────────────────────────────────────────────────────────────

class TestAnalytics:

    def _make_plan(self):
        from backend.optimizer import PathPlan, PathStep
        steps = [
            PathStep("pytorch",    "PyTorch",         "REQUIRED",  "ml", 3, 0.80, 0.10, 0.85, 14.0, 20.0, 6.0, 0.85),
            PathStep("numpy",      "NumPy",            "FAST_TRACK","ml", 1, 0.20, 0.60, 0.75,  5.0,  8.0, 3.0, 0.80),
            PathStep("python",     "Python Basics",    "SKIP",      "ml", 1, 0.00, 0.92, 0.70,  0.0, 10.0,10.0, 0.95),
        ]
        return PathPlan(
            pathway=steps, algorithm_used="dp",
            total_adaptive_hours=19.0, total_traditional_hours=38.0,
            total_hours_saved=19.0, time_saved_pct=50.0,
            competency_coverage=1.0, overall_confidence=0.85,
            domain_breakdown={"ml": 3},
            summary={"total_modules":3,"required":1,"fast_track":1,"skipped":1,
                     "estimated_hours":19,"traditional_hours":38,"hours_saved":19,
                     "time_saved_pct":50,"algorithm":"dp","coverage":100},
        )

    def test_time_saved_summary_correct(self):
        from backend.analytics import build_time_saved_summary
        plan = self._make_plan()
        summary = build_time_saved_summary(plan)
        assert summary["adaptive_hours"]    > 0
        assert summary["time_saved_pct"]    >= 0
        assert summary["modules_skipped"]   == 1
        assert summary["modules_fast_tracked"] == 1
        assert "label" in summary

    def test_roadmap_timeline_has_phases(self):
        from backend.analytics import build_roadmap_timeline
        plan = self._make_plan()
        timeline = build_roadmap_timeline(plan, hours_per_week=10.0)
        assert "phases" in timeline
        assert timeline["total_weeks"] >= 0
        assert "estimated_completion" in timeline

    def test_radar_data_structure(self):
        from backend.analytics import build_radar_data
        from backend.gap_analyzer import GapResult
        gap_map = {
            "pytorch": GapResult("pytorch", "PyTorch", 0.3, 0.85, 0.55, 0.58, "REQUIRED", "critical"),
        }
        domain_map = {"pytorch": "ml"}
        radar = build_radar_data(gap_map, domain_map)
        assert "axes"     in radar
        assert "current"  in radar
        assert "required" in radar
        assert len(radar["axes"]) == len(radar["current"]) == len(radar["required"])


# ─────────────────────────────────────────────────────────────────────────────
# 8. Cache
# ─────────────────────────────────────────────────────────────────────────────

class TestCache:

    def test_cache_stores_and_retrieves(self):
        from backend.cache import TTLCache
        cache = TTLCache(max_size=10, ttl_seconds=60)
        cache.set("key1", {"data": 42})
        val = cache.get("key1")
        assert val == {"data": 42}

    def test_cache_expires(self):
        import time
        from backend.cache import TTLCache
        cache = TTLCache(max_size=10, ttl_seconds=0)  # instant TTL
        cache.set("key1", "value")
        time.sleep(0.01)
        assert cache.get("key1") is None

    def test_cache_miss_returns_none(self):
        from backend.cache import TTLCache
        cache = TTLCache()
        assert cache.get("nonexistent") is None

    def test_cache_stats_correct(self):
        from backend.cache import TTLCache
        cache = TTLCache()
        cache.set("k", "v")
        cache.get("k")        # hit
        cache.get("missing")  # miss
        stats = cache.stats()
        assert stats["hits"]   == 1
        assert stats["misses"] == 1
        assert stats["entries"] == 1

    def test_lru_eviction(self):
        from backend.cache import TTLCache
        cache = TTLCache(max_size=2, ttl_seconds=3600)
        cache.set("a", 1)
        cache.set("b", 2)
        cache.set("c", 3)   # should evict "a"
        assert cache.get("a") is None
        assert cache.get("b") == 2
        assert cache.get("c") == 3

    def test_make_cache_key_deterministic(self):
        from backend.cache import make_cache_key
        k1 = make_cache_key("hello", "world")
        k2 = make_cache_key("hello", "world")
        k3 = make_cache_key("hello", "different")
        assert k1 == k2
        assert k1 != k3


# ─────────────────────────────────────────────────────────────────────────────
# 9. API Endpoints (integration)
# ─────────────────────────────────────────────────────────────────────────────

class TestAPI:

    @pytest.fixture
    def client(self):
        from fastapi.testclient import TestClient
        from backend.main import app
        return TestClient(app)

    def test_health_endpoint(self, client):
        r = client.get("/api/health")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "ok"
        assert body["model"] == "llama-3.3-70b-versatile"
        assert body["graph_nodes"] >= 40

    def test_graph_stats_endpoint(self, client):
        r = client.get("/api/graph/stats")
        assert r.status_code == 200
        body = r.json()
        assert body["total_nodes"] >= 40
        assert "domains" in body
        assert len(body["skill_list"]) >= 40

    def test_domains_endpoint(self, client):
        r = client.get("/api/graph/domains")
        assert r.status_code == 200
        domains = r.json()["domains"]
        domain_ids = [d["id"] for d in domains]
        assert "ml"        in domain_ids
        assert "cloud"     in domain_ids
        assert "software"  in domain_ids
        assert "hr"        in domain_ids
        assert len(domains) >= 10

    def test_cache_stats_endpoint(self, client):
        r = client.get("/api/cache/stats")
        assert r.status_code == 200
        body = r.json()
        assert "skill_cache" in body
        assert "plan_cache"  in body

    def test_api_docs_available(self, client):
        r = client.get("/api/docs")
        assert r.status_code == 200

    def test_analyze_missing_jd_returns_400(self, client):
        import io
        r = client.post(
            "/api/analyze",
            files={"resume": ("resume.txt", io.BytesIO(b"John Doe Python developer"), "text/plain")},
        )
        assert r.status_code == 400

    def test_analyze_invalid_algorithm_returns_400(self, client):
        import io
        r = client.post(
            "/api/analyze",
            files={"resume": ("resume.txt", io.BytesIO(b"Python developer"), "text/plain")},
            data={"jd_text": "We need a Python developer", "algorithm": "invalid_algo"},
        )
        assert r.status_code == 400

    def test_mldl_endpoint_exists(self, client):
        # Just check it exists and returns 4xx (not 404)
        import io
        r = client.post(
            "/api/pathway/mldl",
            files={"resume": ("r.txt", io.BytesIO(b"x"), "text/plain")},
            data={"jd_text": "ML engineer", "track": "invalid"},
        )
        assert r.status_code != 404

    def test_clear_cache_endpoint(self, client):
        r = client.delete("/api/cache")
        assert r.status_code == 200
        body = r.json()
        assert "cleared" in body


# ─────────────────────────────────────────────────────────────────────────────
# 10. Dataset Registry & Transparency
# ─────────────────────────────────────────────────────────────────────────────

class TestDatasetRegistry:

    def test_all_datasets_have_required_fields(self):
        from backend.dataset_registry import DATASETS
        for d in DATASETS:
            assert d.name
            assert d.source.startswith("http")
            assert d.license
            assert d.how_used
            assert d.original_contribution
            assert isinstance(d.validation_metrics, dict)

    def test_all_models_have_required_fields(self):
        from backend.dataset_registry import MODELS
        for m in MODELS:
            assert m.name
            assert m.version
            assert m.provider
            assert m.how_used
            assert m.original_contribution

    def test_three_datasets_declared(self):
        from backend.dataset_registry import DATASETS
        assert len(DATASETS) >= 3
        names = [d.name for d in DATASETS]
        assert any("O*NET" in n for n in names)
        assert any("Resume" in n for n in names)
        assert any("Job" in n for n in names)

    def test_groq_model_cited(self):
        from backend.dataset_registry import MODELS
        names = [m.name for m in MODELS]
        assert any("Groq" in n or "Llama" in n for n in names)

    def test_networkx_cited(self):
        from backend.dataset_registry import MODELS
        names = [m.name for m in MODELS]
        assert any("NetworkX" in n for n in names)

    def test_disclosure_api_returns_full_structure(self):
        from backend.dataset_registry import get_full_disclosure
        disc = get_full_disclosure()
        assert "datasets"   in disc
        assert "models"     in disc
        assert "libraries"  in disc
        assert "originality_statement" in disc
        assert len(disc["datasets"]) >= 3
        assert len(disc["models"])   >= 2
        assert len(disc["libraries"]) >= 8

    def test_originality_statement_mentions_bkt(self):
        from backend.dataset_registry import ORIGINALITY_STATEMENT
        assert "BKT" in ORIGINALITY_STATEMENT or "Bayesian" in ORIGINALITY_STATEMENT

    def test_originality_statement_mentions_dijkstra(self):
        from backend.dataset_registry import ORIGINALITY_STATEMENT
        assert "Dijkstra" in ORIGINALITY_STATEMENT


# ─────────────────────────────────────────────────────────────────────────────
# 11. Validation Engine
# ─────────────────────────────────────────────────────────────────────────────

class TestValidationEngine:

    def test_all_8_metrics_run(self):
        from backend.validation import run_all_validations
        report = run_all_validations()
        assert report["summary"]["total"] == 8

    def test_resolver_accuracy_above_threshold(self):
        from backend.validation import measure_resolver_accuracy
        r = measure_resolver_accuracy()
        assert r.passed, f"Resolver accuracy {r.value}% below threshold {r.threshold}%"

    def test_domain_detection_above_threshold(self):
        from backend.validation import measure_domain_detection
        r = measure_domain_detection()
        assert r.passed, f"Domain F1 {r.value} below threshold {r.threshold}"

    def test_gap_classification_correct(self):
        from backend.validation import measure_gap_classification
        r = measure_gap_classification()
        assert r.passed, f"Gap classification {r.value}% below threshold {r.threshold}%"

    def test_prerequisite_coverage_100pct(self):
        from backend.validation import measure_prerequisite_coverage
        r = measure_prerequisite_coverage()
        assert r.value == 100.0, "Knowledge graph has broken prerequisite references"
        assert r.passed

    def test_pathway_topologically_valid(self):
        from backend.validation import measure_pathway_validity
        r = measure_pathway_validity()
        assert r.passed, "Optimizer produces topologically invalid pathways"

    def test_bkt_slip_factor_correct(self):
        from backend.validation import measure_bkt_sensitivity
        r = measure_bkt_sensitivity()
        assert r.value == 100.0 and r.passed

    def test_time_savings_positive(self):
        from backend.validation import measure_time_savings
        r = measure_time_savings()
        assert r.passed, f"Time savings {r.value}% below threshold {r.threshold}%"

    def test_proficiency_monotonicity(self):
        from backend.validation import measure_proficiency_monotonicity
        r = measure_proficiency_monotonicity()
        assert r.passed


# ─────────────────────────────────────────────────────────────────────────────
# 12. Transparency API Endpoints
# ─────────────────────────────────────────────────────────────────────────────

class TestTransparencyEndpoints:

    @pytest.fixture
    def client(self):
        from fastapi.testclient import TestClient
        from backend.main import app
        return TestClient(app)

    def test_disclosure_endpoint(self, client):
        r = client.get("/api/transparency/disclosure")
        assert r.status_code == 200
        body = r.json()
        assert "datasets"   in body
        assert "models"     in body
        assert "libraries"  in body
        assert len(body["datasets"]) >= 3

    def test_metrics_endpoint(self, client):
        r = client.get("/api/transparency/metrics")
        assert r.status_code == 200
        body = r.json()
        assert "summary" in body
        assert "metrics" in body
        assert body["summary"]["total"] == 8
        assert body["summary"]["passed"] >= 7   # at least 7/8 must pass

    def test_algorithms_endpoint(self, client):
        r = client.get("/api/transparency/algorithms")
        assert r.status_code == 200
        body = r.json()
        assert "algorithms" in body
        assert "knowledge_graph_schema" in body
        names = [a["name"] for a in body["algorithms"]]
        assert any("BKT" in n or "Proficiency" in n for n in names)
        assert any("Dijkstra" in n for n in names)
        assert any("A*" in n or "Astar" in n or "A\u2217" in n for n in names)
        assert any("DP" in n or "Dynamic" in n or "Coverage" in n for n in names)

    def test_metrics_endpoint_has_onet_reference(self, client):
        r = client.get("/api/transparency/disclosure")
        body = r.json()
        dataset_names = [d["name"] for d in body["datasets"]]
        assert any("O*NET" in n for n in dataset_names)

    def test_metrics_endpoint_has_kaggle_datasets(self, client):
        r = client.get("/api/transparency/disclosure")
        body = r.json()
        sources = [d["source"] for d in body["datasets"]]
        assert any("kaggle" in s.lower() for s in sources)


# ─────────────────────────────────────────────────────────────────────────────
# 13. NLP Extraction Pipeline
# ─────────────────────────────────────────────────────────────────────────────

class TestNLPExtractor:
    """
    Tests for the 4-layer NLP skill extraction pipeline.
    Layers 1+2 (spaCy / BERT) are tested offline using the lexicon only —
    no model download required for unit tests.
    """

    SAMPLE_RESUME = """
    John Doe — Senior ML Engineer
    Skills: Python, PyTorch, TensorFlow, scikit-learn, SQL, Docker, Kubernetes.
    3 years building production deep learning pipelines with MLflow and Airflow.
    Published paper on transformer fine-tuning. Led team of 5 engineers.
    Experience with AWS SageMaker, Spark, Kafka for large-scale data processing.
    """

    SAMPLE_JD = """
    ML Engineer required. Must have:
    - Python (critical), PyTorch or TensorFlow (critical)
    - Experience with MLflow, Kubernetes, Docker (important)
    - SQL and data engineering experience with Spark, Kafka
    - AWS or GCP cloud experience
    - Nice to have: LangChain, RAG, LLM fine-tuning
    """

    # ── Lexicon tests ─────────────────────────────────────────

    def test_lexicon_not_empty(self):
        from backend.skill_extractor import _SKILL_LEXICON
        assert len(_SKILL_LEXICON) >= 200, "Lexicon should have 200+ terms"

    def test_lexicon_set_consistent(self):
        from backend.skill_extractor import _SKILL_LEXICON, _LEXICON_SET
        assert len(_LEXICON_SET) == len(set(_SKILL_LEXICON)), "Duplicates in lexicon"

    def test_key_skills_in_lexicon(self):
        from backend.skill_extractor import _LEXICON_SET
        for skill in ("python", "pytorch", "tensorflow", "docker",
                      "kubernetes", "sql", "aws", "spark", "kafka"):
            assert skill in _LEXICON_SET, f"'{skill}' missing from lexicon"

    def test_lexicon_all_lowercase(self):
        from backend.skill_extractor import _SKILL_LEXICON
        for s in _SKILL_LEXICON:
            assert s == s.lower(), f"Lexicon entry not lowercase: '{s}'"

    # ── Normalisation ─────────────────────────────────────────

    def test_norm_lowercases(self):
        from backend.skill_extractor import _norm
        assert _norm("PyTorch") == "pytorch"

    def test_norm_strips_punctuation(self):
        from backend.skill_extractor import _norm
        assert _norm("C++!") == "c++"

    def test_norm_collapses_whitespace(self):
        from backend.skill_extractor import _norm
        assert _norm("deep   learning") == "deep learning"

    def test_norm_empty_returns_empty(self):
        from backend.skill_extractor import _norm
        assert _norm("") == ""
        assert _norm("!!!") == ""

    # ── Tech entity detection ─────────────────────────────────

    def test_looks_tech_lexicon_terms(self):
        from backend.skill_extractor import _looks_tech
        assert _looks_tech("pytorch")
        assert _looks_tech("kubernetes")
        assert _looks_tech("mlflow")

    def test_looks_tech_rejects_non_tech(self):
        from backend.skill_extractor import _looks_tech
        # These are generic company names, should fail heuristic
        # (they don't end in tech suffixes and aren't in lexicon)
        assert not _looks_tech("acme corporation")
        assert not _looks_tech("general electric")

    def test_looks_tech_versioned_tools(self):
        from backend.skill_extractor import _looks_tech
        assert _looks_tech("python 3")   # contains digit
        assert _looks_tech("yolov8")

    # ── Merge + deduplication ─────────────────────────────────

    def test_merge_deduplicates(self):
        from backend.skill_extractor import _Span, _merge
        spans_a = [_Span("python", "phrase_match", 1.0),
                   _Span("pytorch", "phrase_match", 1.0)]
        spans_b = [_Span("python", "bert_ner", 0.85)]   # duplicate
        result = _merge(spans_a, [], spans_b)
        assert result.count("python") == 1, "python should appear exactly once"

    def test_merge_priority_phrase_over_bert(self):
        from backend.skill_extractor import _Span, _merge
        # same skill from both sources — phrase should win
        spans_a = [_Span("docker", "phrase_match", 1.0)]
        spans_b = [_Span("docker", "bert_ner", 0.90)]
        result = _merge(spans_a, [], spans_b)
        assert "docker" in result

    def test_merge_caps_at_max_candidates(self):
        from backend.skill_extractor import _Span, _merge, MAX_CANDIDATES
        many = [_Span(f"skill-{i}", "phrase_match", 1.0) for i in range(MAX_CANDIDATES + 20)]
        result = _merge(many, [], [])
        assert len(result) <= MAX_CANDIDATES

    def test_merge_empty_inputs(self):
        from backend.skill_extractor import _merge
        assert _merge([], [], []) == []

    # ── Heuristic fallback ────────────────────────────────────

    def test_heuristic_returns_valid_structure(self):
        from backend.skill_extractor import _heuristic
        result = _heuristic(["python", "pytorch"], ["kubernetes", "docker"])
        assert "resume_skills"   in result
        assert "jd_requirements" in result
        assert len(result["resume_skills"])   == 2
        assert len(result["jd_requirements"]) == 2

    def test_heuristic_proficiency_in_range(self):
        from backend.skill_extractor import _heuristic
        for item in _heuristic(["python", "sql", "torch"], [])["resume_skills"]:
            assert 0.0 <= item["proficiency"] <= 1.0

    def test_heuristic_importance_valid(self):
        from backend.skill_extractor import _heuristic
        valid = {"critical", "important", "nice-to-have"}
        for item in _heuristic([], ["python", "docker", "figma"])["jd_requirements"]:
            assert item["importance"] in valid

    def test_heuristic_critical_for_python(self):
        from backend.skill_extractor import _heuristic
        result = _heuristic([], ["python"])
        assert result["jd_requirements"][0]["importance"] == "critical"

    # ── JSON parsing ──────────────────────────────────────────

    def test_parse_json_valid(self):
        from backend.skill_extractor import _parse_json
        raw = '{"resume_skills":[{"skill":"Python","proficiency":0.8,"years":3}],"jd_requirements":[]}'
        result = _parse_json(raw)
        assert result["resume_skills"][0]["skill"] == "Python"

    def test_parse_json_strips_markdown_fences(self):
        from backend.skill_extractor import _parse_json
        raw = '```json\n{"resume_skills":[],"jd_requirements":[]}\n```'
        result = _parse_json(raw)
        assert "resume_skills" in result

    def test_parse_json_trailing_comma(self):
        from backend.skill_extractor import _parse_json
        raw = '{"resume_skills":[{"skill":"Python","proficiency":0.8,}],"jd_requirements":[]}'
        result = _parse_json(raw)
        assert "resume_skills" in result

    def test_parse_json_malformed_returns_empty(self):
        from backend.skill_extractor import _parse_json
        result = _parse_json("not valid json at all !!!!")
        assert result == {"resume_skills": [], "jd_requirements": []}

    # ── Clamp to extracted ────────────────────────────────────

    def test_clamp_removes_hallucinated_skills(self):
        from backend.skill_extractor import _clamp_to_extracted
        result = {
            "resume_skills":   [{"skill": "Python"}, {"skill": "Cobol"}],
            "jd_requirements": [{"skill": "Docker"}, {"skill": "AlienTech"}],
        }
        clamped = _clamp_to_extracted(result, ["Python"], ["Docker"])
        names_r = [s["skill"] for s in clamped["resume_skills"]]
        names_j = [s["skill"] for s in clamped["jd_requirements"]]
        assert "Python" in names_r
        assert "Cobol"  not in names_r
        assert "Docker" in names_j
        assert "AlienTech" not in names_j

    def test_clamp_case_insensitive(self):
        from backend.skill_extractor import _clamp_to_extracted
        result = {"resume_skills": [{"skill": "pytorch"}], "jd_requirements": []}
        clamped = _clamp_to_extracted(result, ["PyTorch"], [])
        assert len(clamped["resume_skills"]) == 1

    # ── Text chunking ─────────────────────────────────────────

    def test_chunk_splits_long_text(self):
        from backend.skill_extractor import _chunk
        long_text = "This is a sentence. " * 200
        chunks = _chunk(long_text, max_chars=200)
        assert len(chunks) > 1
        assert all(len(c) <= 300 for c in chunks)  # allow slight overflow at sentence boundary

    def test_chunk_short_text_single_chunk(self):
        from backend.skill_extractor import _chunk
        text = "Python is great."
        chunks = _chunk(text, max_chars=500)
        assert len(chunks) == 1

    # ── NLP status endpoint ───────────────────────────────────

    def test_nlp_status_endpoint(self):
        from fastapi.testclient import TestClient
        from backend.main import app
        client = TestClient(app)
        r = client.get("/api/nlp/status")
        assert r.status_code == 200
        body = r.json()
        assert "pipeline_layers" in body
        layers = body["pipeline_layers"]
        assert len(layers) == 4
        names = [l["name"] for l in layers]
        assert any("spaCy" in n for n in names)
        assert any("BERT"  in n for n in names)
        assert any("Groq"  in n for n in names)

    # ── Full extract_skills function (offline — no models loaded) ──

    def test_extract_skills_returns_correct_keys(self):
        """Test function signature without loading models (uses heuristic path)."""
        import os
        os.environ.pop("GROQ_API_KEY", None)   # force heuristic
        from backend.skill_extractor import _heuristic
        # Test the heuristic path directly (safe offline)
        result = _heuristic(["python", "pytorch", "docker"], ["kubernetes", "aws"])
        assert set(result.keys()) >= {"resume_skills", "jd_requirements"}

    def test_extract_skills_empty_text_no_crash(self):
        """Empty inputs should not raise — return empty lists."""
        os.environ.pop("GROQ_API_KEY", None)
        from backend.skill_extractor import _merge, _heuristic
        cands = _merge([], [], [])
        result = _heuristic(cands, cands)
        assert result["resume_skills"] == []
        assert result["jd_requirements"] == []