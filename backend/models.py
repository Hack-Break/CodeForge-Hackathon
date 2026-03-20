
"""
NeuralPath — Pydantic Models v2
Full rich response schema with metrics, domain info, and breakdown fields.
"""
from pydantic import BaseModel, Field, field_validator
from typing import Literal, Optional


class ScoreBreakdownModel(BaseModel):
    llm_score:         float = 0.0
    base_evidence:     float = 0.0
    years_bonus:       float = 0.0
    complexity_bonus:  float = 0.0
    recency_modifier:  float = 0.0
    leadership_bonus:  float = 0.0
    education_bonus:   float = 0.0
    primary_bonus:     float = 0.0
    final_score:       float = 0.0


class PathStepModel(BaseModel):

    @field_validator("hours_saved", "estimated_hours", "traditional_hours", mode="before")
    @classmethod
    def clamp_non_negative(cls, v):
        """Guard against floating-point negatives from prerequisite expansion."""
        try:
            return max(0.0, float(v))
        except (TypeError, ValueError):
            return 0.0

    module_id:            str
    module_name:          str
    action:               Literal["SKIP", "FAST_TRACK", "REQUIRED"]
    domain:               str = "general"
    difficulty:           int = Field(default=2, ge=1, le=5)
    gap_score:            float = Field(ge=0.0, le=1.0)
    proficiency_current:  float = Field(ge=0.0, le=1.0)
    proficiency_required: float = Field(ge=0.0, le=1.0)
    estimated_hours:      float = Field(ge=0.0)
    traditional_hours:    float = Field(default=0.0, ge=0.0)
    hours_saved:          float = Field(default=0.0, ge=0.0)
    confidence:           float = Field(default=0.75, ge=0.0, le=1.0)
    prerequisites:        list[str] = []
    reason:               str = ""
    score_breakdown:      ScoreBreakdownModel = Field(default_factory=ScoreBreakdownModel)


class SummaryModel(BaseModel):
    total_modules:       int
    required:            int
    fast_track:          int
    skipped:             int
    estimated_hours:     float
    traditional_hours:   float = 0.0
    hours_saved:         float = 0.0
    time_saved_pct:      float = 0.0
    algorithm:           str = "dijkstra"
    coverage:            float = 100.0


class DomainInfoModel(BaseModel):
    domain_id:        str
    display_name:     str
    confidence:       float
    matched_keywords: list[str] = []
    onet_roles:       list[str] = []


class CompareMetricsModel(BaseModel):
    total_modules:      int
    required_count:     int
    fast_track_count:   int
    skip_count:         int
    estimated_hours:    float
    traditional_hours:  float
    time_saved_pct:     float
    overall_confidence: float
    domain_breakdown:   dict[str, int] = {}


class AnalyzeResponse(BaseModel):
    summary:          SummaryModel
    pathway:          list[PathStepModel]
    domain_info:      DomainInfoModel
    graph_stats:      dict
    extracted_skills: list[dict] = []     # raw NLP-extracted resume skills (for evaluation)
    extraction_meta:  dict       = Field(default_factory=dict)  # NLP layer hit counts


class CompareResponse(BaseModel):
    scenario_a:  CompareMetricsModel
    scenario_b:  CompareMetricsModel
    pathway_a:   list[PathStepModel]
    pathway_b:   list[PathStepModel]
    diff:        dict = Field(default_factory=dict)


class HealthResponse(BaseModel):
    status:      str
    model:       str
    version:     str
    graph_nodes: int = 0
    graph_edges: int = 0


class GraphStatsResponse(BaseModel):
    total_nodes:    int
    total_edges:    int
    domains:        dict[str, int]
    avg_difficulty: float
    skill_list:     list[dict]