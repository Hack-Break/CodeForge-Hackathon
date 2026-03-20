// ─── NeuralPath API Client v2 ────────────────────────────────────────────────
// Wired to the new backend with full adaptive engine response fields.

const API_BASE = import.meta.env.VITE_API_URL || "";

// ─────────────────────────────────────────────────────────────────────────────
// Core analyze — resume + JD → adaptive pathway
// ─────────────────────────────────────────────────────────────────────────────
export async function analyzeDocuments(resume, jd, { algorithm = "auto" } = {}) {
  const form = new FormData();
  form.append("resume", resume);
  form.append("algorithm", algorithm);

  if (typeof jd === "string") {
    form.append("jd_text", jd);
  } else {
    form.append("jd", jd);
  }

  const res  = await fetch(`${API_BASE}/api/analyze`, { method: "POST", body: form });
  const data = await res.json().catch(() => null);

  if (!res.ok) throw new Error(data?.detail || `Server error ${res.status}`);
  if (!data)   throw new Error("Empty response from server");
  return data;
}

// ─────────────────────────────────────────────────────────────────────────────
// ML/DL dedicated pathway
// ─────────────────────────────────────────────────────────────────────────────
export async function analyzeMlDl(resume, jd, { track = null, hoursPerWeek = 10 } = {}) {
  const form = new FormData();
  form.append("resume", resume);
  form.append("hours_per_week", hoursPerWeek);
  if (track) form.append("track", track);

  if (typeof jd === "string") {
    form.append("jd_text", jd);
  } else if (jd) {
    form.append("jd", jd);
  }

  const res  = await fetch(`${API_BASE}/api/pathway/mldl`, { method: "POST", body: form });
  const data = await res.json().catch(() => null);

  if (!res.ok) throw new Error(data?.detail || `Server error ${res.status}`);
  return data;
}

// ─────────────────────────────────────────────────────────────────────────────
// Scenario comparison — two resumes vs same JD
// ─────────────────────────────────────────────────────────────────────────────
export async function compareScenarios(resumeA, resumeB, jd) {
  const form = new FormData();
  form.append("resume_a", resumeA);
  form.append("resume_b", resumeB);

  if (typeof jd === "string") {
    form.append("jd_text", jd);
  } else {
    form.append("jd", jd);
  }

  const res  = await fetch(`${API_BASE}/api/compare`, { method: "POST", body: form });
  const data = await res.json().catch(() => null);

  if (!res.ok) throw new Error(data?.detail || `Server error ${res.status}`);
  return data;
}

// ─────────────────────────────────────────────────────────────────────────────
// Analytics endpoints
// ─────────────────────────────────────────────────────────────────────────────
export async function getRadarData(resume, jd) {
  const form = new FormData();
  form.append("resume", resume);
  if (typeof jd === "string") form.append("jd_text", jd); else form.append("jd", jd);
  const res = await fetch(`${API_BASE}/api/analytics/radar`, { method: "POST", body: form });
  return res.json();
}

export async function getTimeline(resume, jd, hoursPerWeek = 10) {
  const form = new FormData();
  form.append("resume", resume);
  form.append("hours_per_week", hoursPerWeek);
  if (typeof jd === "string") form.append("jd_text", jd); else form.append("jd", jd);
  const res = await fetch(`${API_BASE}/api/analytics/timeline`, { method: "POST", body: form });
  return res.json();
}

export async function getTimeSavings(resume, jd) {
  const form = new FormData();
  form.append("resume", resume);
  if (typeof jd === "string") form.append("jd_text", jd); else form.append("jd", jd);
  const res = await fetch(`${API_BASE}/api/analytics/savings`, { method: "POST", body: form });
  return res.json();
}

// ─────────────────────────────────────────────────────────────────────────────
// NLP pipeline diagnostics
// ─────────────────────────────────────────────────────────────────────────────
export async function getNlpStatus() {
  const res = await fetch(`${API_BASE}/api/nlp/status`);
  if (!res.ok) return null;
  return res.json();
}

// ─────────────────────────────────────────────────────────────────────────────
// Transparency / Disclosure endpoints
// ─────────────────────────────────────────────────────────────────────────────
export async function getDisclosure() {
  const res = await fetch(`${API_BASE}/api/transparency/disclosure`);
  return res.json();
}

export async function getValidationMetrics() {
  const res = await fetch(`${API_BASE}/api/transparency/metrics`);
  return res.json();
}

export async function getAlgorithmDocs() {
  const res = await fetch(`${API_BASE}/api/transparency/algorithms`);
  return res.json();
}

// ─────────────────────────────────────────────────────────────────────────────
// Normalise API response — safely maps new rich fields + backwards-compat
// ─────────────────────────────────────────────────────────────────────────────
export function normaliseResult(data) {
  const pathway = Array.isArray(data?.pathway) ? data.pathway : [];

  const safePathway = pathway.map((m, i) => ({
    // Core fields (backwards-compatible)
    module_id:            String(m?.module_id            ?? `module-${i}`),
    module_name:          String(m?.module_name          ?? m?.skill ?? `Module ${i + 1}`),
    action:               ["REQUIRED", "FAST_TRACK", "SKIP"].includes(m?.action) ? m.action : "REQUIRED",
    gap_score:            Math.max(0, Math.min(1, Number(m?.gap_score)            || 0)),
    proficiency_current:  Math.max(0, Math.min(1, Number(m?.proficiency_current)  || 0)),
    proficiency_required: Math.max(0, Math.min(1, Number(m?.proficiency_required) || 0.7)),
    estimated_hours:      Math.max(0, Number(m?.estimated_hours) || 0),
    prerequisites:        Array.isArray(m?.prerequisites) ? m.prerequisites : [],
    reason:               String(m?.reason ?? ""),
    // New rich fields from v2 engine
    domain:            String(m?.domain            ?? "general"),
    difficulty:        Number(m?.difficulty        ?? 2),
    traditional_hours: Math.max(0, Number(m?.traditional_hours) || 0),
    hours_saved:       Math.max(0, Number(m?.hours_saved)       || 0),
    confidence:        Math.max(0, Math.min(1, Number(m?.confidence) || 0.75)),
    score_breakdown:   m?.score_breakdown ?? {},
  }));

  const summary = data?.summary ?? {};

  return {
    summary: {
      total_modules:     Number(summary.total_modules)     || safePathway.length,
      required:          Number(summary.required)          || safePathway.filter(m => m.action === "REQUIRED").length,
      fast_track:        Number(summary.fast_track)        || safePathway.filter(m => m.action === "FAST_TRACK").length,
      skipped:           Number(summary.skipped)           || safePathway.filter(m => m.action === "SKIP").length,
      estimated_hours:   Number(summary.estimated_hours)   || 0,
      // New v2 fields
      traditional_hours: Number(summary.traditional_hours) || 0,
      hours_saved:       Number(summary.hours_saved)       || 0,
      time_saved_pct:    Number(summary.time_saved_pct)    || 0,
      algorithm:         String(summary.algorithm          ?? "auto"),
      coverage:          Number(summary.coverage)          || 100,
    },
    pathway: safePathway,
    // New v2 top-level fields (safe defaults if missing)
    domain_info: data?.domain_info ?? { domain_id: "general", display_name: "General", confidence: 0 },
    graph_stats: data?.graph_stats ?? {},
  };
}
