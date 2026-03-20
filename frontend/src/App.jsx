import { useState, useCallback, useEffect } from "react";
import Nav            from "./components/Nav";
import UploadZone     from "./components/UploadZone";
import PathwayViewer  from "./components/PathwayViewer";
import ReasoningTrace from "./components/ReasoningTrace";
import {
  analyzeDocuments, analyzeMlDl, compareScenarios,
  normaliseResult, getNlpStatus,
} from "./api";
import "./styles/App.css";

// ─── Mode constants ────────────────────────────────────────────
const MODES = { ANALYZE: "analyze", MLDL: "mldl", COMPARE: "compare" };

const ML_TRACKS = [
  { value: "",               label: "Auto-detect from JD" },
  { value: "classical",      label: "Classical ML → Production" },
  { value: "computer-vision",label: "Deep Learning → Computer Vision" },
  { value: "nlp-llm",        label: "Deep Learning → NLP / LLMs" },
  { value: "mlops",          label: "MLOps & ML Platform" },
  { value: "rl",             label: "Reinforcement Learning" },
];

// ─── Main App ──────────────────────────────────────────────────
export default function App() {
  const [mode,     setMode]     = useState(MODES.ANALYZE);
  const [loading,  setLoading]  = useState(false);
  const [error,    setError]    = useState(null);
  const [selected, setSelected] = useState(null);

  // Analyze mode
  const [resume,   setResume]   = useState(null);
  const [jd,       setJd]       = useState(null);
  const [result,   setResult]   = useState(null);

  // ML/DL mode
  const [mlResume, setMlResume] = useState(null);
  const [mlJd,     setMlJd]     = useState(null);
  const [mlTrack,  setMlTrack]  = useState("");
  const [mlResult, setMlResult] = useState(null);

  // Compare mode
  const [cmpResumeA, setCmpResumeA] = useState(null);
  const [cmpResumeB, setCmpResumeB] = useState(null);
  const [cmpJd,      setCmpJd]      = useState(null);
  const [cmpResult,  setCmpResult]  = useState(null);

  // NLP diagnostics — fetched after each analysis
  const [nlpStatus, setNlpStatus] = useState(null);

  // ── Mode switch: clear results ─────────────────────────────
  const handleModeChange = (m) => {
    setMode(m);
    setError(null);
    setSelected(null);
  };

  // ── ANALYZE handler ────────────────────────────────────────
  const handleAnalyze = useCallback(async () => {
    if (!resume || !jd || loading) return;
    setLoading(true); setError(null); setResult(null); setSelected(null); setNlpStatus(null);
    try {
      const raw  = await analyzeDocuments(resume, jd);
      setResult(normaliseResult(raw));
      // Non-blocking — fetch NLP diagnostics after result is shown
      getNlpStatus().then(s => s && setNlpStatus(s)).catch(() => {});
    } catch (e) { setError(e?.message || "Something went wrong."); }
    finally     { setLoading(false); }
  }, [resume, jd, loading]);

  // ── ML/DL handler ──────────────────────────────────────────
  const handleMlDl = useCallback(async () => {
    if (!mlResume || loading) return;
    setLoading(true); setError(null); setMlResult(null); setSelected(null); setNlpStatus(null);
    try {
      const raw = await analyzeMlDl(mlResume, mlJd || "ML Engineer deep learning PyTorch", {
        track: mlTrack || null,
      });
      setMlResult(raw);
      getNlpStatus().then(s => s && setNlpStatus(s)).catch(() => {});
    } catch (e) { setError(e?.message || "Something went wrong."); }
    finally     { setLoading(false); }
  }, [mlResume, mlJd, mlTrack, loading]);

  // ── COMPARE handler ────────────────────────────────────────
  const handleCompare = useCallback(async () => {
    if (!cmpResumeA || !cmpResumeB || !cmpJd || loading) return;
    setLoading(true); setError(null); setCmpResult(null);
    try {
      const raw = await compareScenarios(cmpResumeA, cmpResumeB, cmpJd);
      setCmpResult(raw);
    } catch (e) { setError(e?.message || "Something went wrong."); }
    finally     { setLoading(false); }
  }, [cmpResumeA, cmpResumeB, cmpJd, loading]);

  // ── Active pathway + step ──────────────────────────────────
  const activePathway = mode === MODES.MLDL
    ? normaliseResult(mlResult || {}).pathway
    : (result?.pathway ?? []);
  const defaultStep = activePathway.find(p => p.action === "REQUIRED") ?? activePathway[0] ?? null;
  const activeStep  = selected || defaultStep;

  // ── Summary items ──────────────────────────────────────────
  const summaryFor = (r) => r ? [
    { label: "Required",   val: r.summary.required,              color: "#8B82FF" },
    { label: "Fast-Track", val: r.summary.fast_track,            color: "#FFC83D" },
    { label: "Skipped",    val: r.summary.skipped,               color: "#C8F53C" },
    { label: "Hours",      val: `${r.summary.estimated_hours}h`, color: "#3AF5D8" },
  ] : [];

  return (
    <div className="app-container">
      <Nav mode={mode} onModeChange={handleModeChange} />

      <main className="app-main">

        {/* ═══ MODE: ANALYZE ═══════════════════════════════════ */}
        {mode === MODES.ANALYZE && (
          <>
            <div className="mode-header">
              <h1 className="mode-title">Adaptive Learning Pathway</h1>
              <p className="mode-sub">Upload your resume and a job description to get a personalised training roadmap.</p>
            </div>

            <div className="upload-grid">
              <UploadZone label="Resume"          icon="📋" hint="PDF, DOCX, or TXT" file={resume} onFile={setResume} accent="#5B4FFF" />
              <UploadZone label="Job Description" icon="💼" hint="PDF, DOCX, or TXT" file={jd}     onFile={setJd}     accent="#C8F53C" />
            </div>

            <AnalyzeButton
              ready={!!(resume && jd && !loading)}
              loading={loading}
              onClick={handleAnalyze}
              label="Generate Pathway ↗"
            />
            <ErrorBox error={error} />

            {result && (
              <>
                <MetaBadges result={result} />
                {nlpStatus && <NlpDiagnostics status={nlpStatus} />}
                <SummaryStrip items={summaryFor(result)} />
                <ResultsGrid pathway={result.pathway} selected={selected} onSelect={setSelected} activeStep={activeStep} />
              </>
            )}
          </>
        )}

        {/* ═══ MODE: ML/DL ════════════════════════════════════ */}
        {mode === MODES.MLDL && (
          <>
            <div className="mode-header">
              <h1 className="mode-title">ML/DL Learning Pathway</h1>
              <p className="mode-sub">Get a curriculum-ordered ML/DL roadmap tailored to your level across 5 specialisation tracks.</p>
            </div>

            <div className="upload-grid">
              <UploadZone label="Your Resume" icon="📋" hint="PDF, DOCX, or TXT" file={mlResume} onFile={setMlResume} accent="#5B4FFF" />
              <UploadZone label="ML/DL Job Description (optional)" icon="🧠" hint="Leave empty for a generic ML Engineer path" file={mlJd} onFile={setMlJd} accent="#C8F53C" />
            </div>

            {/* Track selector */}
            <div className="track-select-wrap">
              <label className="track-select-label">Specialisation Track</label>
              <div className="track-btns">
                {ML_TRACKS.map(t => (
                  <button
                    key={t.value}
                    className={`track-btn ${mlTrack === t.value ? "active" : ""}`}
                    onClick={() => setMlTrack(t.value)}
                  >
                    {t.label}
                  </button>
                ))}
              </div>
            </div>

            <AnalyzeButton
              ready={!!(mlResume && !loading)}
              loading={loading}
              onClick={handleMlDl}
              label="Build ML/DL Pathway ↗"
            />
            <ErrorBox error={error} />

            {mlResult && (() => {
              const norm = normaliseResult(mlResult);
              const lvl  = mlResult.level_assessment;
              const tl   = mlResult.timeline;
              const ts   = mlResult.time_saved;
              return (
                <>
                  {/* Level card */}
                  {lvl && (
                    <div className="level-card">
                      <div className="level-card-left">
                        <div className="level-badge">L{lvl.level}</div>
                        <div>
                          <div className="level-label">{lvl.label}</div>
                          <div className="level-sub">
                            Strongest: <strong>{lvl.strongest_area}</strong> ·
                            Weakest: <strong>{lvl.weakest_area}</strong>
                          </div>
                        </div>
                      </div>
                      <div className="level-scores">
                        <LevelBar label="Foundations"  val={lvl.foundation_score} />
                        <LevelBar label="Deep Learning" val={lvl.dl_score} />
                        <LevelBar label="Specialisation" val={lvl.specialisation_score} />
                      </div>
                    </div>
                  )}

                  {/* Track badge */}
                  {mlResult.track && (
                    <div className="meta-badges">
                      <span className="meta-badge domain-badge">🎯 {mlResult.track_description}</span>
                      {ts?.time_saved_pct > 0 && (
                        <span className="meta-badge savings-badge">⚡ {ts.time_saved_pct}% faster</span>
                      )}
                    </div>
                  )}

                  {nlpStatus && <NlpDiagnostics status={nlpStatus} />}

                  <SummaryStrip items={summaryFor(norm)} />

                  {/* Timeline phases */}
                  {tl?.phases?.length > 0 && (
                    <div className="timeline-wrap">
                      <h3 className="timeline-title">📅 Learning Timeline — {tl.estimated_completion}</h3>
                      <div className="timeline-phases">
                        {tl.phases.map(ph => (
                          <div key={ph.phase} className="timeline-phase">
                            <div className="phase-header">
                              <span className="phase-label">{ph.label}</span>
                              <span className="phase-weeks">{ph.weeks}</span>
                              <span className="phase-hours">{ph.phase_hours}h</span>
                            </div>
                            <div className="phase-modules">
                              {ph.modules.map(m => (
                                <span key={m.module_id} className={`phase-mod-tag action-${m.action.toLowerCase()}`}>
                                  {m.module_name}
                                </span>
                              ))}
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}

                  <ResultsGrid pathway={norm.pathway} selected={selected} onSelect={setSelected} activeStep={activeStep} />
                </>
              );
            })()}
          </>
        )}

        {/* ═══ MODE: COMPARE ══════════════════════════════════ */}
        {mode === MODES.COMPARE && (
          <>
            <div className="mode-header">
              <h1 className="mode-title">Scenario Comparison</h1>
              <p className="mode-sub">Upload two resumes against the same JD to see exactly how the adaptive engine personalises differently.</p>
            </div>

            <div className="compare-upload-grid">
              <UploadZone label="Resume A (e.g. Fresher)"  icon="👤" hint="PDF, DOCX, or TXT" file={cmpResumeA} onFile={setCmpResumeA} accent="#5B4FFF" />
              <UploadZone label="Resume B (e.g. Senior)"   icon="👤" hint="PDF, DOCX, or TXT" file={cmpResumeB} onFile={setCmpResumeB} accent="#FFC83D" />
              <UploadZone label="Job Description (shared)" icon="💼" hint="Same JD for both"   file={cmpJd}     onFile={setCmpJd}     accent="#C8F53C" />
            </div>

            <AnalyzeButton
              ready={!!(cmpResumeA && cmpResumeB && cmpJd && !loading)}
              loading={loading}
              onClick={handleCompare}
              label="Compare Scenarios ↗"
            />
            <ErrorBox error={error} />

            {cmpResult && (
              <>
                {/* Delta summary */}
                {cmpResult.diff && (
                  <div className="compare-diff-card">
                    <div className="diff-title">📊 Comparison Summary</div>
                    <p className="diff-interpretation">{cmpResult.diff.interpretation}</p>
                    <div className="diff-grid">
                      <DiffMetric label="More experienced"   val={`Candidate ${cmpResult.diff.more_experienced}`} />
                      <DiffMetric label="Hours delta"        val={`${Math.abs(cmpResult.diff.hours_delta)}h ${cmpResult.diff.hours_delta > 0 ? "(A needs more)" : "(B needs more)"}`} />
                      <DiffMetric label="Time saved — A"     val={`${cmpResult.diff.time_saved_pct_a}%`} highlight />
                      <DiffMetric label="Time saved — B"     val={`${cmpResult.diff.time_saved_pct_b}%`} highlight />
                      <DiffMetric label="Required modules Δ" val={`${Math.abs(cmpResult.diff.required_delta)} fewer for ${cmpResult.diff.required_delta > 0 ? "B" : "A"}`} />
                    </div>
                  </div>
                )}

                {/* Side-by-side metrics */}
                <div className="compare-side-grid">
                  <ScenarioPanel
                    label="Candidate A"
                    color="#8B82FF"
                    metrics={cmpResult.scenario_a}
                    pathway={cmpResult.pathway_a?.map(normaliseStep)}
                  />
                  <ScenarioPanel
                    label="Candidate B"
                    color="#FFC83D"
                    metrics={cmpResult.scenario_b}
                    pathway={cmpResult.pathway_b?.map(normaliseStep)}
                  />
                </div>
              </>
            )}
          </>
        )}

      </main>
    </div>
  );
}

// ─── Shared sub-components ─────────────────────────────────────

function AnalyzeButton({ ready, loading, onClick, label }) {
  return (
    <div className="analyze-area">
      <button
        className={`analyze-btn ${ready ? "ready" : "disabled"}`}
        onClick={onClick}
        disabled={!ready}
      >
        {loading ? "⚡ Analyzing…" : label}
      </button>
      {loading && (
        <div className="loading-bar-wrap">
          <div className="loading-bar-track"><div className="loading-bar-fill" /></div>
          <p className="loading-text">NLP pipeline extracting skills · building your adaptive pathway…</p>
        </div>
      )}
    </div>
  );
}

function ErrorBox({ error }) {
  if (!error) return null;
  return <div className="error-box">⚠ {error}</div>;
}

function MetaBadges({ result }) {
  const pct    = result?.summary?.time_saved_pct ?? 0;
  const domain = result?.domain_info?.display_name ?? "";
  const conf   = result?.domain_info?.confidence ?? 0;
  const algo   = result?.summary?.algorithm ?? "";
  if (!domain && !pct) return null;
  return (
    <div className="meta-badges">
      {domain && <span className="meta-badge domain-badge">🎯 {domain}{conf > 0 && <span className="meta-conf"> {Math.round(conf*100)}%</span>}</span>}
      {pct > 0 && <span className="meta-badge savings-badge">⚡ {pct}% faster than traditional</span>}
      {algo   && <span className="meta-badge algo-badge">🔁 {algo.toUpperCase()}</span>}
    </div>
  );
}

function SummaryStrip({ items }) {
  if (!items?.length) return null;
  return (
    <div className="summary-grid">
      {items.map(s => (
        <div key={s.label} className="summary-card">
          <div className="summary-val" style={{ color: s.color }}>{s.val}</div>
          <div className="summary-label">{s.label}</div>
        </div>
      ))}
    </div>
  );
}

function ResultsGrid({ pathway, selected, onSelect, activeStep }) {
  if (!pathway?.length) return (
    <div className="empty-state">
      <span className="empty-state-icon">🤔</span>
      <p>No skills could be extracted. Try a more detailed resume or job description.</p>
    </div>
  );
  return (
    <div className="results-grid">
      <PathwayViewer pathway={pathway} selected={selected} onSelect={onSelect} />
      <ReasoningTrace step={activeStep} />
    </div>
  );
}

function LevelBar({ label, val }) {
  const pct = Math.round((val || 0) * 100);
  return (
    <div className="level-bar-row">
      <div className="level-bar-label">{label} <span>{pct}%</span></div>
      <div className="level-bar-track">
        <div className="level-bar-fill" style={{ width: `${pct}%` }} />
      </div>
    </div>
  );
}

function DiffMetric({ label, val, highlight }) {
  return (
    <div className="diff-metric">
      <div className="diff-metric-label">{label}</div>
      <div className="diff-metric-val" style={{ color: highlight ? "#c8f53c" : "#a99fff" }}>{val}</div>
    </div>
  );
}

function ScenarioPanel({ label, color, metrics, pathway }) {
  if (!metrics) return null;
  return (
    <div className="scenario-panel" style={{ borderColor: `${color}30` }}>
      <div className="scenario-label" style={{ color }}>{label}</div>
      <div className="scenario-stats">
        <div className="scenario-stat"><span>Required</span><strong style={{ color }}>{metrics.required_count}</strong></div>
        <div className="scenario-stat"><span>Fast-Track</span><strong style={{ color: "#FFC83D" }}>{metrics.fast_track_count}</strong></div>
        <div className="scenario-stat"><span>Skipped</span><strong style={{ color: "#C8F53C" }}>{metrics.skip_count}</strong></div>
        <div className="scenario-stat"><span>Hours</span><strong style={{ color: "#3AF5D8" }}>{metrics.estimated_hours}h</strong></div>
        <div className="scenario-stat"><span>Time saved</span><strong style={{ color: "#c8f53c" }}>{metrics.time_saved_pct}%</strong></div>
      </div>
      {pathway?.length > 0 && (
        <div className="scenario-pathway">
          {pathway.slice(0, 6).map((s, i) => {
            const colors = { REQUIRED: "#8B82FF", FAST_TRACK: "#FFC83D", SKIP: "#C8F53C" };
            return (
              <div key={i} className="scenario-mod-row">
                <span className="scenario-mod-dot" style={{ background: colors[s.action] || "#888" }} />
                <span className="scenario-mod-name">{s.module_name}</span>
                <span className="scenario-mod-hrs">{s.estimated_hours}h</span>
              </div>
            );
          })}
          {pathway.length > 6 && <div className="scenario-more">+{pathway.length - 6} more modules</div>}
        </div>
      )}
    </div>
  );
}

// normalise a raw pathway step (from compare response) to the same shape
function normaliseStep(m, i) {
  return {
    module_id:            m?.module_id ?? `mod-${i}`,
    module_name:          m?.module_name ?? `Module ${i+1}`,
    action:               ["REQUIRED","FAST_TRACK","SKIP"].includes(m?.action) ? m.action : "REQUIRED",
    estimated_hours:      Number(m?.estimated_hours) || 0,
    traditional_hours:    Number(m?.traditional_hours) || 0,
    hours_saved:          Number(m?.hours_saved) || 0,
    confidence:           Number(m?.confidence) || 0.75,
    proficiency_current:  Number(m?.proficiency_current) || 0,
    proficiency_required: Number(m?.proficiency_required) || 0.7,
    gap_score:            Number(m?.gap_score) || 0,
    prerequisites:        Array.isArray(m?.prerequisites) ? m.prerequisites : [],
    domain:               m?.domain ?? "general",
    difficulty:           Number(m?.difficulty) || 2,
    reason:               m?.reason ?? "",
    score_breakdown:      m?.score_breakdown ?? {},
  };
}

// ─── NLP Diagnostics strip ─────────────────────────────────────
// Shows a compact, collapsible readout of the 4-layer NLP pipeline
// hit counts from the most recent extraction call.
function NlpDiagnostics({ status }) {
  const [open, setOpen] = useState(false);
  if (!status) return null;

  const spacy   = status.spacy_ner_hits    ?? 0;
  const phrase  = status.phrase_match_hits ?? 0;
  const bert    = status.bert_ner_hits     ?? 0;
  const total   = spacy + phrase + bert;
  const rCount  = status.resume_skill_count ?? 0;
  const jCount  = status.jd_skill_count    ?? 0;
  const groqOk  = status.groq_scored;
  const fallback= status.groq_fallback;

  return (
    <div className="nlp-diag-wrap">
      <button
        className="nlp-diag-toggle"
        onClick={() => setOpen(o => !o)}
        title="NLP extraction pipeline diagnostics"
      >
        <span className="nlp-diag-icon">🔬</span>
        <span className="nlp-diag-summary">
          NLP pipeline · {rCount} resume skills · {jCount} JD skills · {total} raw detections
        </span>
        <span className="nlp-diag-caret">{open ? "▲" : "▼"}</span>
      </button>

      {open && (
        <div className="nlp-diag-body">
          <div className="nlp-diag-grid">
            <NlpLayer layer="L1 spaCy NER"         hits={spacy}  active={spacy  > 0} model={status.models?.spacy}   />
            <NlpLayer layer="L2 PhraseMatcher"      hits={phrase} active={phrase > 0} model="1 400-skill lexicon"    />
            <NlpLayer layer="L3 BERT NER"           hits={bert}   active={bert   > 0} model={status.models?.bert}    />
            <NlpLayer
              layer="L4 Groq scoring"
              hits={groqOk ? rCount + jCount : 0}
              active={groqOk}
              model={status.models?.scorer_llm}
              warn={fallback ? "fallback scores used" : null}
            />
          </div>
          <div className="nlp-diag-note">
            Extraction is deterministic (NLP layers) · LLM assigned proficiency scores only
          </div>
        </div>
      )}
    </div>
  );
}

function NlpLayer({ layer, hits, active, model, warn }) {
  return (
    <div className={`nlp-layer ${active ? "nlp-layer-active" : "nlp-layer-idle"}`}>
      <div className="nlp-layer-header">
        <span className={`nlp-layer-dot ${active ? "dot-on" : "dot-off"}`} />
        <span className="nlp-layer-name">{layer}</span>
        <span className="nlp-layer-hits">{hits} hits</span>
      </div>
      <div className="nlp-layer-model">{model ?? "—"}</div>
      {warn && <div className="nlp-layer-warn">⚠ {warn}</div>}
    </div>
  );
}
