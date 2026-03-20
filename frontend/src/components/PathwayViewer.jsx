import "../styles/PathwayViewer.css";
import { getAction } from "../constants";

// Difficulty labels
const DIFF_LABEL = { 1: "Beginner", 2: "Intermediate", 3: "Intermediate", 4: "Advanced", 5: "Expert" };
const DOMAIN_ICON = {
  ml: "🧠", software: "💻", cloud: "☁️", "data-eng": "📊",
  security: "🔒", product: "📋", general: "🎯",
};

export default function PathwayViewer({ pathway, selected, onSelect }) {
  if (!Array.isArray(pathway) || pathway.length === 0) {
    return (
      <div className="pathway-viewer">
        <p style={{ color: "var(--muted)", fontSize: "0.9rem" }}>No modules to display.</p>
      </div>
    );
  }

  const totalHours = pathway.reduce((s, m) => s + (m.estimated_hours || 0), 0);
  const totalSaved = pathway.reduce((s, m) => s + (m.hours_saved     || 0), 0);

  return (
    <div className="pathway-viewer">
      <div className="pathway-viewer-header">
        <h2>Your Learning Pathway</h2>
        <p className="pathway-viewer-meta">
          {pathway.length} modules · {totalHours}h adaptive
          {totalSaved > 0 && ` · ${totalSaved}h saved`}
          {" · "}click any row to see AI reasoning
        </p>
      </div>

      <div className="pathway-list">
        {pathway.map((step, i) => (
          <ModuleRow
            key={step.module_id || i}
            step={step}
            index={i}
            isSelected={selected?.module_id === step.module_id}
            onSelect={onSelect}
          />
        ))}
      </div>
    </div>
  );
}

function ModuleRow({ step, index, isSelected, onSelect }) {
  const cfg    = getAction(step.action);
  const curPct = Math.round((step.proficiency_current  || 0) * 100);
  const reqPct = Math.round((step.proficiency_required || 0) * 100);
  const isSkip = step.action === "SKIP";
  const conf   = Math.round((step.confidence || 0.75) * 100);
  const diffLabel = DIFF_LABEL[step.difficulty] ?? "";
  const domainIcon = DOMAIN_ICON[step.domain] ?? "📚";

  return (
    <div
      className={`module-row action-${step.action.toLowerCase()} ${isSelected ? "selected" : ""}`}
      style={{ borderColor: isSelected ? cfg.color : "rgba(255,255,255,0.05)" }}
      onClick={() => onSelect(step)}
      role="button"
      tabIndex={0}
      onKeyDown={(e) => e.key === "Enter" && onSelect(step)}
      aria-label={`Module: ${step.module_name}`}
    >
      {/* Badge */}
      <div
        className="module-badge"
        style={{ background: cfg.bgColor, color: cfg.color, border: `1px solid ${cfg.border}` }}
      >
        {isSkip ? "✓" : index + 1}
      </div>

      {/* Content */}
      <div className="module-content">
        <div className="module-name-row">
          <span className={`module-name ${isSkip ? "skip" : ""}`}>
            {domainIcon} {step.module_name}
          </span>
          <span
            className="action-badge"
            style={{ background: cfg.bgColor, color: cfg.color, border: `1px solid ${cfg.border}` }}
          >
            {cfg.label}
          </span>
          {diffLabel && !isSkip && (
            <span className="diff-badge">{diffLabel}</span>
          )}
        </div>

        {/* Progress bar — non-skip only */}
        {!isSkip && (
          <div className="module-progress">
            <div
              className="module-progress-fill"
              style={{ width: `${curPct}%`, background: cfg.color }}
            />
          </div>
        )}
      </div>

      {/* Right side */}
      <div className="module-meta-right">
        <div className="module-hours" style={{ color: cfg.color }}>
          {isSkip ? "0h" : `${step.estimated_hours}h`}
        </div>
        <div className="module-pct">{curPct}% → {reqPct}%</div>
        {step.hours_saved > 0 && (
          <div className="module-saved">−{step.hours_saved}h saved</div>
        )}
        <div className="module-conf" title="Recommendation confidence">
          {conf}% conf
        </div>
      </div>
    </div>
  );
}
