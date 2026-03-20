import "../styles/Nav.css";

export default function Nav({ mode, onModeChange }) {
  return (
    <nav className="navbar">
      <div className="navbar-logo">
        Neural<span>Path</span>
      </div>

      {/* Mode switcher */}
      <div className="navbar-modes">
        <button
          className={`mode-btn ${mode === "analyze" ? "active" : ""}`}
          onClick={() => onModeChange?.("analyze")}
        >
          🎯 Analyze
        </button>
        <button
          className={`mode-btn ${mode === "mldl" ? "active" : ""}`}
          onClick={() => onModeChange?.("mldl")}
        >
          🧠 ML/DL Path
        </button>
        <button
          className={`mode-btn ${mode === "compare" ? "active" : ""}`}
          onClick={() => onModeChange?.("compare")}
        >
          ⚖️ Compare
        </button>
      </div>

      <div className="navbar-tag">AI-Adaptive Onboarding Engine</div>
    </nav>
  );
}
