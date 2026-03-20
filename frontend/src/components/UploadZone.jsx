import { useRef, useState } from "react";
import "../styles/UploadZone.css";

export default function UploadZone({ label, icon, hint, file, onFile, accent }) {
  const inputRef          = useRef(null);
  const [dragging, setDragging] = useState(false);

  // ── Drag handlers ──────────────────────────────────────────
  const onDragOver  = (e) => { e.preventDefault(); setDragging(true); };
  const onDragLeave = ()  => setDragging(false);
  const onDrop      = (e) => {
    e.preventDefault();
    setDragging(false);
    const dropped = e.dataTransfer.files[0];
    if (dropped) onFile(dropped);
  };

  // ── Click to browse ────────────────────────────────────────
  const onClick = () => inputRef.current?.click();

  const onChange = (e) => {
    const picked = e.target.files?.[0];
    if (picked) onFile(picked);
  };

  // ── Dynamic styles based on state ─────────────────────────
  const borderColor = dragging || file
    ? accent
    : "rgba(255,255,255,0.1)";

  const bgColor = file
    ? `${accent}0A`
    : dragging
    ? `${accent}07`
    : "transparent";

  return (
    <div
      className={`upload-zone ${file ? "has-file" : ""} ${dragging ? "dragging" : ""}`}
      style={{ borderColor, background: bgColor }}
      onDragOver={onDragOver}
      onDragLeave={onDragLeave}
      onDrop={onDrop}
      onClick={onClick}
      role="button"
      tabIndex={0}
      onKeyDown={(e) => e.key === "Enter" && onClick()}
      aria-label={`Upload ${label}`}
    >
      <input
        ref={inputRef}
        type="file"
        accept=".pdf,.docx,.txt"
        style={{ display: "none" }}
        onChange={onChange}
      />

      {/* Icon */}
      <span className="upload-zone-icon">
        {file ? "✅" : icon}
      </span>

      {/* Title */}
      <div
        className="upload-zone-title"
        style={{ color: file ? accent : "var(--paper)" }}
      >
        {file ? "File ready" : label}
      </div>

      {/* File info OR hint */}
      {file ? (
        <>
          <div className="upload-zone-filename">{file.name}</div>
          <div className="upload-zone-filesize">
            {(file.size / 1024).toFixed(0)} KB · click to replace
          </div>
        </>
      ) : (
        <>
          <div className="upload-zone-hint">{hint}</div>
          <div className="upload-zone-types">
            {["PDF", "DOCX", "TXT"].map((ext) => (
              <span key={ext} className="upload-type-badge">{ext}</span>
            ))}
          </div>
        </>
      )}
    </div>
  );
}
