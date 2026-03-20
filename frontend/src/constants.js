// ─── NeuralPath Constants ────────────────────────────────────

export const ACTION = {
  REQUIRED: {
    color:   "#8B82FF",
    bgColor: "rgba(91,79,255,0.12)",
    border:  "rgba(91,79,255,0.3)",
    label:   "Required",
  },
  FAST_TRACK: {
    color:   "#FFC83D",
    bgColor: "rgba(255,200,61,0.1)",
    border:  "rgba(255,200,61,0.3)",
    label:   "Fast-Track",
  },
  SKIP: {
    color:   "#C8F53C",
    bgColor: "rgba(200,245,60,0.08)",
    border:  "rgba(200,245,60,0.2)",
    label:   "Skip",
  },
};

export function getAction(key) {
  return ACTION[key] || ACTION.REQUIRED;
}
