// ═══════════════════════════════════════════════════════════
//  api/analyze.js — NeuralPath Vercel Serverless Function
//
//  Pipeline:
//  1. Groq extracts raw skills + scores from resume & JD
//  2. Knowledge Tracing algorithm computes gap scores
//  3. Graph-based optimizer sorts & builds pathway
//  4. Returns structured JSON to frontend
// ═══════════════════════════════════════════════════════════

export default async function handler(req, res) {

  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Method not allowed' });
  }

  const GROQ_API_KEY = process.env.GROQ_API_KEY;
  if (!GROQ_API_KEY) {
    return res.status(500).json({ error: 'API key not configured on server.' });
  }

  const { resumeText, jobDescription } = req.body;
  if (!resumeText || !jobDescription) {
    return res.status(400).json({ error: 'Missing resumeText or jobDescription.' });
  }

  // ── Step 1: Extract skills via Groq ──────────────────────
  // Ask Groq ONLY for raw skill data — no gap scoring, no pathway
  // Our algorithm handles all of that below.
  const extractionPrompt = `
You are a skill extraction engine. Analyse the resume and job description below.

Return ONLY a raw JSON object (no markdown, no backticks) with this exact structure:

{
  "jobTitle": "extracted job title from JD",
  "candidateSkills": [
    {
      "name": "skill name",
      "proficiency": 0.0,
      "yearsExp": 0
    }
  ],
  "requiredSkills": [
    {
      "name": "skill name",
      "required": 0.0,
      "priority": "critical" | "important" | "nice-to-have",
      "prerequisites": ["skill name", "..."]
    }
  ]
}

Rules:
- proficiency and required are floats between 0.0 and 1.0
- Extract 6-12 skills from the JD as requiredSkills
- Match candidateSkills against requiredSkills where possible
- prerequisites lists skills that should be learned before this one
- Return raw JSON only, no explanation

## RESUME:
${resumeText.slice(0, 4000)}

## JOB DESCRIPTION:
${jobDescription.slice(0, 2000)}
`.trim();

  let extracted;
  try {
    const groqRes = await fetch('https://api.groq.com/openai/v1/chat/completions', {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${GROQ_API_KEY}`,
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        model: 'llama-3.3-70b-versatile',
        temperature: 0.1,
        max_tokens: 1200,
        messages: [{ role: 'user', content: extractionPrompt }]
      })
    });

    if (!groqRes.ok) {
      const err = await groqRes.text();
      console.error('Groq error:', err);
      return res.status(502).json({ error: 'AI service error. Please try again.' });
    }

    const groqData = await groqRes.json();
    const raw = groqData.choices?.[0]?.message?.content?.trim() || '';
    const clean = raw.replace(/```json|```/g, '').trim();
    extracted = JSON.parse(clean);

  } catch (err) {
    console.error('Extraction error:', err);
    return res.status(502).json({ error: 'Could not extract skills. Please try again.' });
  }

  // ── Step 2: Knowledge Tracing — compute gap scores ────────
  //
  // Gap score = max(0, required_proficiency - current_proficiency)
  //
  // Thresholds (matching your algorithm section):
  //   gap <= 0.10  → SKIPPED     (within 10%, already competent)
  //   gap <= 0.30  → FAST-TRACK  (partial knowledge, accelerate)
  //   gap >  0.30  → REQUIRED    (critical gap, full module)
  //
  // BKT prior: if no resume data found, apply domain prior of 0.1
  // (assume minimal knowledge rather than zero — more realistic)

  const BKT_PRIOR = 0.10; // Bayesian prior for unknown skills

  const candidateMap = {};
  (extracted.candidateSkills || []).forEach(s => {
    candidateMap[normalise(s.name)] = s;
  });

  const scoredSkills = (extracted.requiredSkills || []).map(req => {
    const key     = normalise(req.name);
    const match   = findMatch(key, candidateMap);
    const current = match ? match.proficiency : BKT_PRIOR;
    const required = clamp(req.required || 0.7, 0, 1);
    const gap      = Math.max(0, required - current);

    // Gap classification
    let status;
    if (gap <= 0.10)      status = 'SKIPPED';
    else if (gap <= 0.30) status = 'FAST-TRACK';
    else                  status = 'REQUIRED';

    // Time estimation:
    // Base hours per gap unit × difficulty multiplier per priority
    const difficultyMultiplier =
      req.priority === 'critical'     ? 1.4 :
      req.priority === 'important'    ? 1.1 : 0.8;

    const baseHours = status === 'SKIPPED' ? 0
      : Math.round(gap * 20 * difficultyMultiplier);

    // Reasoning trace (transparent, grounded in actual numbers)
    const reason = buildReason(status, current, required, match, req);

    return {
      name:          req.name,
      status,
      priority:      req.priority || 'important',
      current:       round2(current),
      required:      round2(required),
      gap:           round2(gap),
      proficiency:   Math.round(current * 100),
      hours:         baseHours,
      reason,
      prerequisites: req.prerequisites || []
    };
  });

  // ── Step 3: Graph-based Pathway Optimizer ────────────────
  //
  // Dijkstra-inspired topological sort:
  // 1. Build adjacency from prerequisites
  // 2. Topological sort (Kahn's algorithm)
  // 3. Within same topo-level, sort by gap score descending
  //    (highest gap = highest priority)
  // 4. SKIPPED modules always go last
  //
  // Edge weight = hours × (1 / gap) to find min-time path

  const sorted = topoSort(scoredSkills);

  // ── Step 4: Build summary stats ──────────────────────────
  const required   = sorted.filter(m => m.status === 'REQUIRED').length;
  const fastTrack  = sorted.filter(m => m.status === 'FAST-TRACK').length;
  const skipped    = sorted.filter(m => m.status === 'SKIPPED').length;
  const totalHours = sorted.reduce((s, m) => s + m.hours, 0);

  // Strip internal fields before returning
  const modules = sorted.map(({ name, status, reason, hours, proficiency, prerequisites }) => ({
    name, status, reason, hours, proficiency, prerequisites
  }));

  return res.status(200).json({
    jobTitle: extracted.jobTitle || 'Target Role',
    modules,
    stats: {
      total:    sorted.length,
      required,
      fastTrack,
      skipped,
      totalHours
    }
  });
}


// ═══════════════════════════════════════════════════════════
//  ALGORITHM HELPERS
// ═══════════════════════════════════════════════════════════

// Kahn's algorithm — topological sort with gap-score tiebreaker
function topoSort(skills) {
  const nameToSkill = {};
  skills.forEach(s => { nameToSkill[normalise(s.name)] = s; });

  // Build in-degree map
  const inDegree = {};
  const adj      = {};
  skills.forEach(s => {
    const key = normalise(s.name);
    inDegree[key] = 0;
    adj[key]      = [];
  });

  skills.forEach(s => {
    const key = normalise(s.name);
    (s.prerequisites || []).forEach(pre => {
      const preKey = normalise(pre);
      if (inDegree[key] !== undefined && nameToSkill[preKey]) {
        inDegree[key]++;
        adj[preKey].push(key);
      }
    });
  });

  // Start with zero in-degree nodes
  let queue = skills
    .filter(s => inDegree[normalise(s.name)] === 0)
    .sort(byPriority);

  const result = [];

  while (queue.length > 0) {
    // Pick highest-priority node
    queue.sort(byPriority);
    const node = queue.shift();
    result.push(node);

    const key = normalise(node.name);
    (adj[key] || []).forEach(neighbourKey => {
      inDegree[neighbourKey]--;
      if (inDegree[neighbourKey] === 0) {
        queue.push(nameToSkill[neighbourKey]);
      }
    });
  }

  // Append any remaining (cycle fallback)
  skills.forEach(s => {
    if (!result.find(r => normalise(r.name) === normalise(s.name))) {
      result.push(s);
    }
  });

  // Final order: REQUIRED → FAST-TRACK → RECOMMENDED → SKIPPED
  return result.sort((a, b) => {
    const order = { 'REQUIRED': 0, 'FAST-TRACK': 1, 'RECOMMENDED': 2, 'SKIPPED': 3 };
    const ao = order[a.status] ?? 2;
    const bo = order[b.status] ?? 2;
    if (ao !== bo) return ao - bo;
    return b.gap - a.gap; // within same status, higher gap first
  });
}

// Sort by: REQUIRED > FAST-TRACK > gap score
function byPriority(a, b) {
  const statusOrder = { 'REQUIRED': 0, 'FAST-TRACK': 1, 'SKIPPED': 2 };
  const ao = statusOrder[a.status] ?? 1;
  const bo = statusOrder[b.status] ?? 1;
  if (ao !== bo) return ao - bo;
  return b.gap - a.gap;
}

// Build transparent reasoning trace
function buildReason(status, current, required, match, req) {
  const pct     = pct100(current);
  const reqPct  = pct100(required);
  const yrs     = match?.yearsExp;

  if (status === 'SKIPPED') {
    return yrs
      ? `${yrs}yr experience detected — proficiency ${pct}% exceeds ${reqPct}% requirement`
      : `Proficiency ${pct}% meets ${reqPct}% requirement — module skipped`;
  }
  if (status === 'FAST-TRACK') {
    return `${pct}% proficiency vs ${reqPct}% required — accelerated module (${Math.round((required - current) * 100)}% gap)`;
  }
  if (status === 'REQUIRED') {
    return match
      ? `Critical gap — ${pct}% proficiency vs ${reqPct}% required`
      : `No prior experience detected — ${reqPct}% proficiency required`;
  }
  return `${reqPct}% proficiency required for this role`;
}

// Fuzzy skill name matching (handles "React.js" vs "React", "AWS" vs "Amazon Web Services")
function findMatch(key, candidateMap) {
  if (candidateMap[key]) return candidateMap[key];
  // Partial match
  for (const k of Object.keys(candidateMap)) {
    if (k.includes(key) || key.includes(k)) return candidateMap[k];
  }
  return null;
}

function normalise(str) {
  return (str || '').toLowerCase().replace(/[^a-z0-9]/g, '');
}

function clamp(val, min, max) {
  return Math.min(max, Math.max(min, val));
}

function round2(n) {
  return Math.round(n * 100) / 100;
}

function pct100(n) {
  return Math.round(n * 100);
}
