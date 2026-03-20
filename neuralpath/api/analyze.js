// ═══════════════════════════════════════════════════════════
//  api/analyze.js — Vercel Serverless Function
//  Proxies requests to Groq API.
//  Your GROQ_API_KEY stays safe as an environment variable —
//  it is never exposed to the browser.
// ═══════════════════════════════════════════════════════════

export default async function handler(req, res) {

  // ── Only allow POST ──────────────────────────────────────
  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Method not allowed' });
  }

  // ── Read API key from environment ────────────────────────
  const GROQ_API_KEY = process.env.GROQ_API_KEY;
  if (!GROQ_API_KEY) {
    return res.status(500).json({ error: 'API key not configured on server.' });
  }

  // ── Parse body ───────────────────────────────────────────
  const { resumeText, jobDescription } = req.body;

  if (!resumeText || !jobDescription) {
    return res.status(400).json({ error: 'Missing resumeText or jobDescription.' });
  }

  // ── Build prompt ─────────────────────────────────────────
  const prompt = `
You are an expert career coach and skill gap analyser.

Given a candidate's resume and a job description, generate a structured learning pathway.

## Resume:
${resumeText}

## Job Description:
${jobDescription}

## Instructions:
Analyse the resume against the job description and return a JSON object (no markdown, no backticks, raw JSON only) with this exact structure:

{
  "jobTitle": "Extracted job title from the JD",
  "modules": [
    {
      "name": "Skill or topic name",
      "status": "SKIPPED" | "REQUIRED" | "FAST-TRACK" | "RECOMMENDED",
      "reason": "One short sentence explaining why (e.g. '4yr proficiency detected in resume' or 'Critical gap: 20% proficiency vs 85% required')",
      "hours": <number — estimated learning hours, 0 if skipped>,
      "proficiency": <number 0-100 — candidate's current proficiency>
    }
  ]
}

Status definitions:
- SKIPPED: Candidate already has strong proficiency (>80%). Set hours to 0.
- REQUIRED: Critical gap — role requires this, candidate proficiency is low (<40%).
- FAST-TRACK: Candidate has partial knowledge (40-80%), needs acceleration.
- RECOMMENDED: Nice-to-have skill, not critical but beneficial.

Return 6-10 modules. Order by priority: REQUIRED first, then FAST-TRACK, then RECOMMENDED, then SKIPPED last.
Return raw JSON only. No explanation, no markdown.
`.trim();

  // ── Call Groq API ────────────────────────────────────────
  try {
    const groqRes = await fetch('https://api.groq.com/openai/v1/chat/completions', {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${GROQ_API_KEY}`,
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        model: 'llama-3.3-70b-versatile',
        temperature: 0.3,
        max_tokens: 1500,
        messages: [
          { role: 'user', content: prompt }
        ]
      })
    });

    if (!groqRes.ok) {
      const errBody = await groqRes.text();
      console.error('Groq error:', errBody);
      return res.status(502).json({ error: 'AI service error. Please try again.' });
    }

    const groqData = await groqRes.json();
    const rawText  = groqData.choices?.[0]?.message?.content?.trim();

    if (!rawText) {
      return res.status(502).json({ error: 'Empty response from AI.' });
    }

    // ── Parse JSON from model response ────────────────────
    let parsed;
    try {
      // Strip any accidental markdown fences
      const clean = rawText.replace(/```json|```/g, '').trim();
      parsed = JSON.parse(clean);
    } catch (parseErr) {
      console.error('JSON parse error:', parseErr, '\nRaw:', rawText);
      return res.status(502).json({ error: 'Could not parse AI response. Please try again.' });
    }

    // ── Validate structure ────────────────────────────────
    if (!parsed.modules || !Array.isArray(parsed.modules)) {
      return res.status(502).json({ error: 'Unexpected AI response format.' });
    }

    return res.status(200).json(parsed);

  } catch (err) {
    console.error('Fetch error:', err);
    return res.status(500).json({ error: 'Internal server error.' });
  }
}
