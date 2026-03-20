// ═══════════════════════════════════════════════════════════
//  UPLOAD.JS — NeuralPath
//  Handles resume + JD upload on index.html,
//  calls /api/analyze, replaces static pathway card.
// ═══════════════════════════════════════════════════════════

(function initUpload() {

  // ── DOM refs ─────────────────────────────────────────────
  const resumeZone = document.getElementById('resume-zone');
  const resumeInput = document.getElementById('resume-input');
  const jdZone = document.getElementById('jd-zone');
  const jdInput = document.getElementById('jd-input');
  const jdTextarea = document.getElementById('jd-textarea');
  const generateBtn = document.getElementById('generate-btn');
  const pathwayCard = document.querySelector('.pathway-card');

  if (!resumeZone || !jdZone || !generateBtn || !pathwayCard) {
    console.warn('Upload.js: elements not found yet.');
    return;
  }

  // ── State ────────────────────────────────────────────────
  let resumeText = '';

  // ── Resume zone: click ───────────────────────────────────
  resumeZone.addEventListener('click', () => resumeInput.click());

  resumeInput.addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (file) handleResumeFile(file);
  });

  // ── Resume zone: drag & drop ─────────────────────────────
  resumeZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    resumeZone.classList.add('drag-over');
  });
  resumeZone.addEventListener('dragleave', () => {
    resumeZone.classList.remove('drag-over');
  });
  resumeZone.addEventListener('drop', (e) => {
    e.preventDefault();
    resumeZone.classList.remove('drag-over');
    const file = e.dataTransfer.files[0];
    if (file) handleResumeFile(file);
  });

  // ── JD textarea: enable button on input ──────────────────
  jdTextarea.addEventListener('input', checkReady);
  jdTextarea.addEventListener('paste', () => setTimeout(checkReady, 50));

  // ── JD file upload ───────────────────────────────────────
  jdZone.addEventListener('click', (e) => {
    // Only trigger file pick if not clicking textarea
    if (e.target === jdTextarea) return;
    if (jdTextarea.value.trim().length === 0) jdInput.click();
  });

  jdInput.addEventListener('change', async (e) => {
    const file = e.target.files[0];
    if (!file) return;
    const ext = file.name.split('.').pop().toLowerCase();
    try {
      let text = '';
      if (ext === 'txt') text = await readTxt(file);
      else if (ext === 'pdf') text = await readPdf(file);
      else if (ext === 'docx') text = await readDocx(file);
      if (text) {
        jdTextarea.value = text.slice(0, 3000);
        checkReady();
      }
    } catch (err) {
      console.warn('JD file read error:', err);
    }
  });

  // ── Handle resume file ───────────────────────────────────
  async function handleResumeFile(file) {
    const ext = file.name.split('.').pop().toLowerCase();
    if (!['pdf', 'docx', 'txt'].includes(ext)) {
      alert('Please upload a PDF, DOCX, or TXT file.');
      return;
    }

    // Show uploaded state
    resumeZone.classList.add('has-file');
    resumeZone.querySelector('.upload-icon').textContent = '✅';
    resumeZone.querySelector('h4').textContent = file.name;
    resumeZone.querySelector('p').textContent = 'File ready — add a job description to continue.';

    try {
      if (ext === 'txt') resumeText = await readTxt(file);
      else if (ext === 'pdf') resumeText = await readPdf(file);
      else if (ext === 'docx') resumeText = await readDocx(file);
    } catch (err) {
      console.warn('Resume read error:', err);
      resumeText = `Resume: ${file.name}`;
    }

    checkReady();
  }

  // ── File readers ─────────────────────────────────────────
  function readTxt(file) {
    return new Promise((res, rej) => {
      const r = new FileReader();
      r.onload = e => res(e.target.result);
      r.onerror = rej;
      r.readAsText(file);
    });
  }

  async function readPdf(file) {
    const lib = window.pdfjsLib;
    if (!lib) return `PDF: ${file.name}`;
    lib.GlobalWorkerOptions.workerSrc =
      'https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/pdf.worker.min.js';
    try {
      const buf = await file.arrayBuffer();
      const pdf = await lib.getDocument({ data: buf }).promise;
      let text = '';
      for (let i = 1; i <= pdf.numPages; i++) {
        const page = await pdf.getPage(i);
        const content = await page.getTextContent();
        text += content.items.map(i => i.str).join(' ') + '\n';
      }
      return text || `PDF: ${file.name}`;
    } catch (e) {
      return `PDF: ${file.name}`;
    }
  }

  async function readDocx(file) {
    if (!window.mammoth) return `DOCX: ${file.name}`;
    try {
      const buf = await file.arrayBuffer();
      const result = await mammoth.extractRawText({ arrayBuffer: buf });
      return result.value || `DOCX: ${file.name}`;
    } catch (e) {
      return `DOCX: ${file.name}`;
    }
  }

  // ── Enable generate button ───────────────────────────────
  function checkReady() {
    const hasResume = resumeText.trim().length > 10;
    const hasJD = jdTextarea.value.trim().length > 20;
    generateBtn.disabled = !(hasResume && hasJD);
    generateBtn.textContent = hasResume && hasJD
      ? 'Generate Pathway ↗'
      : !hasResume
        ? 'Upload your resume first…'
        : 'Add a job description…';
  }

  // ── Generate pathway ─────────────────────────────────────
  generateBtn.addEventListener('click', async () => {
    setLoading(true);

    try {
      const res = await fetch('/api/analyze', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          resumeText: resumeText.slice(0, 6000),
          jobDescription: jdTextarea.value.trim().slice(0, 3000)
        })
      });
      const text = await res.text();
      if (!text) throw new Error('Empty response from server. Check Vercel function logs.');
      const data = JSON.parse(text);
      if (!res.ok) throw new Error(data.error || `Error ${res.status}`);
      renderPathway(data);

    } catch (err) {
      console.error('Generate error:', err);
      setLoading(false);
      pathwayCard.innerHTML = `
        <div style="text-align:center;padding:40px 20px">
          <div style="font-size:2rem;margin-bottom:12px">⚠️</div>
          <p style="color:rgba(255,255,255,0.5);margin-bottom:16px">${esc(err.message)}</p>
          <button onclick="location.reload()" style="background:rgba(255,255,255,0.08);border:1px solid rgba(255,255,255,0.12);color:rgba(255,255,255,0.6);padding:8px 20px;border-radius:8px;font-size:0.85rem">Try Again</button>
        </div>`;
    }
  });

  // ── Set loading state on pathway card ────────────────────
  function setLoading(on) {
    generateBtn.disabled = on;
    if (on) {
      generateBtn.textContent = 'Analysing…';
      pathwayCard.innerHTML = `
        <div style="display:flex;flex-direction:column;align-items:center;gap:16px;padding:48px 20px">
          <div class="pathway-loading-orb"></div>
          <p style="color:rgba(255,255,255,0.5);font-size:0.9rem;letter-spacing:.04em">Building your personalised pathway…</p>
        </div>`;

      // Inject loading orb style once
      if (!document.getElementById('pathway-orb-style')) {
        const s = document.createElement('style');
        s.id = 'pathway-orb-style';
        s.textContent = `
          .pathway-loading-orb {
            width:48px;height:48px;border-radius:50%;
            border:2px solid rgba(190,242,100,0.15);
            border-top-color:var(--lime);
            animation:spin .9s linear infinite;
          }
          @keyframes spin { to { transform:rotate(360deg); } }
        `;
        document.head.appendChild(s);
      }
    } else {
      generateBtn.disabled = false;
      generateBtn.textContent = 'Generate Pathway ↗';
    }
  }

  // ── Render results into pathway card ─────────────────────
  function renderPathway(data) {
    const modules = Array.isArray(data.modules) ? data.modules : [];
    const skipped = modules.filter(m => m.status === 'SKIPPED').length;
    const totalHrs = modules.reduce((s, m) => s + (Number(m.hours) || 0), 0);

    pathwayCard.innerHTML = `
      <h4>Generated Pathway — ${esc(data.jobTitle || 'Your Role')}</h4>
      <p>Based on resume analysis · ${modules.length} modules selected · ${skipped} skipped · Est. ${totalHrs}h total</p>
      <div class="pathway-modules" id="pathway-modules-list"></div>
      <div class="pathway-summary">
        <div class="psumm-item"><span class="psumm-num">${totalHrs}h</span><span class="psumm-lbl">Total</span></div>
        <div class="psumm-item"><span class="psumm-num">${modules.length - skipped}</span><span class="psumm-lbl">Modules</span></div>
        <div class="psumm-item"><span class="psumm-num">${skipped}</span><span class="psumm-lbl">Skipped</span></div>
        <div class="psumm-item"><span class="psumm-num">${modules.filter(m => m.status === 'REQUIRED').length}</span><span class="psumm-lbl">Critical</span></div>
      </div>`;

    const list = document.getElementById('pathway-modules-list');
    modules.forEach((mod, i) => {
      const row = buildModuleRow(mod, i);
      list.appendChild(row);
    });

    // Animate bars
    setTimeout(() => {
      list.querySelectorAll('.mod-fill[data-w]').forEach(bar => {
        bar.style.width = bar.dataset.w + '%';
      });
    }, 100);

    // Scroll to results
    pathwayCard.scrollIntoView({ behavior: 'smooth', block: 'start' });

    // Inject summary styles once
    if (!document.getElementById('psumm-style')) {
      const s = document.createElement('style');
      s.id = 'psumm-style';
      s.textContent = `
        .pathway-summary {
          display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin-top:20px;
        }
        .psumm-item {
          background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.07);
          border-radius:10px;padding:14px;text-align:center;
          display:flex;flex-direction:column;gap:4px;
        }
        .psumm-num { font-size:1.4rem;font-weight:700;color:var(--lime);letter-spacing:-0.02em; }
        .psumm-lbl { font-size:0.7rem;color:rgba(255,255,255,0.3);text-transform:uppercase;letter-spacing:.08em; }
        @media(max-width:480px){ .pathway-summary{grid-template-columns:repeat(2,1fr);} }
      `;
      document.head.appendChild(s);
    }

    setLoading(false);
  }

  // ── Build a module row ───────────────────────────────────
  function buildModuleRow(mod, index) {
    const raw = (mod.status || 'RECOMMENDED').toUpperCase();
    const status = raw === 'FAST-TRACK' ? 'fast' : raw.toLowerCase().replace('required', 'req').replace('recommended', 'rec').replace('skipped', 'skip');

    const badge = status === 'skip' ? '✓'
      : status === 'req' ? String(index + 1)
        : status === 'fast' ? '→'
          : String(index + 1);

    const barColor = status === 'skip' ? 'var(--lime)'
      : status === 'req' ? '#a99fff'
        : status === 'fast' ? '#ffc83d'
          : 'var(--lime)';

    const pct = Math.min(100, Math.max(0, Number(mod.proficiency) || 0));
    const hours = Number(mod.hours) || 0;

    const row = document.createElement('div');
    row.className = 'module-row';
    row.style.animationDelay = (index * 0.06) + 's';
    row.innerHTML = `
      <div class="mod-status ${status}">${badge}</div>
      <div class="mod-content">
        <div class="mod-name">${esc(mod.name || 'Module')}</div>
        <div class="mod-meta">${esc(raw)} — ${esc(mod.reason || '')}</div>
      </div>
      <div class="mod-bar-wrap">
        <div class="mod-time">${hours}h</div>
        <div class="mod-bar">
          <div class="mod-fill" data-w="${pct}" style="width:0;background:${barColor};transition:width 1s cubic-bezier(.4,0,.2,1)"></div>
        </div>
      </div>`;
    return row;
  }

  // ── HTML escape ──────────────────────────────────────────
  function esc(str) {
    return String(str)
      .replace(/&/g, '&amp;').replace(/</g, '&lt;')
      .replace(/>/g, '&gt;').replace(/"/g, '&quot;');
  }

})();
