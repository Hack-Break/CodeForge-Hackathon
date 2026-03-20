// ═══════════════════════════════════════════════════════════
//  DEMO.JS — NeuralPath AI Demo
//  Reads resume (PDF/DOCX/TXT) + JD, calls /api/analyze,
//  renders the generated learning pathway.
// ═══════════════════════════════════════════════════════════

(function initDemo() {

  // ── DOM refs ─────────────────────────────────────────────
  const resumeZone    = document.getElementById('resume-zone');
  const resumeInput   = document.getElementById('resume-input');
  const resumeFile    = document.getElementById('resume-file');
  const resumeFilename= document.getElementById('resume-filename');
  const jdText        = document.getElementById('jd-text');
  const analyseBtn    = document.getElementById('analyse-btn');
  const demoHint      = document.getElementById('demo-hint');

  const uploadSection = document.getElementById('upload-section');
  const loadingSection= document.getElementById('loading-section');
  const resultsSection= document.getElementById('results-section');
  const errorBox      = document.getElementById('error-box');
  const errorMsg      = document.getElementById('error-msg');
  const errorResetBtn = document.getElementById('error-reset-btn');
  const resetBtn      = document.getElementById('reset-btn');

  const loadingLabel  = document.getElementById('loading-label');
  const loadingSteps  = ['ls-1','ls-2','ls-3','ls-4'];

  const resultsTitle  = document.getElementById('results-title');
  const resultsMeta   = document.getElementById('results-meta');
  const pathwayList   = document.getElementById('pathway-list');
  const resultsSummary= document.getElementById('results-summary');

  const sumTotal    = document.getElementById('sum-total');
  const sumModules  = document.getElementById('sum-modules');
  const sumSkipped  = document.getElementById('sum-skipped');
  const sumCritical = document.getElementById('sum-critical');

  // ── State ────────────────────────────────────────────────
  let resumeText = '';

  // ── Resume zone: click to open file picker ───────────────
  resumeZone.addEventListener('click', () => resumeInput.click());

  resumeInput.addEventListener('change', e => {
    const file = e.target.files[0];
    if (file) handleResumeFile(file);
  });

  // ── Resume zone: drag & drop ─────────────────────────────
  resumeZone.addEventListener('dragover', e => {
    e.preventDefault();
    resumeZone.classList.add('drag-over');
  });

  resumeZone.addEventListener('dragleave', () => {
    resumeZone.classList.remove('drag-over');
  });

  resumeZone.addEventListener('drop', e => {
    e.preventDefault();
    resumeZone.classList.remove('drag-over');
    const file = e.dataTransfer.files[0];
    if (file) handleResumeFile(file);
  });

  // ── JD text input → enable button check ─────────────────
  jdText.addEventListener('input', checkReady);

  // ── Parse resume file ────────────────────────────────────
  async function handleResumeFile(file) {
    const ext = file.name.split('.').pop().toLowerCase();
    const allowed = ['pdf','docx','txt'];

    if (!allowed.includes(ext)) {
      showError('Unsupported file type. Please upload a PDF, DOCX, or TXT file.');
      return;
    }

    resumeZone.classList.add('has-file');
    resumeFile.hidden = false;
    resumeFilename.textContent = file.name;

    try {
      if (ext === 'txt') {
        resumeText = await readTxt(file);
      } else if (ext === 'pdf') {
        resumeText = await readPdf(file);
      } else if (ext === 'docx') {
        resumeText = await readDocx(file);
      }
      checkReady();
    } catch (err) {
      console.error('File read error:', err);
      showError('Could not read the file. Try a different format.');
    }
  }

  // ── Text file reader ─────────────────────────────────────
  function readTxt(file) {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload  = e => resolve(e.target.result);
      reader.onerror = reject;
      reader.readAsText(file);
    });
  }

  // ── PDF reader via pdf.js ────────────────────────────────
  async function readPdf(file) {
    const pdfjsLib = window['pdfjs-dist/build/pdf'];
    pdfjsLib.GlobalWorkerOptions.workerSrc =
      'https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/pdf.worker.min.js';

    const arrayBuffer = await file.arrayBuffer();
    const pdf = await pdfjsLib.getDocument({ data: arrayBuffer }).promise;
    let fullText = '';

    for (let i = 1; i <= pdf.numPages; i++) {
      const page    = await pdf.getPage(i);
      const content = await page.getTextContent();
      fullText += content.items.map(item => item.str).join(' ') + '\n';
    }
    return fullText;
  }

  // ── DOCX reader via mammoth.js ───────────────────────────
  async function readDocx(file) {
    const arrayBuffer = await file.arrayBuffer();
    const result = await mammoth.extractRawText({ arrayBuffer });
    return result.value;
  }

  // ── Enable analyse button when both inputs are ready ─────
  function checkReady() {
    const hasResume = resumeText.trim().length > 50;
    const hasJD     = jdText.value.trim().length > 30;
    analyseBtn.disabled = !(hasResume && hasJD);
    demoHint.textContent = !hasResume
      ? 'Upload your resume to begin.'
      : !hasJD
        ? 'Add a job description to continue.'
        : 'Ready! Click Generate Pathway.';
  }

  // ── Analyse button ───────────────────────────────────────
  analyseBtn.addEventListener('click', runAnalysis);

  async function runAnalysis() {
    showSection('loading');
    animateLoadingSteps();

    try {
      const payload = {
        resumeText: resumeText.slice(0, 6000), // trim for token limit
        jobDescription: jdText.value.trim().slice(0, 3000)
      };

      const res = await fetch('/api/analyze', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });

      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.error || `Server error: ${res.status}`);
      }

      const data = await res.json();
      renderResults(data);

    } catch (err) {
      console.error('Analysis error:', err);
      showError(err.message || 'Analysis failed. Please try again.');
    }
  }

  // ── Loading step animation ───────────────────────────────
  function animateLoadingSteps() {
    const messages = [
      'Reading your resume…',
      'Analysing job requirements…',
      'Detecting skill gaps…',
      'Building your pathway…'
    ];
    let step = 0;

    loadingSteps.forEach(id => {
      const el = document.getElementById(id);
      el.className = 'loading-step';
    });

    const interval = setInterval(() => {
      if (step > 0) {
        document.getElementById(loadingSteps[step - 1]).classList.add('done');
        document.getElementById(loadingSteps[step - 1]).classList.remove('active');
      }
      if (step < loadingSteps.length) {
        document.getElementById(loadingSteps[step]).classList.add('active');
        loadingLabel.textContent = messages[step];
        step++;
      } else {
        clearInterval(interval);
      }
    }, 900);
  }

  // ── Render results ───────────────────────────────────────
  function renderResults(data) {
    showSection('results');

    resultsTitle.textContent = 'Generated Pathway — ' + (data.jobTitle || 'Your Role');

    const modules  = data.modules || [];
    const skipped  = modules.filter(m => m.status === 'SKIPPED').length;
    const required = modules.filter(m => m.status === 'REQUIRED').length;
    const totalHrs = modules.reduce((sum, m) => sum + (m.hours || 0), 0);

    resultsMeta.textContent =
      `Based on resume analysis · ${modules.length} modules · ${skipped} skipped · Est. ${totalHrs}h total`;

    // Render each module card
    pathwayList.innerHTML = '';
    modules.forEach((mod, i) => {
      const item = buildPathwayItem(mod, i);
      pathwayList.appendChild(item);
    });

    // Trigger skill bar animations after paint
    setTimeout(() => {
      document.querySelectorAll('.skill-fill').forEach(bar => {
        bar.style.width = bar.dataset.width + '%';
      });
    }, 100);

    // Summary
    sumTotal.textContent    = totalHrs + 'h';
    sumModules.textContent  = modules.length - skipped;
    sumSkipped.textContent  = skipped;
    sumCritical.textContent = required;
    resultsSummary.hidden   = false;
  }

  // ── Build a single pathway item card ─────────────────────
  function buildPathwayItem(mod, index) {
    const status = (mod.status || 'RECOMMENDED').toLowerCase().replace(' ', '-');
    // status: skipped | required | fast-track | recommended

    const badgeLabel = status === 'skipped' ? '✓'
      : status === 'required' ? '!'
      : status === 'fast-track' ? '→'
      : String(index + 1);

    const proficiency = Math.min(100, Math.max(0, mod.proficiency || 0));
    const hours = mod.hours || 0;

    const item = document.createElement('div');
    item.className = `pathway-item pathway-item--${status}`;
    item.style.animationDelay = (index * 0.07) + 's';

    item.innerHTML = `
      <div class="pathway-item__badge badge--${status}">${badgeLabel}</div>
      <div class="pathway-item__body">
        <div class="pathway-item__name">${escHtml(mod.name || 'Module')}</div>
        <div class="pathway-item__meta meta--${status}">
          <strong>${escHtml(mod.status || 'RECOMMENDED')}</strong>
          ${escHtml(mod.reason || '')}
        </div>
      </div>
      <div class="pathway-item__right">
        <span class="pathway-item__hours">${hours}h</span>
        <div class="skill-bar">
          <div class="skill-fill fill--${status}" data-width="${proficiency}" style="width:0"></div>
        </div>
      </div>
    `;
    return item;
  }

  // ── Show/hide sections ───────────────────────────────────
  function showSection(name) {
    uploadSection.hidden  = name !== 'upload';
    loadingSection.hidden = name !== 'loading';
    resultsSection.hidden = name !== 'results';
    errorBox.hidden       = name !== 'error';
  }

  function showError(msg) {
    errorMsg.textContent = msg;
    showSection('error');
  }

  // ── Reset ────────────────────────────────────────────────
  function resetDemo() {
    resumeText = '';
    resumeInput.value = '';
    jdText.value = '';
    resumeZone.classList.remove('has-file', 'drag-over');
    resumeFile.hidden = true;
    analyseBtn.disabled = true;
    demoHint.textContent = 'Upload your resume and add a job description to begin.';
    pathwayList.innerHTML = '';
    resultsSummary.hidden = true;
    showSection('upload');
  }

  resetBtn.addEventListener('click', resetDemo);
  errorResetBtn.addEventListener('click', resetDemo);

  // ── Utility ──────────────────────────────────────────────
  function escHtml(str) {
    return String(str)
      .replace(/&/g,'&amp;')
      .replace(/</g,'&lt;')
      .replace(/>/g,'&gt;')
      .replace(/"/g,'&quot;');
  }

})();
