// ═══════════════════════════════════════════════════════════
//  CURSOR.JS — Custom Cursor · Trail · Ripple · Label
//  Styles live in css/cursor.css
//  Uses .cursor / .cursor-ring classes to match your CSS.
// ═══════════════════════════════════════════════════════════

(function initCursor() {

  // ── 1. Create DOM elements ──────────────────────────────
  const dot   = document.createElement('div');
  const ring  = document.createElement('div');
  const label = document.createElement('div');

  dot.className   = 'cursor';
  ring.className  = 'cursor-ring';
  label.id        = 'cursor-label';   // label uses ID (unique, no conflict)

  document.body.append(dot, ring, label);

  // ── 2. State ────────────────────────────────────────────
  let mx = window.innerWidth  / 2;
  let my = window.innerHeight / 2;
  let rx = mx, ry = my;   // ring position (lagged)

  // Trail colours — lime-adjacent palette to match --lime
  const TRAIL_COLORS = ['#bef264', '#a3e635', '#86efac', '#fde047', '#6ee7b7'];
  let trailIdx = 0;
  let lastTrailTime = 0;

  // ── 3. Mouse move ────────────────────────────────────────
  document.addEventListener('mousemove', e => {
    mx = e.clientX;
    my = e.clientY;

    // Dot + label snap instantly
    dot.style.left   = mx + 'px';
    dot.style.top    = my + 'px';
    label.style.left = mx + 'px';
    label.style.top  = my + 'px';

    spawnTrail(mx, my);
  });

  // ── 4. Trail particle ────────────────────────────────────
  // Throttled to ~33fps to avoid flooding the DOM
  function spawnTrail(x, y) {
    const now = performance.now();
    if (now - lastTrailTime < 30) return;
    lastTrailTime = now;

    const p = document.createElement('div');
    p.className = 'cursor-trail';

    const size = 3 + Math.random() * 4;
    p.style.left       = (x + (Math.random() - 0.5) * 10) + 'px';
    p.style.top        = (y + (Math.random() - 0.5) * 10) + 'px';
    p.style.width      = size + 'px';
    p.style.height     = size + 'px';
    p.style.background = TRAIL_COLORS[trailIdx % TRAIL_COLORS.length];
    trailIdx++;

    document.body.appendChild(p);
    setTimeout(() => p.remove(), 600);
  }

  // ── 5. Click ripple ──────────────────────────────────────
  function spawnRipple(x, y, color) {
    const r = document.createElement('div');
    r.className = 'cursor-ripple';

    const size = 60 + Math.random() * 30;
    r.style.left   = x + 'px';
    r.style.top    = y + 'px';
    r.style.width  = size + 'px';
    r.style.height = size + 'px';
    r.style.border = '2px solid ' + color;

    document.body.appendChild(r);
    setTimeout(() => r.remove(), 560);
  }

  document.addEventListener('click', e => {
    spawnRipple(e.clientX, e.clientY, TRAIL_COLORS[trailIdx % TRAIL_COLORS.length]);

    // Dot pulse on click
    dot.style.transform = 'translate(-50%,-50%) scale(2.5)';
    setTimeout(() => dot.style.transform = 'translate(-50%,-50%) scale(1)', 150);
  });

  // ── 6. Ring animation loop (rAF) ─────────────────────────
  // Ring lags behind cursor with 10% lerp per frame
  function rafLoop() {
    rx += (mx - rx) * 0.10;
    ry += (my - ry) * 0.10;
    ring.style.left = rx + 'px';
    ring.style.top  = ry + 'px';
    requestAnimationFrame(rafLoop);
  }
  rafLoop();

  // ── 7. Hover states ──────────────────────────────────────
  const INTERACTIVE = 'button, a, .flow-box, .step, .upload-zone, .tech-item, [data-cursor]';

  function getCursorConfig(el) {
    if (el.dataset.cursor) return { label: el.dataset.cursor, ringSize: 64 };
    const tag = el.tagName.toLowerCase();
    if (tag === 'a')      return { label: 'VISIT', ringSize: 56 };
    if (tag === 'button') return { label: 'CLICK', ringSize: 56 };
    return                       { label: '',       ringSize: 60 };
  }

  document.querySelectorAll(INTERACTIVE).forEach(el => {
    el.addEventListener('mouseenter', () => {
      const cfg = getCursorConfig(el);

      dot.style.width  = '20px';
      dot.style.height = '20px';

      ring.style.width  = cfg.ringSize + 'px';
      ring.style.height = cfg.ringSize + 'px';
      ring.style.opacity = '0.8';

      if (cfg.label) {
        label.textContent = cfg.label;
        label.classList.add('visible');
      }
    });

    el.addEventListener('mouseleave', () => {
      dot.style.width  = '12px';
      dot.style.height = '12px';

      ring.style.width   = '40px';
      ring.style.height  = '40px';
      ring.style.opacity = '0.5';

      label.classList.remove('visible');
    });
  });

  // ── 8. Press feel ────────────────────────────────────────
  document.addEventListener('mousedown', () => {
    dot.style.width  = '7px';
    dot.style.height = '7px';
    ring.style.width  = '28px';
    ring.style.height = '28px';
  });

  document.addEventListener('mouseup', () => {
    dot.style.width  = '12px';
    dot.style.height = '12px';
    ring.style.width  = '40px';
    ring.style.height = '40px';
  });

  // ── 9. Hide / show when pointer leaves window ────────────
  document.addEventListener('mouseleave', () => {
    dot.style.opacity  = '0';
    ring.style.opacity = '0';
  });

  document.addEventListener('mouseenter', () => {
    dot.style.opacity  = '1';
    ring.style.opacity = '0.5';
  });

})();