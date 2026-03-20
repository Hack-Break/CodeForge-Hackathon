// ─── Animated Metric Counters ───
// Counts up .metric-big numbers when the metrics grid scrolls into view.

/**
 * Animates a single counter element from 0 up to `target`.
 * @param {HTMLElement} el     - The element whose textContent will be updated.
 * @param {number}      target - The final numeric value.
 * @param {string}      suffix - Unit suffix appended to the number (e.g. '%' or '×').
 */
function animateCounter(el, target, suffix) {
  let current = 0;
  const increment = target / 60;

  const timer = setInterval(() => {
    current += increment;
    if (current >= target) {
      el.textContent = target + suffix;
      clearInterval(timer);
      return;
    }
    el.textContent = Math.round(current) + suffix;
  }, 16);
}

const metricObserver = new IntersectionObserver(entries => {
  entries.forEach(e => {
    if (e.isIntersecting) {
      e.target.querySelectorAll('.metric-big').forEach(el => {
        const text = el.textContent.trim();
        if (text.endsWith('%'))      animateCounter(el, parseFloat(text), '%');
        else if (text.endsWith('×')) animateCounter(el, parseFloat(text), '×');
      });
      metricObserver.unobserve(e.target);
    }
  });
}, { threshold: 0.5 });

document.querySelectorAll('.metrics-grid').forEach(el => metricObserver.observe(el));
