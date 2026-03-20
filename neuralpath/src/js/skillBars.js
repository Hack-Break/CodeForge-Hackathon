// ─── Skill Bar Animations ───
// Animates .mod-fill and .skill-fill progress bars when they scroll into view.

const skillObserver = new IntersectionObserver(entries => {
  entries.forEach(e => {
    if (e.isIntersecting) {
      e.target.querySelectorAll('.mod-fill, .skill-fill').forEach(bar => {
        const targetWidth = bar.style.width;
        bar.style.width = '0';
        setTimeout(() => {
          bar.style.transition = 'width 1s cubic-bezier(0.4, 0, 0.2, 1)';
          bar.style.width = targetWidth;
        }, 100);
      });
    }
  });
}, { threshold: 0.3 });

document.querySelectorAll('.pathway-card, .hcard-1').forEach(el => skillObserver.observe(el));
