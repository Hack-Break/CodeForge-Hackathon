# NeuralPath — Project Structure

## Folder Layout

```
neuralpath/
├── index.html                      ← Entry point — links all CSS & JS
└── src/
    ├── css/
    │   ├── variables.css           ← Design tokens (CSS custom properties)
    │   ├── base.css                ← Reset, body, section, .section-tag, .section-h
    │   ├── animations.css          ← All @keyframes + .reveal scroll classes
    │   ├── cursor.css              ← Custom cursor dot & ring
    │   ├── buttons.css             ← .btn-primary, .btn-secondary, .btn-github
    │   ├── nav.css                 ← Fixed navigation bar
    │   ├── hero.css                ← Hero section, orbs, grid floor, floating cards, stats
    │   ├── responsive.css          ← All @media breakpoints (always last)
    │   └── components/
    │       ├── steps.css           ← "How It Works" step grid
    │       ├── flow.css            ← System data flow diagram
    │       ├── upload.css          ← Upload zones + pathway card + module rows
    │       ├── tech.css            ← Tech stack grid
    │       ├── metrics.css         ← Metrics / validation section
    │       ├── algo.css            ← Algorithm cards + code blocks
    │       └── cta-footer.css      ← CTA section + footer
    └── js/
        ├── cursor.js               ← Cursor tracking & hover scale
        ├── scrollReveal.js         ← IntersectionObserver for .reveal elements
        ├── skillBars.js            ← Skill/module progress bar animations
        ├── counters.js             ← Animated metric counter (% and ×)
        └── smoothScroll.js         ← Smooth anchor scroll behaviour

```

## How to Edit

| I want to change…            | Edit this file                          |
|------------------------------|-----------------------------------------|
| Colors / fonts / spacing     | `src/css/variables.css`                 |
| Navigation bar               | `src/css/nav.css`                       |
| Hero headline / description  | `index.html` + `src/css/hero.css`       |
| Step cards (How It Works)    | `index.html` + `src/css/components/steps.css` |
| Flow diagram nodes           | `index.html` + `src/css/components/flow.css`  |
| Upload mockup / modules      | `index.html` + `src/css/components/upload.css`|
| Tech stack items             | `index.html` + `src/css/components/tech.css`  |
| Metric numbers               | `index.html` + `src/css/components/metrics.css` |
| Algorithm code blocks        | `index.html` + `src/css/components/algo.css`  |
| CTA text / badges            | `index.html` + `src/css/components/cta-footer.css` |
| Cursor behaviour             | `src/js/cursor.js`                      |
| Scroll animations            | `src/js/scrollReveal.js`                |
| Skill bar animations         | `src/js/skillBars.js`                   |
| Metric counter animation     | `src/js/counters.js`                    |
| Mobile breakpoints           | `src/css/responsive.css`                |

## Running Locally

Just open `index.html` in a browser — no build step required.
For a proper dev server (hot reload):

```bash
npx serve .
# or
python3 -m http.server 8080
```
