/**
 * frontend/js/app.js — Lumarium Main Application Controller
 * Wires together routing, user input, arbiter controls, and API calls.
 */

document.addEventListener('DOMContentLoaded', async () => {
  // 1. Navigation Routing
  const navLinks = document.querySelectorAll('.nav-link');
  const viewPanels = document.querySelectorAll('.view-panel');

  function switchView(viewName) {
    navLinks.forEach(link => {
      const isActive = link.dataset.view === viewName;
      link.classList.toggle('active', isActive);
      link.setAttribute('aria-selected', isActive ? 'true' : 'false');
    });

    viewPanels.forEach(panel => {
      const isTarget = panel.id === `view-${viewName}`;
      panel.classList.toggle('active-view', isTarget);
    });

    // Lazy load or refresh views
    if (viewName === 'practice') {
      const currentSentence = document.getElementById('sentence-input')?.value || '';
      Practice.init(currentSentence);
    }
  }

  navLinks.forEach(link => {
    link.addEventListener('click', () => {
      switchView(link.dataset.view);
    });
  });

  const brandHome = document.getElementById('brand-home-btn');
  if (brandHome) {
    brandHome.addEventListener('click', () => switchView('analyze'));
  }

  // 2. Form Submission for Analysis
  const analyzeForm = document.getElementById('analyze-form');
  const sentenceInput = document.getElementById('sentence-input');

  if (analyzeForm && sentenceInput) {
    analyzeForm.addEventListener('submit', async (e) => {
      e.preventDefault();
      const text = sentenceInput.value.trim();
      if (!text) return;

      const threshold = parseFloat(document.getElementById('slider-threshold')?.value || '0.85');
      const fallbackRadio = document.querySelector('input[name="observatory-fallback"]:checked');
      const fallback = fallbackRadio ? fallbackRadio.value : 'hmm';
      const ruleOverridable = document.getElementById('checkbox-overrule')?.checked ?? true;

      await Analyzer.runAnalysis(text, {
        threshold,
        fallback,
        rule_overridable: ruleOverridable,
        use_transformer: true,
      });
    });
  }

  // 3. Exemplar Chips
  const exemplarChips = document.querySelectorAll('.chip-btn');
  exemplarChips.forEach(chip => {
    chip.addEventListener('click', () => {
      const text = chip.dataset.sentence;
      if (text && sentenceInput) {
        sentenceInput.value = text;
        switchView('analyze');
        analyzeForm.dispatchEvent(new Event('submit'));
      }
    });
  });

  // 4. Practice Chamber Events
  const quizSubmitBtn = document.getElementById('quiz-submit-btn');
  const quizNextBtn = document.getElementById('quiz-next-btn');
  const refreshQuizBtn = document.getElementById('refresh-quiz-btn');

  if (quizSubmitBtn) {
    quizSubmitBtn.addEventListener('click', () => Practice.submitAnswer());
  }

  if (quizNextBtn) {
    quizNextBtn.addEventListener('click', () => Practice.nextQuestion());
  }

  if (refreshQuizBtn) {
    refreshQuizBtn.addEventListener('click', () => {
      const currentSentence = sentenceInput?.value || '';
      Practice.loadQuestions(currentSentence);
    });
  }

  // 5. NLP Observatory Controls
  const sliderThresh = document.getElementById('slider-threshold');
  const labelThresh = document.getElementById('val-threshold-label');
  const applyArbiterBtn = document.getElementById('apply-arbiter-btn');

  if (sliderThresh && labelThresh) {
    sliderThresh.addEventListener('input', () => {
      labelThresh.textContent = parseFloat(sliderThresh.value).toFixed(2);
    });
  }

  if (applyArbiterBtn) {
    applyArbiterBtn.addEventListener('click', async () => {
      const text = sentenceInput?.value || 'Maine kal school mein football khela.';
      const threshold = parseFloat(sliderThresh?.value || '0.85');
      const fallbackRadio = document.querySelector('input[name="observatory-fallback"]:checked');
      const fallback = fallbackRadio ? fallbackRadio.value : 'hmm';
      const ruleOverridable = document.getElementById('checkbox-overrule')?.checked ?? true;

      applyArbiterBtn.disabled = true;
      applyArbiterBtn.textContent = 'Applying…';

      await Analyzer.runAnalysis(text, {
        threshold,
        fallback,
        rule_overridable: ruleOverridable,
        use_transformer: true,
      });

      applyArbiterBtn.disabled = false;
      applyArbiterBtn.textContent = 'Apply & Re-analyze ✦';
    });
  }

  // 6. Initial Application Bootstrapping
  await Components.checkSystemHealth();
  await Components.initCompendium();

  // Run initial analysis on default sentence
  if (sentenceInput && sentenceInput.value.trim()) {
    await Analyzer.runAnalysis(sentenceInput.value.trim());
  }
});
