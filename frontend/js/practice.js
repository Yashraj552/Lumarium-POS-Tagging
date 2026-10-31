/**
 * frontend/js/practice.js — Practice Chamber Quiz Engine for Lumarium
 * Handles fetching dynamic questions, tracking score, and giving immediate feedback.
 */

const Practice = {
  questions: [],
  currentIndex: 0,
  score: 0,
  totalAnswered: 0,
  selectedAnswer: null,

  async init(sentence) {
    await this.loadQuestions(sentence);
  },

  async loadQuestions(sentence) {
    const titleEl = document.getElementById('question-title-text');
    if (titleEl) titleEl.textContent = 'Preparing customized questions…';

    try {
      const res = await fetch('/api/practice', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: sentence || '' }),
      });

      if (!res.ok) throw new Error('Failed to load practice questions');
      const data = await res.json();
      this.questions = data.questions || [];
      this.currentIndex = 0;
      this.renderCurrentQuestion();
    } catch (err) {
      if (titleEl) titleEl.textContent = `Could not load practice questions: ${err.message}`;
    }
  },

  renderCurrentQuestion() {
    const qBadge = document.getElementById('question-category-badge');
    const qTitle = document.getElementById('question-title-text');
    const optionsContainer = document.getElementById('quiz-options-container');
    const submitBtn = document.getElementById('quiz-submit-btn');
    const nextBtn = document.getElementById('quiz-next-btn');
    const feedbackBanner = document.getElementById('quiz-feedback-banner');

    if (!this.questions.length) {
      if (qTitle) qTitle.textContent = 'No questions available. Try analyzing a sentence first!';
      return;
    }

    const q = this.questions[this.currentIndex % this.questions.length];
    this.selectedAnswer = null;

    if (qBadge) qBadge.textContent = q.title || `Question ${this.currentIndex + 1}`;
    if (qTitle) qTitle.innerHTML = q.question;
    if (submitBtn) {
      submitBtn.disabled = true;
      submitBtn.style.display = 'inline-flex';
    }
    if (nextBtn) nextBtn.style.display = 'none';
    if (feedbackBanner) {
      feedbackBanner.style.display = 'none';
      feedbackBanner.className = 'feedback-banner';
    }

    if (optionsContainer) {
      optionsContainer.innerHTML = '';
      (q.options || []).forEach(opt => {
        const btn = document.createElement('button');
        btn.className = 'quiz-option-btn';
        btn.textContent = opt;

        btn.addEventListener('click', () => {
          document.querySelectorAll('.quiz-option-btn').forEach(b => b.classList.remove('selected'));
          btn.classList.add('selected');
          this.selectedAnswer = opt;
          if (submitBtn) submitBtn.disabled = false;
        });

        optionsContainer.appendChild(btn);
      });
    }

    this.updateScoreboard();
  },

  submitAnswer() {
    if (!this.selectedAnswer) return;

    const q = this.questions[this.currentIndex % this.questions.length];
    const isCorrect = (this.selectedAnswer === q.correct);
    const feedbackBanner = document.getElementById('quiz-feedback-banner');
    const submitBtn = document.getElementById('quiz-submit-btn');
    const nextBtn = document.getElementById('quiz-next-btn');

    this.totalAnswered += 1;
    if (isCorrect) {
      this.score += 10;
    }

    this.updateScoreboard();

    if (feedbackBanner) {
      feedbackBanner.style.display = 'block';
      if (isCorrect) {
        feedbackBanner.className = 'feedback-banner feedback-success';
        feedbackBanner.innerHTML = `<strong>${q.explanation_correct || 'Correct!'}</strong>`;
      } else {
        feedbackBanner.className = 'feedback-banner feedback-error';
        feedbackBanner.innerHTML = `
          <strong>${q.explanation_incorrect || 'Almost!'}</strong>
          <div style="margin-top:0.35rem;">Correct Answer: <strong>${Components.escapeHtml(q.correct)}</strong></div>
        `;
      }
    }

    // Disable options
    document.querySelectorAll('.quiz-option-btn').forEach(btn => {
      btn.disabled = true;
      if (btn.textContent === q.correct) {
        btn.style.borderColor = 'var(--sage)';
        btn.style.background = 'var(--sage-soft)';
      }
    });

    if (submitBtn) submitBtn.style.display = 'none';
    if (nextBtn) nextBtn.style.display = 'inline-flex';
  },

  nextQuestion() {
    this.currentIndex += 1;
    this.renderCurrentQuestion();
  },

  updateScoreboard() {
    const scoreEl = document.getElementById('quiz-score-display');
    const accEl = document.getElementById('quiz-accuracy-display');
    const countEl = document.getElementById('quiz-count-display');

    if (scoreEl) scoreEl.textContent = `${this.score} pts`;
    if (countEl) countEl.textContent = `${this.totalAnswered}`;
    if (accEl) {
      const acc = this.totalAnswered > 0 ? Math.round((this.score / (this.totalAnswered * 10)) * 100) : 100;
      accEl.textContent = `${acc}%`;
    }
  }
};
