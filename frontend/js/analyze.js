/**
 * frontend/js/analyze.js — Analysis Engine and Visual Rendering for Lumarium
 * Handles POST /api/analyze and rendering word tiles, inspector, translation,
 * morphology, language mix, and named entities.
 */

const Analyzer = {
  currentData: null,
  activeTokenIndex: 0,

  /**
   * Executes POS analysis via the FastAPI endpoint.
   */
  async runAnalysis(text, options = {}) {
    const resultsSection = document.getElementById('analysis-results-section');
    const submitBtn = document.getElementById('analyze-submit-btn');

    try {
      if (submitBtn) {
        submitBtn.disabled = true;
        submitBtn.innerHTML = '<span>Analyzing…</span> ✦';
      }

      const payload = {
        text: text.trim(),
        threshold: options.threshold !== undefined ? options.threshold : 0.85,
        fallback: options.fallback || 'hmm',
        rule_overridable: options.rule_overridable !== undefined ? options.rule_overridable : true,
        use_transformer: options.use_transformer !== undefined ? options.use_transformer : true,
      };

      const res = await fetch('/api/analyze', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });

      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || 'Analysis request failed.');
      }

      const data = await res.json();
      this.currentData = data;
      this.activeTokenIndex = 0;

      // Render all subcomponents
      this.renderWordTiles(data.tokens);
      this.renderTokenInspector(data.tokens[0]);
      this.renderSynthesis(data.summary);
      this.renderTriptych(data.translation, data.text);
      this.renderMorphology(data.morphology);
      this.renderLanguageSpectrum(data.language_identification);
      this.renderEntities(data.entities);

      // Also update the Observatory
      if (window.Components) {
        window.Components.renderObservatory(data);
      }

      if (resultsSection) {
        resultsSection.style.display = 'block';
        resultsSection.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
      }

      return data;
    } catch (err) {
      alert(`Could not analyze sentence: ${err.message}`);
    } finally {
      if (submitBtn) {
        submitBtn.disabled = false;
        submitBtn.innerHTML = '<span>Analyze Sentence</span> ✦';
      }
    }
  },

  /**
   * Renders the interactive word cards sequence.
   */
  renderWordTiles(tokens) {
    const container = document.getElementById('tokens-stream');
    if (!container || !tokens) return;
    container.innerHTML = '';

    tokens.forEach((t, idx) => {
      const tile = document.createElement('div');
      tile.className = `token-tile ${t.color_class} ${idx === this.activeTokenIndex ? 'active-token' : ''}`;
      tile.title = `${t.word} → ${t.friendly_name} (Confidence: ${t.confidence ? (t.confidence * 100).toFixed(0) + '%' : '—'})`;
      tile.innerHTML = `
        <span class="token-tile-word">${Components.escapeHtml(t.word)}</span>
        <span class="token-tile-badge">${t.pos}</span>
        <span class="token-tile-role">${t.friendly_name.split(' ')[0]}</span>
      `;

      tile.addEventListener('click', () => {
        document.querySelectorAll('.token-tile').forEach(el => el.classList.remove('active-token'));
        tile.classList.add('active-token');
        this.activeTokenIndex = idx;
        this.renderTokenInspector(t);
      });

      container.appendChild(tile);
    });
  },

  /**
   * Renders the Word Illumination Inspector (Token Details Panel).
   */
  renderTokenInspector(token) {
    const container = document.getElementById('token-inspector');
    if (!container || !token) return;

    container.style.borderLeftColor = token.accent || 'var(--sage)';
    const trf = token.candidates?.transformer;
    const rule = token.candidates?.rule;
    const hmm = token.candidates?.hmm;

    const trfConf = trf?.conf ? Math.round(trf.conf * 100) : null;
    const hmmConf = hmm?.conf ? Math.round(hmm.conf * 100) : null;

    container.innerHTML = `
      <div class="inspector-header">
        <div>
          <span style="font-size:0.78rem; font-weight:700; text-transform:uppercase; letter-spacing:0.04em; color:var(--text-muted);">
            Active Illumination
          </span>
          <div class="inspector-word-title">${Components.escapeHtml(token.word)}</div>
        </div>
        <div style="text-align:right;">
          <span class="inspector-pos-pill" style="background:${token.accent || 'var(--sage)'};">
            <span>${token.icon || '✦'}</span>
            <span>${token.pos} · ${token.friendly_name}</span>
          </span>
          <div style="font-size:0.8rem; font-weight:600; color:var(--text-muted); margin-top:0.35rem;">
            Word Category: ${token.category}
          </div>
        </div>
      </div>

      <div class="inspector-definition-box">
        <strong style="color:var(--text);">✦ Linguistic Definition:</strong> ${token.friendly_desc}
      </div>

      <h4 style="font-size:1.05rem; font-family:'Newsreader', serif; margin-bottom:0.75rem; color:var(--text);">
        Why did Lumarium assign this tag?
      </h4>
      <div class="model-insights-grid">
        <div class="insight-item">
          <span class="insight-label">🧠 Transformer Tier</span>
          <span class="insight-value">${trf ? `${trf.tag} (${trfConf}%)` : 'Not triggered'}</span>
          ${trfConf ? `<div class="confidence-bar-track"><div class="confidence-bar-fill" style="width:${trfConf}%;"></div></div>` : ''}
        </div>

        <div class="insight-item">
          <span class="insight-label">📚 Morphological Rules</span>
          <span class="insight-value">${rule ? `${rule.tag} (${rule.rule || 'regex'})` : 'No rule fired'}</span>
        </div>

        <div class="insight-item">
          <span class="insight-label">📊 Second-Order HMM</span>
          <span class="insight-value">${hmm ? `${hmm.tag} (${hmmConf}%)` : 'Not triggered'}</span>
          ${hmmConf ? `<div class="confidence-bar-track"><div class="confidence-bar-fill" style="width:${hmmConf}%;"></div></div>` : ''}
        </div>
      </div>

      <div class="details-collapsible">
        <button id="toggle-technical-btn" class="details-toggle-btn">
          <span>▼ Show Technical Arbitration Logs</span>
        </button>
        <div id="technical-raw-box" class="technical-raw-box">
Final Chosen Tag: ${token.pos}
Selected Tier: ${token.source}
Confidence Score: ${token.confidence}
Arbiter Rationale: ${token.reason}
Candidates: ${JSON.stringify(token.candidates, null, 2)}
        </div>
      </div>
    `;

    const toggleBtn = container.querySelector('#toggle-technical-btn');
    const rawBox = container.querySelector('#technical-raw-box');
    if (toggleBtn && rawBox) {
      toggleBtn.addEventListener('click', () => {
        const isOpen = rawBox.style.display === 'block';
        rawBox.style.display = isOpen ? 'none' : 'block';
        toggleBtn.innerHTML = isOpen ? '<span>▼ Show Technical Arbitration Logs</span>' : '<span>▲ Hide Technical Logs</span>';
      });
    }
  },

  /**
   * Renders the Grammatical Synthesis chips.
   */
  renderSynthesis(summary) {
    const container = document.getElementById('grammatical-synthesis');
    if (!container || !summary) return;

    const chips = summary.friendly_counts || [];
    const chipsHtml = chips.map(c => `
      <div class="synthesis-chip">
        <span>${c.icon}</span>
        <span><strong>${c.count}</strong> ${c.name} (${c.tag})</span>
      </div>
    `).join('');

    container.innerHTML = `
      <div class="synthesis-title">
        <span>✦</span>
        <span>Grammatical Synthesis</span>
      </div>
      <p style="font-size:0.9rem; color:var(--text-muted); margin-bottom:0.75rem;">
        Your sentence is composed of <strong>${summary.total_tokens} words</strong> with the following grammatical constituents:
      </p>
      <div class="synthesis-chips-grid">${chipsHtml}</div>
    `;
  },

  /**
   * Renders the Script & Translation Triptych.
   */
  renderTriptych(trans, rawText) {
    const container = document.getElementById('translation-triptych');
    if (!container || !trans) return;

    container.innerHTML = `
      <div class="triptych-card">
        <span class="triptych-tag">💬 Input Sentence (${trans.language_label || 'Code-Mixed'})</span>
        <div class="triptych-content">${Components.escapeHtml(rawText)}</div>
      </div>

      <div class="triptych-card">
        <span class="triptych-tag">📜 Native Script (Devanagari)</span>
        <div class="triptych-content">${Components.escapeHtml(trans.native_script || '—')}</div>
      </div>

      <div class="triptych-card">
        <span class="triptych-tag">📖 Fluent English Meaning</span>
        <div class="triptych-content">${Components.escapeHtml(trans.english || '—')}</div>
      </div>
    `;
  },

  /**
   * Renders Word Anatomy (Morphology).
   */
  renderMorphology(morph) {
    const container = document.getElementById('morphology-table-container');
    if (!container || !morph) return;

    const rows = morph.original || [];
    let html = `
      <table class="custom-table">
        <thead>
          <tr>
            <th>Word</th>
            <th>Stem / Root</th>
            <th>Affix / Ending</th>
            <th>Number</th>
            <th>Tense / Feature</th>
          </tr>
        </thead>
        <tbody>
    `;

    rows.forEach(r => {
      html += `
        <tr>
          <td><strong>${Components.escapeHtml(r.Token)}</strong></td>
          <td>${Components.escapeHtml(r.Root)}</td>
          <td><code>${Components.escapeHtml(r.Suffix)}</code></td>
          <td>${Components.escapeHtml(r.Number)}</td>
          <td><span style="color:var(--text-muted); font-size:0.82rem;">${Components.escapeHtml(r.Other)}</span></td>
        </tr>
      `;
    });
    html += '</tbody></table>';
    container.innerHTML = html;
  },

  /**
   * Renders the Language Spectrum Bar.
   */
  renderLanguageSpectrum(lid) {
    const container = document.getElementById('language-spectrum-container');
    if (!container || !lid) return;

    const stats = lid.stats || {};
    const mr = stats['Marathi Tokens'] || 0;
    const en = stats['English Tokens'] || 0;
    const ot = stats['Other Tokens'] || 0;
    const total = Math.max(1, stats['Total Tokens'] || 1);

    const pMr = Math.round((mr / total) * 100);
    const pEn = Math.round((en / total) * 100);
    const pOt = Math.max(0, 100 - pMr - pEn);

    const diagnosis = pMr >= pEn
      ? 'Predominantly regional (Hindi/Marathi) with English elements'
      : 'Predominantly English with code-mixed Hindi/Marathi borrowings';

    container.innerHTML = `
      <div class="spectrum-track">
        <div class="spectrum-seg-mr" style="width:${pMr}%;" title="Regional: ${pMr}%"></div>
        <div class="spectrum-seg-en" style="width:${pEn}%;" title="English: ${pEn}%"></div>
        <div class="spectrum-seg-ot" style="width:${pOt}%;" title="Other: ${pOt}%"></div>
      </div>
      <div class="spectrum-legend">
        <div><span class="legend-dot" style="background:var(--sage);"></span>Regional: ${pMr}% (${mr} words)</div>
        <div><span class="legend-dot" style="background:var(--lavender);"></span>English: ${pEn}% (${en} words)</div>
        <div><span class="legend-dot" style="background:#8E98A8;"></span>Other: ${pOt}%</div>
      </div>
      <div style="background:var(--surface-2); padding:0.6rem 0.85rem; border-radius:var(--radius-sm); font-size:0.88rem; color:var(--text);">
        <strong style="color:var(--sage-deep);">✦ Spectrum Diagnosis:</strong> ${diagnosis}.
      </div>
    `;
  },

  /**
   * Renders Named Entity Badges.
   */
  renderEntities(entities) {
    const container = document.getElementById('ner-entities-container');
    if (!container) return;

    const list = entities?.original || [];
    if (list.length === 0) {
      container.innerHTML = `
        <div style="font-size:0.88rem; color:var(--text-muted); font-style:italic; padding:0.5rem 0;">
          No named individuals, geographic locations, or institutions were detected in this sentence.
        </div>
      `;
      return;
    }

    container.innerHTML = list.map(e => `
      <div class="entity-badge-card">
        <span class="entity-type-tag">${Components.escapeHtml(e['Entity Type'])}</span>
        <span class="entity-name-text">${Components.escapeHtml(e.Entity)}</span>
      </div>
    `).join('');
  }
};
