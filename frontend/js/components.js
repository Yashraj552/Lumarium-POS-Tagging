/**
 * frontend/js/components.js — Reusable Component Handlers for Lumarium
 * Handles System Health, Compendium rendering, and Observatory tables.
 */

const Components = {
  /**
   * Checks API health and updates the header status indicator.
   */
  async checkSystemHealth() {
    const statusText = document.getElementById('system-status-text');
    try {
      const res = await fetch('/api/health');
      if (!res.ok) throw new Error('API unavailable');
      const data = await res.json();
      
      const models = data.models || {};
      const activeCount = Object.values(models).filter(m => m.available).length;
      const trfName = models.transformer?.model_name ? 'HingRoBERTa' : 'Transformer';
      statusText.textContent = `${activeCount}/3 Tiers Active (${trfName} on ${models.transformer?.device || 'CPU'})`;
      statusText.parentElement.title = `Rules: ${models.rules?.available ? 'OK' : 'Off'} | HMM: ${models.hmm?.available ? 'OK' : 'Off'} | ${trfName}: ${models.transformer?.available ? 'OK' : 'Off'}`;

      if (data.arbiter_defaults) {
        const slider = document.getElementById('slider-threshold');
        const label = document.getElementById('val-threshold-label');
        if (slider && label && data.arbiter_defaults.threshold) {
          slider.value = data.arbiter_defaults.threshold;
          label.textContent = parseFloat(data.arbiter_defaults.threshold).toFixed(2);
        }
      }
    } catch (err) {
      statusText.textContent = 'Offline (Check backend)';
      statusText.style.color = 'var(--rose)';
    }
  },

  /**
   * Fetches and renders the Universal Dependencies Grammar Compendium.
   */
  async initCompendium() {
    const filterBar = document.getElementById('compendium-filter-bar');
    const container = document.getElementById('compendium-cards-container');
    if (!filterBar || !container) return;

    try {
      const res = await fetch('/api/guide');
      const data = await res.json();
      const categories = data.categories || {};
      const guide = data.guide || [];

      // Render filter buttons
      filterBar.innerHTML = '';
      Object.keys(categories).forEach((cat, idx) => {
        const btn = document.createElement('button');
        btn.className = `filter-btn ${idx === 0 ? 'active' : ''}`;
        btn.textContent = cat;
        btn.addEventListener('click', () => {
          document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
          btn.classList.add('active');
          const allowedTags = categories[cat];
          const filtered = allowedTags ? guide.filter(item => allowedTags.includes(item.tag)) : guide;
          this.renderGuideCards(container, filtered);
        });
        filterBar.appendChild(btn);
      });

      // Initial render (All)
      this.renderGuideCards(container, guide);
    } catch (err) {
      container.innerHTML = `<p style="color:var(--text-muted)">Failed to load compendium: ${err.message}</p>`;
    }
  },

  renderGuideCards(container, items) {
    container.innerHTML = '';
    items.forEach(item => {
      const card = document.createElement('div');
      card.className = 'guide-reference-card';
      card.style.borderLeft = `5px solid ${item.color || 'var(--sage)'}`;
      card.innerHTML = `
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.5rem;">
          <h4 style="font-size:1.2rem; font-family:'Newsreader', serif; font-weight:700; color:var(--text);">
            ${item.emoji} ${item.name}
          </h4>
          <span style="background:${item.bg || 'var(--surface-2)'}; color:${item.color || 'var(--text)'}; font-weight:700; font-size:0.75rem; padding:0.2rem 0.55rem; border-radius:12px;">
            ${item.tag}
          </span>
        </div>
        <p style="font-size:0.92rem; color:var(--text); margin-bottom:0.75rem; line-height:1.5;">
          ${item.definition}
        </p>
        <div style="background:var(--surface-2); border-radius:var(--radius-sm); padding:0.65rem 0.85rem; margin-bottom:0.75rem; font-size:0.85rem;">
          <strong style="color:var(--sage-deep);">✦ Tip:</strong> ${item.student_tip}
        </div>
        <div style="background:#F6F8F5; border-radius:var(--radius-sm); padding:0.65rem 0.85rem; font-size:0.85rem; margin-bottom:0.5rem;">
          <strong style="color:var(--sage-deep);">Code-Mixed:</strong> ${item.code_mixed_example}
          <div style="font-size:0.75rem; color:var(--text-muted); margin-top:0.2rem;">${item.code_mixed_notes || ''}</div>
        </div>
        <div style="background:var(--surface-2); border-radius:var(--radius-sm); padding:0.65rem 0.85rem; font-size:0.85rem;">
          <strong style="color:var(--text);">English:</strong> ${item.english_example}
        </div>
      `;
      container.appendChild(card);
    });
  },

  /**
   * Renders the NLP Observatory Decision Trace and WordNet validation tables.
   */
  renderObservatory(analysisData) {
    const traceContainer = document.getElementById('observatory-trace-table-container');
    const wordnetContainer = document.getElementById('observatory-wordnet-container');
    const summaryLabel = document.getElementById('observatory-attribution-summary');

    if (!analysisData || !traceContainer) return;

    const trace = analysisData.decision_trace || [];
    const sourceCounts = analysisData.summary?.source_counts || {};
    const total = analysisData.summary?.total_tokens || 1;

    // Summary caption
    const sharesText = Object.entries(sourceCounts)
      .map(([src, cnt]) => `<strong>${src}</strong>: ${cnt}/${total} (${((cnt / total) * 100).toFixed(1)}%)`)
      .join(' · ');
    if (summaryLabel) {
      summaryLabel.innerHTML = `Model Attribution Share: ${sharesText}`;
    }

    // Trace Table
    let traceHtml = `
      <table class="custom-table">
        <thead>
          <tr>
            <th>Token</th>
            <th>Final Tag</th>
            <th>Source Model</th>
            <th>Confidence</th>
            <th>Rule Tier</th>
            <th>Transformer</th>
            <th>HMM Tier</th>
            <th>Decision Reason</th>
          </tr>
        </thead>
        <tbody>
    `;

    trace.forEach(row => {
      const confPct = Math.round((row.confidence || 0) * 100);
      traceHtml += `
        <tr>
          <td><strong>${this.escapeHtml(row.token)}</strong></td>
          <td><span class="token-tile-badge badge-${row.final_tag.toLowerCase()}">${row.final_tag}</span></td>
          <td><code>${row.source}</code></td>
          <td>
            <div style="display:flex; align-items:center; gap:0.5rem;">
              <span>${row.confidence ? row.confidence.toFixed(2) : '—'}</span>
              <div class="confidence-bar-track" style="width:60px; height:5px; margin:0;">
                <div class="confidence-bar-fill" style="width:${confPct}%;"></div>
              </div>
            </div>
          </td>
          <td>${this.escapeHtml(row.rule_candidate)}</td>
          <td>${this.escapeHtml(row.transformer_candidate)}</td>
          <td>${this.escapeHtml(row.hmm_candidate)}</td>
          <td style="color:var(--text-muted); font-size:0.82rem;">${this.escapeHtml(row.reason)}</td>
        </tr>
      `;
    });
    traceHtml += '</tbody></table>';
    traceContainer.innerHTML = traceHtml;

    // WordNet Table
    const valRows = analysisData.validation || [];
    if (wordnetContainer) {
      if (valRows.length > 0) {
        let wnHtml = `
          <table class="custom-table">
            <thead>
              <tr>
                <th>English Token</th>
                <th>Ensemble Tag</th>
                <th>WordNet Lexical Evidence</th>
                <th>Validated Tag</th>
              </tr>
            </thead>
            <tbody>
        `;
        valRows.forEach(r => {
          wnHtml += `
            <tr>
              <td><strong>${this.escapeHtml(r.Token || r.token)}</strong></td>
              <td><code>${r['Ensemble POS'] || r.ensemble_pos}</code></td>
              <td style="color:var(--text-muted);">${this.escapeHtml(r['WordNet Evidence'] || r.wordnet_evidence)}</td>
              <td><strong style="color:var(--sage-deep);">${r['Validated POS'] || r.validated_pos}</strong></td>
            </tr>
          `;
        });
        wnHtml += '</tbody></table>';
        wordnetContainer.innerHTML = wnHtml;
      } else {
        wordnetContainer.innerHTML = '<p style="color:var(--text-muted); padding:1rem 0;">WordNet validation evidence will appear when an English translation is computed.</p>';
      }
    }
  },

  escapeHtml(str) {
    if (!str) return '';
    return String(str)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#039;');
  }
};
