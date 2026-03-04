/**
 * Health page: poll /api/status and update service cards and k8s badge.
 */
(function () {
  const CARD_ON = 'flex items-center gap-3 rounded-xl border px-4 py-3 border-emerald-500/50 bg-emerald-500/10 text-emerald-800';
  const CARD_OFF = 'flex items-center gap-3 rounded-xl border px-4 py-3 border-stone-200 bg-stone-50 text-stone-500';
  const DOT_ON = 'status-dot h-3 w-3 shrink-0 rounded-full bg-emerald-500';
  const DOT_OFF = 'status-dot h-3 w-3 shrink-0 rounded-full bg-stone-300';

  function updateCard(service, on, valueText) {
    const card = document.querySelector('[data-service="' + service + '"]');
    if (!card) return;
    card.className = on ? CARD_ON : CARD_OFF;
    const dot = card.querySelector('.status-dot');
    if (dot) dot.className = on ? DOT_ON : DOT_OFF;
    const value = card.querySelector('.status-value');
    if (value) value.textContent = valueText;
  }

  function poll() {
    fetch('/api/status')
      .then(function (r) { return r.json(); })
      .then(function (data) {
        updateCard('redis', data.redis_enabled, data.redis_key_count + ' keys');
        updateCard('postgres', data.postgres_enabled, data.postgres_table_count + ' tables');
        updateCard('pgvector', data.postgres_vector_available, data.postgres_vector_available ? 'available' : 'not loaded');
        updateCard('rq', data.rq_workers_running, data.rq_worker_count + ' running');
        updateCard('ai', data.ai_available, data.ai_model_count + ' models');
        const badge = document.getElementById('k8s-badge');
        if (badge) badge.classList.toggle('hidden', !data.running_on_kubernetes);
      })
      .catch(function () {});
  }

  setInterval(poll, 10000);
})();
