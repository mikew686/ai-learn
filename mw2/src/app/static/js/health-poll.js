/**
 * Health page: poll /mw2/v1/status and update service cards.
 */
(function () {
  const STYLES = {
    pass: {
      card: 'flex items-center gap-3 rounded-xl border px-4 py-3 border-emerald-500/50 bg-emerald-500/10 text-emerald-800',
      dot: 'status-dot h-3 w-3 shrink-0 rounded-full bg-emerald-500',
    },
    warn: {
      card: 'flex items-center gap-3 rounded-xl border px-4 py-3 border-amber-500/50 bg-amber-500/10 text-amber-800',
      dot: 'status-dot h-3 w-3 shrink-0 rounded-full bg-amber-500',
    },
    fail: {
      card: 'flex items-center gap-3 rounded-xl border px-4 py-3 border-stone-200 bg-stone-50 text-stone-500',
      dot: 'status-dot h-3 w-3 shrink-0 rounded-full bg-stone-300',
    },
  };

  function updateCard(componentName, status, description) {
    const card = document.querySelector('[data-service="' + componentName + '"]');
    if (!card) return;
    const s = STYLES[status] || STYLES.fail;
    card.className = s.card;
    const dot = card.querySelector('.status-dot');
    if (dot) dot.className = s.dot;
    const value = card.querySelector('.status-value');
    if (value) value.textContent = description;
  }

  function poll() {
    fetch('/mw2/v1/status')
      .then(function (r) { return r.json(); })
      .then(function (data) {
        const results = data.health_results || [];
        results.forEach(function (item) {
          updateCard(item.component_name, item.status, item.description);
        });
      })
      .catch(function () {});
  }

  poll();
  setInterval(poll, 10000);
})();
