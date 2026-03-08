# Implementation Plan – Design to Code

This document maps the target design ([design.md](design.md)) to concrete code changes. It does not cover the current system in detail; it is a task-oriented plan for implementing the design.

---

## Phase 1: CSS – Pico CSS Base + Visual Refinements

### 1.1 Approach

- **Implementation base:** [Pico CSS](https://picocss.com/) – minimal CSS for semantic HTML, responsive by default, light/dark via `prefers-color-scheme`, no JavaScript.
- **Visual reference:** [Modern Digital Portfolio – No JS](https://github.com/Sohail7739/web-design-portfolio-no-js) – gradients, glassmorphism, hover motion, mobile-first.
- **Customization:** `site.css` for overrides, status components, and refinements.

### 1.2 Look and feel requirements

Implement the visual design in [design.md](design.md) § Look and Feel Requirements:

- **Typography:** Outfit (body), JetBrains Mono (code). Override Pico defaults via CSS variables if needed.
- **Colour:** Stone neutral. Status: pass=emerald, warn=amber, fail=stone. Page gradient. Light/dark compatible.
- **Layout:** Mobile-first. Max-width ~42rem, centred. CSS Grid and Flexbox.
- **Components:** Status cards (`.card[data-status]`), status dot, badge. Optional glassmorphism.
- **Interaction:** Focus outline, transitions, hover effects.

### 1.3 Add Pico CSS and site overrides

**Add Pico CSS:**
- Option A: CDN `<link href="https://cdn.jsdelivr.net/npm/@picocss/pico@2/css/pico.min.css" rel="stylesheet">`
- Option B: npm `@picocss/pico`, copy to `static/css/pico.min.css`, link via `url_for`

**Add `site.css` after Pico:**
- Focus styles (`a:focus-visible`, `button:focus-visible`)
- Heading superscript (mw)²
- Status components: `.card[data-status="pass"|"warn"|"fail"]`, `.status-dot`, `.badge`
- Page gradient background
- Optional: glassmorphism, hover transitions
- Override Pico variables for typography (Outfit, JetBrains Mono) and colours if needed

**base.html:** Link Pico first, then `site.css`. Use semantic HTML; Pico styles elements by default. Add custom classes only where needed (status cards, badge).

### 1.4 Update base.html

**Remove:** Existing CDN stylesheet and inline theme/config script.

**Add:** Pico CSS link, then `site.css` link. Google Fonts (Outfit, JetBrains Mono).

**Update markup:** Use semantic HTML (`<header>`, `<nav>`, `<main>`, `<footer>`, `<section>`) so Pico applies. Add `container` class for centred content if using Pico’s class. Status cards use `data-status` and custom classes from `site.css`.

---

## Phase 2: Health – HTMX Polling + Jinja Macro

**Current state:** Route at `/health/dashboard`. `get_health_context()` (from `utils.health.run_checks`) returns `{"health_results": [{component_name, status, description}, ...]}`. Status is `pass`, `warn`, or `fail`. Template iterates `health_results`; `health-poll.js` polls `/mw2/v1/status` and updates DOM.

### 2.1 Create Jinja macro

**File:** `app/templates/health/_macros.html` (create)

```jinja2
{% macro service_card(item) %}
<li>
  <div class="card" data-service="{{ item.component_name }}" data-status="{{ item.status }}">
    <span class="status-dot"></span>
    <span class="card-label">{{ item.component_name }}</span>
    <span class="card-value">{{ item.description }}</span>
  </div>
</li>
{% endmacro %}
```

### 2.2 Create services partial

**File:** `app/templates/health/_services.html` (create)

```jinja2
{% from "health/_macros.html" import service_card %}
<ul class="service-list">
  {% for item in health_results %}
  {{ service_card(item) }}
  {% endfor %}
</ul>
```

Requires `health_results` in context (from `get_health_context()`).

### 2.3 Add fragment route

**File:** `app/routes/health.py`

Add route:

```python
@health_bp.route("/fragments/services")
def services_fragment():
    """Return services HTML fragment for HTMX polling."""
    return render_template("health/_services.html", **get_health_context())
```

### 2.4 Update health index template

**File:** `app/templates/health/index.html`

- Replace `{% for item in health_results %}` loop body with macro: `{% from "health/_macros.html" import service_card %}` and `{{ service_card(item) }}`.
- Wrap the `<ul>` in a div with HTMX attributes:
  - `hx-get="{{ url_for('health.services_fragment') }}"`
  - `hx-trigger="every 10s"`
  - `hx-swap="innerHTML"`
- **Remove** `{% block scripts %}` block that loads `health-poll.js`.

### 2.5 Add HTMX to base.html

**File:** `app/templates/base.html`

Add before `</body>` (or in `{% block scripts %}` for pages that need it):

```html
<script src="https://unpkg.com/htmx.org@2.0.0"></script>
```

Or use a local copy in `static/js/htmx.min.js` and `url_for`.

### 2.6 Delete health-poll.js

**File:** `app/static/js/health-poll.js` – **Delete**. Replaced by HTMX.

---

## Phase 3: T7e – Translation Page (Use Cases 2 & 3)

The design defines three translation flows. Implement in this order: Use Case 2 (poll for full result) first, then Use Case 3 (streaming).

### 3.1 RQ queue access from Flask app

**Current state:** `rqworker` uses `get_redis_connection()` and `Queue`. Flask app does not enqueue jobs.

**Action:** Ensure Flask can enqueue jobs. Add helper or use existing Redis + RQ:

**File:** `app/services/translation.py` (create)

- `enqueue_translation_job(text: str, ...) -> str` returns `job_id`.
- Uses `Queue(connection=get_redis_connection())` and `queue.enqueue(...)`.
- Job function will be defined in a module the worker can import.

### 3.2 Translation job function (worker)

**File:** `app/jobs/translation.py` or `rqworker/jobs/translation.py` (create)

- `def run_translation(text: str, job_id: str)` – call AI, store result in Redis (for Use Case 2) or publish tokens to Redis pub/sub (for Use Case 3).
- Worker must be able to import this. Register in `rqworker/__main__.py` or ensure it's on the worker's Python path.

### 3.3 Use Case 2: Poll for full result

**Routes:**

- `POST /t7e/translate` – create job, return HTML fragment containing:
  - A div with `hx-get="/t7e/job/{job_id}"`, `hx-trigger="every 2s"`, `hx-swap="innerHTML"`.
  - Initial content: "Processing…" spinner.
- `GET /t7e/job/<job_id>` – if job pending/running: return "Processing…" fragment. If finished: return full result fragment.

**Worker job:** Run translation, store full result in Redis key `job:{job_id}:result` (or use RQ's result backend). Set status in `job:{job_id}:status` (e.g. "pending", "done", "failed").

**Templates:**

- `t7e/index.html` – form with `hx-post="/t7e/translate"`, `hx-target` to output div, `hx-swap="innerHTML"`.
- `t7e/_job_poll.html` – partial for the polling div (includes hx-get, hx-trigger).
- `t7e/_job_result.html` – partial for "Processing…" and for full result.

### 3.4 Use Case 3: Streaming results

**Routes:**

- `POST /t7e/translate` – create job. Return HTML that includes `hx-sse="connect:/t7e/job/{job_id}/stream"` (HTMX SSE extension).
- `GET /t7e/job/<job_id>/stream` – SSE endpoint. Subscribe to Redis channel `job:{job_id}`. Yield `data: {chunk}\n\n` for each message. Close when worker sends "done".

**Worker job:** Call AI with `stream=True`. For each token, `redis.publish(f"job:{job_id}", chunk)`. At end, `redis.publish(f"job:{job_id}", "done")` or use a sentinel.

**AI client streaming:** Add support in `utils/ai_llm/ai_client.py` or a wrapper:

- `chat.completions.create(..., stream=True)` returns an iterator.
- Worker iterates and publishes each chunk.

**HTMX SSE:** Include `hx-sse` extension. Check HTMX docs for exact attribute syntax and swap behaviour (append vs replace).

### 3.5 T7e template structure

**File:** `app/templates/t7e/index.html`

- Form: text input/textarea, submit button.
- Output div for HTMX to target.
- Option: toggle or separate URLs for "poll" vs "stream" mode, or implement both and let user choose.

### 3.6 Translation service – shared logic

**File:** `app/services/translation.py`

- `enqueue_translation_job(text, mode="poll"|"stream")` – enqueue with appropriate job function.
- `get_job_status(job_id)` – return status and result (for poll endpoint).
- Optional: `subscribe_job_stream(job_id)` – generator that yields from Redis pub/sub (for SSE route).

---

## Phase 4: Dependency & Config

### 4.1 Add HTMX

- Option A: CDN `<script src="https://unpkg.com/htmx.org@2.0.0"></script>` in base.html.
- Option B: Download to `static/js/htmx.min.js`, link via `url_for`.
- For Use Case 3: include `htmx-ext-sse.js` (HTMX SSE extension).

### 4.2 RQ in Flask

Ensure `rq` is a dependency. Flask app uses `utils.redis.get_redis_connection()` and `rq.Queue` to enqueue. No Flask-RQ required if using plain RQ.

### 4.3 Redis pub/sub

`redis-py` supports `pubsub()`. Worker and Flask SSE route both use `get_redis_connection()`.

---

## File Checklist

| Action | Path |
|--------|------|
| Add | Pico CSS (CDN or `static/css/pico.min.css`) |
| Modify | `app/static/css/site.css` (overrides, status components, refinements) |
| Create | `app/templates/health/_macros.html` |
| Create | `app/templates/health/_services.html` |
| Create | `app/services/translation.py` |
| Create | `app/jobs/translation.py` (or `rqworker/jobs/translation.py`) |
| Create | `app/templates/t7e/_job_poll.html` (Use Case 2) |
| Create | `app/templates/t7e/_job_result.html` (Use Case 2) |
| Create | `app/templates/t7e/_job_stream.html` (Use Case 3, if needed) |
| Modify | `app/templates/base.html` |
| Modify | `app/templates/health/index.html` |
| Modify | `app/templates/root/index.html` (semantic HTML) |
| Modify | `app/templates/t7e/index.html` |
| Modify | `app/routes/health.py` (add services_fragment) |
| Modify | `app/routes/t7e.py` |
| Modify | `app/__init__.py` (if new blueprint or routes) |
| Delete | `app/static/js/health-poll.js` |

---

## Order of Implementation

1. **Phase 1** – CSS: Add Pico CSS, update site.css (overrides, status components, refinements), update base.html and templates with semantic HTML. Mobile-first; test on touch and desktop.
2. **Phase 2** – Health: Macro, partial, fragment route, HTMX in base, update health template, remove health-poll.js.
3. **Phase 3 (Use Case 2)** – Translation poll: Job function, translation service, POST/GET routes, templates.
4. **Phase 3 (Use Case 3)** – Translation stream: Streaming in AI client/worker, Redis pub/sub, SSE route, HTMX SSE.

Phase 1 and 2 are independent and can be done in parallel after Phase 1 templates are updated. Phase 3 depends on RQ and Redis being wired in.
