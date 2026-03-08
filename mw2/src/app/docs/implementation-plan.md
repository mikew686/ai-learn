# Implementation Plan – Design to Code

This document maps the target design ([design.md](design.md)) to concrete code changes. It does not cover the current system in detail; it is a task-oriented plan for implementing the design.

---

## Phase 1: CSS – Modular Utility Layer

### 1.1 Create modular CSS files

**Current state:** `site.css` has ~19 lines of overrides. `base.html` loads Tailwind CDN + `site.css`. All layout/component styles come from Tailwind utility classes in templates.

**Target:** Replace Tailwind with five modular CSS files. Remove CDN dependency.

| File | Action | Content |
|------|--------|---------|
| `static/css/variables.css` | **Create** | `:root` with `--color-*`, `--spacing-*`, `--font-*`. Design tokens for stone/emerald/sky palette, Outfit/JetBrains Mono. |
| `static/css/layout.css` | **Create** | `.container` (max-width, margin auto, px, py), responsive breakpoints via `@media (max-width: ...)`. Desktop-first: base styles for large viewport; media queries for smaller. |
| `static/css/components.css` | **Create** | `.card`, `.card[data-status="on"]`, `.card[data-status="off"]`, `.status-dot`, `.badge`, `.btn`. Match current visual: rounded-xl, borders, emerald/stone colours. |
| `static/css/utilities.css` | **Create** | `.text-muted`, `.font-mono`, etc. |
| `static/css/site.css` | **Modify** | Keep focus styles (`a:focus-visible`, `button:focus-visible`), `h1 sup` styling. Remove Tailwind-specific comment. |

**Documentation:** Follow [css-documentation.md](css-documentation.md) for each file. Use human-readable comments: file headers describing purpose, section headings (`---` blocks), inline notes for non-obvious rules. Document variables with one-line descriptions; document components with usage and `data-*` attributes.

### 1.2 Update base.html

**Remove:**
- Tailwind CDN `<script src="https://cdn.tailwindcss.com">`
- Tailwind config `<script> tailwind.config = {...} </script>`

**Add:** Five `<link>` tags in order:

```html
<link rel="stylesheet" href="{{ url_for('static', filename='css/variables.css') }}">
<link rel="stylesheet" href="{{ url_for('static', filename='css/layout.css') }}">
<link rel="stylesheet" href="{{ url_for('static', filename='css/components.css') }}">
<link rel="stylesheet" href="{{ url_for('static', filename='css/utilities.css') }}">
<link rel="stylesheet" href="{{ url_for('static', filename='css/site.css') }}">
```

**Update body/main/header/nav/footer:** Replace Tailwind classes with new semantic classes. For example:
- `class="min-h-screen bg-gradient-to-br from-stone-100..."` → `class="page"` (define in layout.css)
- `class="mx-auto max-w-2xl px-6 py-16"` → `class="container"`
- `class="font-medium text-stone-700 hover:text-stone-900"` → `.nav-link` in components.css

**Keep:** Google Fonts links (or move to self-hosted later). Meta viewport.

---

## Phase 2: Health – HTMX Polling + Jinja Macro

### 2.1 Refactor health service – `get_services_list()`

**File:** `app/services/health.py`

**Add function:**

```python
def get_services_list() -> list[dict]:
    """Return list of service items for templates. Keys: service, enabled, value_text."""
```

**Logic:** Call `get_health_context()` (or internal helpers). Build list:

```python
[
    {"service": "redis", "enabled": redis_connected, "value_text": f"{redis_key_count} keys"},
    {"service": "postgres", "enabled": postgres_enabled, "value_text": f"{postgres_table_count} tables"},
    {"service": "pgvector", "enabled": postgres_vector_available, "value_text": "available" if postgres_vector_available else "not loaded"},
    {"service": "rq", "enabled": rq_worker_count > 0, "value_text": f"{rq_worker_count} running"},
    {"service": "ai", "enabled": ai_success, "value_text": f"{ai_model_count} models"},
]
```

**Update `get_health_context()`:** Add `"services": get_services_list()` and keep existing flat keys for API compatibility (or deprecate flat keys if API is updated).

### 2.2 Create Jinja macro

**File:** `app/templates/health/_macros.html` (create)

```jinja2
{% macro service_card(service, enabled, value_text) %}
<li>
  <div class="card" data-service="{{ service }}" data-status="{{ 'on' if enabled else 'off' }}">
    <span class="status-dot"></span>
    <span class="card-label">{{ service }}</span>
    <span class="card-value">{{ value_text }}</span>
  </div>
</li>
{% endmacro %}
```

(Adjust class names to match `components.css`.)

### 2.3 Create services partial

**File:** `app/templates/health/_services.html` (create)

```jinja2
{% from "health/_macros.html" import service_card %}
<ul class="service-list">
  {% for item in services %}
  {{ service_card(item.service, item.enabled, item.value_text) }}
  {% endfor %}
</ul>
```

Requires `services` in context.

### 2.4 Add fragment route

**File:** `app/routes/health.py`

Add route:

```python
@health_bp.route("/fragments/services")
def services_fragment():
    """Return services HTML fragment for HTMX polling."""
    services = get_services_list()
    return render_template("health/_services.html", services=services)
```

### 2.5 Update health index template

**File:** `app/templates/health/index.html`

- Replace five hard-coded `<li>` blocks with `{% from "health/_macros.html" import service_card %}` and `{% for item in services %}{{ service_card(...) }}{% endfor %}`.
- Wrap the `<ul>` in a div with HTMX attributes:
  - `hx-get="{{ url_for('health.services_fragment') }}"`
  - `hx-trigger="every 10s"`
  - `hx-swap="innerHTML"`
- Update k8s badge to use CSS class for visibility (e.g. `.hidden` when `not running_on_kubernetes`).
- **Remove** `{% block scripts %}` block that loads `health-poll.js`.

### 2.6 Add HTMX to base.html

**File:** `app/templates/base.html`

Add before `</body>` (or in `{% block scripts %}` for pages that need it):

```html
<script src="https://unpkg.com/htmx.org@2.0.0"></script>
```

Or use a local copy in `static/js/htmx.min.js` and `url_for`.

### 2.7 Delete health-poll.js

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
| Create | `app/static/css/variables.css` |
| Create | `app/static/css/layout.css` |
| Create | `app/static/css/components.css` |
| Create | `app/static/css/utilities.css` |
| Create | `app/templates/health/_macros.html` |
| Create | `app/templates/health/_services.html` |
| Create | `app/services/translation.py` |
| Create | `app/jobs/translation.py` (or `rqworker/jobs/translation.py`) |
| Create | `app/templates/t7e/_job_poll.html` (Use Case 2) |
| Create | `app/templates/t7e/_job_result.html` (Use Case 2) |
| Create | `app/templates/t7e/_job_stream.html` (Use Case 3, if needed) |
| Modify | `app/static/css/site.css` |
| Modify | `app/templates/base.html` |
| Modify | `app/templates/health/index.html` |
| Modify | `app/templates/root/index.html` (replace Tailwind classes) |
| Modify | `app/templates/t7e/index.html` |
| Modify | `app/routes/health.py` |
| Modify | `app/routes/t7e.py` |
| Modify | `app/services/health.py` |
| Modify | `app/__init__.py` (if new blueprint or routes) |
| Delete | `app/static/js/health-poll.js` |

---

## Order of Implementation

1. **Phase 1** – CSS: Create modular files, update base.html, migrate templates from Tailwind to semantic classes. Test layout on desktop and mobile.
2. **Phase 2** – Health: Add `get_services_list()`, macro, partial, fragment route, HTMX in base, update health template, remove health-poll.js.
3. **Phase 3 (Use Case 2)** – Translation poll: Job function, translation service, POST/GET routes, templates.
4. **Phase 3 (Use Case 3)** – Translation stream: Streaming in AI client/worker, Redis pub/sub, SSE route, HTMX SSE.

Phase 1 and 2 are independent and can be done in parallel after Phase 1 templates are updated. Phase 3 depends on RQ and Redis being wired in.
