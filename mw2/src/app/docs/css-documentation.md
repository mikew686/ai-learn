# CSS Documentation Guide

**Approach:** [Pico CSS](https://picocss.com/) as base. `main.css` for overrides and custom components.

---

## Look and feel reference

CSS implements the visual design in [design.md](design.md) § Look and Feel Requirements.

| Aspect | Requirements |
|--------|--------------|
| Typography | Outfit (body), JetBrains Mono (code). Override Pico via variables if needed. |
| Colour | Stone neutral. Status: pass=emerald, warn=amber, fail=stone. Page gradient. Light/dark via `prefers-color-scheme`. |
| Layout | Mobile-first. Max-width ~42rem, centred. CSS Grid, Flexbox. |
| Cards | Rounded, border, flex, gap. `data-status` (pass/warn/fail) drives styling. Optional glassmorphism. |
| Status dot | ~12px, rounded-full. Colour by status. |
| Interaction | Focus 2px outline. Transitions for hover/focus. Optional hover motion. Heading superscript (mw)² styled. |

---

## main.css structure

Pico provides base styles for semantic HTML. See [Pico’s documentation](https://picocss.com/docs/) for defaults. `main.css` contains:

- **Overrides:** Focus styles, heading superscript, typography, and list styles where Pico’s defaults conflict.
- **Custom components:** Status cards (`.card[data-status]`), `.status-dot`, `.badge`.
- **Refinements:** Page gradient, transitions, layout tuning.

---

## Overriding Pico: specificity and list styles

Pico CSS styles `ul` inside `main` with list markers (bullets). When we use a semantic `<ul class="service-list">` for card-like items, we want no bullets. A simple `.service-list { list-style: none }` can be overridden by Pico’s more specific selectors.

**Specificity:** Pico uses `main ul` (specificity `0,1,1`). Our `.service-list` alone has `0,1,0`, so Pico wins when both match. To reliably override, we need equal or higher specificity.

**Workaround:** Scope the override under the containing `section`:

```css
#services .service-list,
#services .service-list li {
  list-style: none;
}
```

`#services` (ID selector) raises specificity so our rules win over Pico’s `main ul`. Apply `list-style: none` to both the `ul` and its `li` elements so any inherited or child-level list styling is removed.

**Learning point:** When overriding a framework, check which selectors the framework uses. Use IDs or chained selectors to increase specificity only where necessary. See [MDN: Specificity](https://developer.mozilla.org/en-US/docs/Web/CSS/Specificity).

---

## Documented components and sections

| Section | Purpose |
|---------|---------|
| Typography | Custom font families (Outfit, JetBrains Mono). `.text-muted` for secondary text. |
| Page layout | `.page` gradient background. `.container` max-width and padding, mobile-first. |
| Header | Nav as flex layout, `list-style: none` for nav `ul`, hover styles. |
| Focus | `:focus-visible` outline for keyboard accessibility. |
| Heading superscript | `h1 sup` sizing for (mw)². |
| Status card | `.service-list` (see workaround above). `.card` with `data-status` for pass/warn/fail. `.card-label`, `.card-value`. |
| Status dot | `.status-dot` – 12px circle, colour from parent `data-status`. |
| Section heading | `.section-heading` – uppercase, muted, small. |
| Footer | Top border, spacing, link hover. |
| Prose | `.prose p` for content sections. |

---

## File header and section comments

`main.css` begins with a block comment: purpose and relationship to Pico.

Group related rules under short headings using `---` delimiters.

---

## Internal documentation

- **Variables.** One-line comment per custom property.
- **Components.** One-line description of role and `data-*` attributes before each block.
- **Non-obvious rules.** Inline comment where intent is unclear (e.g. specificity overrides).
