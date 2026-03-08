# CSS Documentation Guide

**Approach:** [Pico CSS](https://picocss.com/) as base. [Modern Digital Portfolio – No JS](https://github.com/Sohail7739/web-design-portfolio-no-js) as visual reference. `site.css` for overrides and custom components.

---

## Look and feel reference

CSS implements the visual design in [design.md](design.md) § Look and Feel Requirements.

| Aspect | Requirements |
|--------|--------------|
| Typography | Outfit (body), JetBrains Mono (code). Override Pico via variables if needed. |
| Colour | Stone neutral. Status: pass=emerald, warn=amber, fail=stone. Page gradient. Light/dark via `prefers-color-scheme`. |
| Layout | Mobile-first. Max-width ~42rem, centred. CSS Grid, Flexbox. |
| Cards | Rounded, border, flex, gap. `data-status="pass"|"warn"|"fail"` drives styling. Optional glassmorphism. |
| Status dot | ~12px, rounded-full. Colour by status. |
| Interaction | Focus 2px outline. Transitions for hover/focus. Optional hover motion. Heading superscript (mw)² styled. |

---

## site.css structure

Pico provides base styles. `site.css` contains:

- **Overrides:** Focus styles, heading superscript, typography (if Pico variables insufficient).
- **Custom components:** Status cards (`.card[data-status]`), `.status-dot`, `.badge`.
- **Refinements:** Page gradient, glassmorphism, transitions. Inspired by portfolio reference.

---

## File header

`site.css` begins with a brief block comment: purpose, relationship to Pico and visual reference.

---

## Section comments

Group related rules. Prefix each group with a short heading using `---` delimiters.

---

## Internal documentation

- **Variables.** One-line comment per custom property.
- **Components.** One-line description of role and `data-*` attributes before each block.
- **Non-obvious rules.** Single-line comment where intent is unclear.
