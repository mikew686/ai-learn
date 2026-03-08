# CSS Documentation Guide

---

## File headers

Each CSS file begins with a brief block comment: file name, one-sentence description, purpose.

---

## Section comments

Group related rules. Prefix each group with a short heading using `---` delimiters. Keep headings scannable.

---

## Internal documentation

- **Variables.** One-line comment per custom property describing scope (e.g. text colour, spacing unit).
- **Components.** One-line description of role and applicable `data-*` attributes before each component block.
- **Non-obvious rules.** Single-line comment where intent is not clear from the selector or property.
- Avoid restating the property name; describe intent only.

---

## File roles

| File           | Header documents                         | Body documents                    |
|----------------|------------------------------------------|-----------------------------------|
| variables.css  | Design tokens; scope                     | Variable groups                   |
| layout.css     | Page structure; desktop-first approach   | Layout blocks                     |
| components.css | Reusable UI elements                     | Components and variants           |
| utilities.css  | Helper classes                           | Utility definitions               |
| site.css       | Overrides and exceptions                 | Override blocks                   |
