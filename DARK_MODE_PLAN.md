# Dark Mode — Polish Pass Plan

Status: **DM3 done ✅ · DM6 done ✅ · DM5 done ✅ · palette pass done ✅** · DM4/DM2/DM1 pending · Owner: clpatel · Started: 2026-06-14
Branch: `feat/dark-mode-polish` (DM4/DM2/DM1 land here — Branching updated post-merge)

**Session (2026-06-15):** DM3 + DM6 + palette + DM5 merged to main (ca50213). Started feat/dark-mode-polish for final three items.

This plan is the §7 follow-through from [UI_UX_PLAN.md](UI_UX_PLAN.md). §7 introduced
the theme token dict and the toggle; this pass fixes the six visual defects that
remain once dark mode is the default surface. Every item is small, scoped to the
render layer, and shares the same token dict — so they are interdependent polish,
not isolated features.

---

## Branching decision

**One branch, one PR, per-item commits.** All six changes touch the same two
render files (`render_helpers.py`, `render_ui.py`) and the same `THEME` token dict
introduced in §7. They are visual polish on a single surface that only makes sense
reviewed together — splitting them into six branches buys no isolation and costs
merge-conflict churn (every branch would edit overlapping lines in
`section_container_html` / `create_grid_display`). Keep the existing discipline:

- One **commit** per item (reviewable, revertable, mirrors the §-per-test pattern).
- Extend `tests/test_dark_theme.py` per item where the change emits assertable
  tokens/strings (smoke-test the HTML the same way §1–§6 did).
- `pytest -q` green before each commit; single PR off `feat/dark-theme-toggle` at
  the end.

---

## Optimized request (canonical spec)

The six items below are the source of truth. JSON restatement of the original
request, for unambiguous implementation:

```json
{
  "branch": "feat/dark-theme-toggle",
  "items": [
    {
      "id": "DM1",
      "title": "Remove sidebar nav, move theme toggle beside main title",
      "file": "render_ui.py",
      "current": "Toggle button lives in st.sidebar (lines 182-188); sidebar reads as an unwanted nav rail.",
      "target": "Drop the sidebar block. Render the 🌙/☀️ toggle inline, immediately left of st.title('Stock Price Predictor') (line 190), e.g. st.columns([_, _]) with the icon button in a narrow first column.",
      "accept": "No sidebar rendered; toggle sits on the title row and still flips st.session_state.theme + reruns."
    },
    {
      "id": "DM2",
      "title": "Subtitle 'Displaying Predictions for …' matches delta-from-close color",
      "file": "display_market_status.py",
      "current": "Subtitle hardcoded color #555555 (line 44).",
      "target": "Drive subtitle color from the theme's muted delta-label token (THEME[theme]['text_delta_label'] = #64748b dark). Thread `theme` into generate_market_status_header() and display_market_status().",
      "accept": "Subtitle color === text_delta_label for the active theme; no hardcoded hex."
    },
    {
      "id": "DM3",
      "title": "Main ticker section renders as a real grouping card, not a top bar",
      "files": ["render_helpers.py", "render_ui.py"],
      "current": "section_container_html() emits a lone opening <div> (helpers 335-342); content is in separate st.* calls; close </div> orphaned (render_ui 121). Streamlit auto-closes the div per-block, so only an empty bar paints; the chart iframe can't nest in a string div regardless.",
      "target": "Replace the string-div wrapper with a native st.container(border=True) given a stable key, then inject scoped CSS (background = section_bg, padding ~18px, radius 20px, border = section_border) targeting that container via :has()/key selector. All ticker content (header, cards, chart, grid) goes inside the `with` block so it's visually grouped.",
      "accept": "Visible filled card surrounds the entire ticker block with even padding on all four edges; not a thin top bar."
    },
    {
      "id": "DM4",
      "title": "Chart title tile gets visible border + card shading, full width",
      "file": "render_helpers.py",
      "current": "generate_chart_widget_html header div uses a left→right gradient (linear-gradient(90deg,…)) with only a bottom border (line 212), so left/right/top edges vanish against the section bg.",
      "target": "Use the prediction-card treatment: background = t['card_bg'] (vertical gradient), full `1px solid t['card_border']` on all sides, t['card_shadow']. Match the card radius and span the chart's full width.",
      "accept": "Chart title tile has a visible border on all four sides and the same shading/shadow as the LR/XGBoost prediction cards."
    },
    {
      "id": "DM5",
      "title": "Stats grid splits 50-50 across two rows, centered to chart",
      "file": "render_helpers.py",
      "current": "create_grid_display uses grid-template-columns: repeat(auto-fit, minmax(145px,1fr)) (lines 327-328); six items pack 4+2 unevenly.",
      "target": "Force three equal columns: repeat(3, 1fr) → 6 items render 3+3 (two equal rows). Narrow-viewport fallback repeat(2, 1fr) (3 rows of 2) via the section min-width. Grid centered under the chart.",
      "accept": "Six stat tiles always split evenly across two rows (3+3), equal widths, centered to the chart."
    },
    {
      "id": "DM6",
      "title": "Next-day prediction section gets the same grouping card",
      "files": ["render_helpers.py", "render_ui.py"],
      "current": "Next-day block (render_ui 152-160) already calls section_container_html, so it inherits the DM3 broken-bar defect.",
      "target": "Apply the DM3 native-container fix to the next-day per-ticker block too — each next-day ticker section gets the same filled grouping card holding its header + model cards.",
      "accept": "Each next-day ticker section is wrapped in the same visible grouping card as the main section."
    }
  ]
}
```

---

## Per-item detail

### DM1 — Toggle beside title, kill the sidebar
[render_ui.py:182-188](render_ui.py#L182-L188) puts the button in `st.sidebar`.
That sidebar *is* the "nav bar" the screenshot flags. Remove the `with st.sidebar:`
block; render the icon inline on the title row. Cleanest: `col_icon, col_title =
st.columns([1, 12])`, button in `col_icon`, `st.title` in `col_title`. Keep the
flip logic (`st.session_state.theme` swap + `st.rerun()`) verbatim.

### DM2 — Subtitle color = delta-label token
[display_market_status.py:44](display_market_status.py#L44) hardcodes `#555555`.
The "Δ from close" muted label in the cards uses `text_delta_label` (`#64748b`
dark / `#94a3b8` light — [render_helpers.py:14](render_helpers.py#L14)). Thread the
active theme into `display_market_status()` (it already has access via
`st.session_state`) and `generate_market_status_header()`, and set the subtitle
`color` from that token. Removes the only off-palette grey in the header.

> Confirmed: match the `text_delta_label` muted color **only** — not the green/red
> up/down value color. Subtitle color is a fixed theme token, never per-delta.

### DM3 — Real grouping card (the structural one) ✅
Landed at `4e7af7b` on `feat/dark-theme-toggle`. `render_section_container()` helper + scoped `.st-key-` CSS replaces broken `section_container_html()`.
This is the root defect behind the "card is just a bar at the top" complaint.
**Cause:** Streamlit renders each `st.markdown`/`components.html` in its own DOM
block and sanitizes unbalanced tags, so the lone opening `<div>` from
[render_helpers.py:335](render_helpers.py#L335) is force-closed immediately — the
section bg paints as an empty strip, and the manual `</div>` at
[render_ui.py:121](render_ui.py#L121) is orphaned. A string-concat wrapper *cannot*
contain the chart, which is a `components.html` iframe.

**Fix:** use a native `st.container(border=True)` with a stable `key`, wrap all
ticker content in its `with` block, and style it with one scoped CSS rule
(`st.markdown` `<style>` targeting the container's key via `:has()` /
`[data-testid] :has(.<key>)`), setting `background: section_bg`, `padding`,
`border-radius`, `border: section_border`. Retire `section_container_html` and the
orphaned close tag.

### DM4 — Chart tile border + shading
[render_helpers.py:212](render_helpers.py#L212): swap the 90deg gradient + bottom-
only border for `card_bg` + full `card_border` + `card_shadow`, matching the
prediction cards. The chart body below it ([render_helpers.py:215](render_helpers.py#L215))
keeps `chart_bg`; only the title tile changes.

### DM5 — 50-50 stats grid
[render_helpers.py:327-328](render_helpers.py#L327-L328): `auto-fit` is what
produces the uneven 4+2. Six metrics → `repeat(3, 1fr)` gives a clean 3+3. Keep a
narrow fallback to `repeat(2, 1fr)` (2+2+2) if the section gets tight.

### DM6 — Next-day grouping card
[render_ui.py:152-160](render_ui.py#L152-L160) already wraps with the broken
`section_container_html`. Once DM3 lands as a reusable native-container helper,
apply the same helper here so each next-day ticker gets an identical filled card.

---

## Implementation order
Dependency: only **DM6 → DM3** (DM6 reuses DM3's container helper). **DM1, DM2,
DM4, DM5 are fully independent** — each touches its own lines, no shared dependency,
can land in any order.

1. **DM3** first — structural container fix DM6 depends on; changes how content
   mounts. Build the reusable native-container helper here.
2. **DM6** — apply the DM3 helper to the next-day section (now trivial).
3. **DM4**, **DM5** — independent CSS tweaks inside the now-correct container.
4. **DM2** — header subtitle token threading (independent).
5. **DM1** — toggle relocation (independent; clean visual cap last).

## Testing
Extend [tests/test_dark_theme.py](tests/test_dark_theme.py): assert the chart tile
HTML carries `card_border`/`card_bg` (DM4), the grid HTML carries
`repeat(3, 1fr)` (DM5), the subtitle HTML carries the `text_delta_label` token
(DM2). DM1/DM3/DM6 are layout — smoke-test the emitted container markup and
**manually verify in the live app** (per the standing rule) before each commit.
```

(MARKET_NOW override for forcing closed/weekend states — see §0 of UI_UX_PLAN.md.)
