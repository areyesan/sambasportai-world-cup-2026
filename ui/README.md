# Interactive UIs

Open the HTML files directly in a browser, or serve the folder locally:

```bash
python -m http.server 8000 --directory ui
```

Then open `http://localhost:8000`.

## Files

- `group_stage_review_and_round32_predictions.html`: evaluation dashboard and Round-of-32 predictions.
- `pre_tournament_prediction_center.html`: frozen pre-tournament v8 dashboard.
- `group_stage_matchday_1.html`
- `group_stage_matchday_2.html`
- `group_stage_matchday_3.html`

The files are self-contained except for country flags, which use a public flag image endpoint.

## GitHub Pages

The included `.github/workflows/pages.yml` publishes this directory. Use `ui/index.html` as the landing page.
