# Interactive report page plan

Goal: one static HTML page on GitHub Pages holds the whole report. The README
becomes a short summary and run instructions.

## Decisions

- Plots use Plotly.js, loaded from a CDN. The page embeds the data as JSON.
- No PNG figures. Remove Matplotlib and the `figures/` folder.
- No Sphinx API docs. Remove `docs/`, the `docs` dependency group, and the old
  docs workflow.
- The page comes from the same code that made the README results, so the
  numbers cannot drift.

## Steps

| Step | Work                                                                                      | Status |
| ---- | ----------------------------------------------------------------------------------------- | ------ |
| 1    | `plots.py`: build JSON specs for the facet views, row scaling, profiles, real world       | done   |
| 2    | `content/`: move the static README sections (methods, limits, terms) into the package      | done   |
| 3    | `site.py`: render Markdown to HTML with figure placeholders, fill a Jinja template        | done   |
| 4    | `site.js`: draw the specs with Plotly, add filters (backend, layout, profile, operation)  | done   |
| 5    | Real-world calculator: sliders for file size, uses, price, and download speed             | done   |
| 6    | CLI: `report` writes `site/index.html`, drop figure and README options                    | done   |
| 7    | Cut the README down, remove Matplotlib code, `figures/`, Sphinx, and old tests            | done   |
| 8    | Workflow: build the page and deploy it to GitHub Pages                                    | done   |

## Checks

- Python tests cover the specs and the HTML structure (headings, embedded JSON,
  placeholders, links).
- `node --check` covers the JavaScript syntax when Node is present.
- A person clicks through the page once, because the tests cannot run a browser.

## What changed

- `plots.py` builds the JSON specs. `site.py` builds the page. `content/` holds
  the method text. `site/` holds the template, style, and script.
- `report` writes `site/index.html`, and no longer writes PNG files or edits the
  README.
- The README is now a short summary and the run instructions.
- `publish-site.yml` builds the page and deploys it to the `pages` branch.

## Still to do by hand

- Click through the page in a browser: filters, the explorer, and the
  real-world calculator. The tests check structure and syntax, not clicks.
- Check that GitHub Pages serves the `pages` branch.
- Confirm the Fleming and Wallace DOI opens.
