# Modeling Agents with Probabilistic Programs

Source for [agentmodels.org](https://agentmodels.org), an interactive online book on modeling rational and biased agents for (PO)MDPs and reinforcement learning using probabilistic programs. Code examples run directly in the browser via [WebPPL](http://webppl.org).

By Owain Evans, Andreas Stuhlmüller, John Salvatier, and Daniel Filan.

## Local development

With Docker (no Ruby setup needed):

~~~~
docker compose up
~~~~

Or with a local Ruby (3.x):

~~~~
bundle install
bundle exec jekyll serve
~~~~

Either way, the site is served at <http://localhost:4000>.

## How the site works

- **Chapters** live in `chapters/*.md` (Jekyll/kramdown Markdown, front matter sets title, description, and chapter order via filename). Pages with `status: stub` appear unlinked in the table of contents.
- **Code boxes**: every `~~~~` fenced block is turned into an editable, runnable WebPPL editor in the browser (`wpEditor.setup` in `assets/js/main.js`).
- **Math** is written as kramdown `$$...$$` and rendered client-side with KaTeX (auto-render finds the `\(...\)` / `\[...\]` delimiters kramdown emits).
- **Citations**: markers like `refp:key`, `reft:key`, and `cite:key` in the text are resolved client-side against `bibliography.bib`.
- **Vendored assets**: all third-party JS/CSS/fonts (jQuery, KaTeX, Pure, Merriweather, the WebPPL editor and viz bundles) are checked in under `assets/vendor/` — the site has no runtime dependencies on external CDNs.

## The WebPPL bundle (intentionally frozen)

`assets/js/webppl.min.js` is a 2019 build of WebPPL bundled with the packages the book uses (`webppl-agents`, `webppl-dp`, `webppl-timeit`, `webppl-nn`). WebPPL is no longer actively maintained, so the bundle is **frozen on purpose**: the goal is keeping the book's examples working, not tracking upstream.

If you ever need to regenerate it, the historical recipe is in `scripts/update-webppl` (uses the dependencies pinned in `package.json`); expect to need an old Node version.

## Deployment

The site is published with GitHub Pages from the `gh-pages` branch — pushing to `gh-pages` deploys. The `Gemfile` pins the [`github-pages`](https://pages.github.com/versions/) gem so local builds match production. CI (GitHub Actions) builds the site and checks internal links on every push.

## Repo layout

~~~~
chapters/        Book chapters (Markdown)
index.md         Landing page and table of contents
bibliography.bib Citations, rendered client-side
assets/css       Site stylesheet
assets/js        Site JS + frozen WebPPL bundle
assets/vendor    Vendored third-party assets
assets/img       Figures
assets/tex       LaTeX sources for some figures
_layouts, _includes  Jekyll templates
notes/, scratch/ Old planning notes (not published)
scripts/         Historical build scripts for the WebPPL bundle
~~~~
