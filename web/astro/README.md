# Face Gallery — Astro static export

Scalable static-site export for the local AI face recognizer — a **paginated,
lazy-loading, thumbnail-based** site that stays fast with thousands of
persons/photos. Pure SSG — no backend after the build. This is the only HTML
gallery export; the older single-page `export_html` generator was removed.

## How it works

Two layers:

1. **Python data + image exporter** — `ExportService.export_astro()` in
   `app/services/export_service.py`. Reads the SQLite DB and writes a *bundle*:
   - JSON into `src/data/` (build-time only — never shipped to the browser):
     `manifest.json`, `persons.json`, `photos.json`, `map-data.json`,
     `slideshow-data.json`.
   - Three image variants per photo into `public/assets/images/`
     (`thumbs/` ~320px, `medium/` ~1280px, `original/` full).
   - The minimal `search-index.json` into `public/assets/data/` (served at
     runtime for the debounced search box).
   - Standalone parity pages (`map.html`, `slideshow.html`, optional
     `collage_index.html`) into `public/`, reusing the existing generators.

2. **Astro SSG** — `getStaticPaths` runs at build time (in Node), so it can read
   the *entire* dataset and bake each list page (60 items) and each detail page
   into its own static HTML. The browser only ever loads one page's markup plus
   thumbnails (lazy-loaded). See `src/lib/data.ts` for the loaders/pagination
   and `src/pages/` for the routes.

`astro-relative-links` rewrites absolute URLs to page-relative ones so the
finished `dist/` opens straight off the filesystem (`file://`). The only feature
that needs a server is the search box (it `fetch()`es the index — `file://`
blocks fetch); it degrades gracefully with a hint.

## Build

The Python exporter normally runs the build for you (Export dialog →
*"Gyors statikus weboldal (Astro)"*), which does `npm install` (first run) +
`npm run build` and copies `dist/` to your chosen folder.

Manual build (for development), after a bundle has been generated into
`src/data/` + `public/assets/`:

```sh
cd web/astro
npm install      # first time only
npm run build    # → dist/
```

Generated files (`src/data/*.json`, `public/assets/`, `public/*.html`, `dist/`,
`node_modules/`) are git-ignored.
