// @ts-check
import { defineConfig } from 'astro/config';
import relativeLinks from 'astro-relative-links';

// Pure static output. `build.format: 'directory'` emits `/persons/page/1/index.html`
// style folders, and `astro-relative-links` rewrites every absolute href/src into a
// relative one so the exported `dist/` opens straight off the filesystem (file://)
// without a web server. The only feature that still needs a server is the search box
// (it fetch()es search-index.json — file:// blocks fetch), which degrades gracefully.
export default defineConfig({
  output: 'static',
  trailingSlash: 'always',
  build: {
    format: 'directory',
    inlineStylesheets: 'always',
  },
  integrations: [relativeLinks()],
  // No site URL on purpose: relative links make the bundle origin-independent.
  devToolbar: { enabled: false },
});
