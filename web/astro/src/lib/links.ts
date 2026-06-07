// Make directory-style links point explicitly at their index.html.
//
// With build.format: 'directory' every page is emitted as `<route>/index.html`.
// A web server resolves `/photos/page/1/` to that index.html, but the
// filesystem (file://) does not — it shows a directory listing instead. So the
// exported bundle, which must open by double-clicking, needs every internal
// navigation target to name the actual file. astro-relative-links then rewrites
// these absolute paths to page-relative ones.
export function link(path: string): string {
  if (!path) return path;
  // Already an external URL or a fragment-only link.
  if (/^[a-z]+:/i.test(path) || path.startsWith('#')) return path;
  const [base, hash] = path.split('#');
  // Already names a file (its last segment has an extension, e.g. map.html).
  const lastSeg = base.replace(/\/$/, '').split('/').pop() || '';
  if (lastSeg.includes('.')) return path;
  const withIndex = base.endsWith('/') ? base + 'index.html' : base + '/index.html';
  return hash ? `${withIndex}#${hash}` : withIndex;
}
