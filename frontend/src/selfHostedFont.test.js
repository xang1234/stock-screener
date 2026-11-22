import { existsSync, readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';

// Paths go through a variable: Vite rewrites a literal
// `new URL('../public/…', import.meta.url)` to the file's public URL.
const local = (path) => new URL(path, import.meta.url);
const read = (path) => readFileSync(local(path), 'utf8');

const css = read('./index.css');
const faces = (css.match(/@font-face\s*{[^}]*}/g) ?? []).map((block) => ({
  block,
  src: block.match(/url\(\s*['"]?([^'")]+)['"]?\s*\)/)?.[1],
  display: block.match(/font-display:\s*([a-z]+)/)?.[1],
  ranges: (block.match(/unicode-range:\s*([^;]+);/)?.[1] ?? '').split(',').map((part) => {
    const [lo, hi = lo] = part.trim().replace(/^U\+/, '').split('-');
    return [parseInt(lo, 16), parseInt(hi, 16)];
  }),
}));
const covers = (face, codePoint) => face.ranges.some(([lo, hi]) => codePoint >= lo && codePoint <= hi);
const latin = faces.find((face) => covers(face, 'A'.codePointAt(0)));

describe('the Inter font', () => {
  it('is not fetched from a third-party stylesheet', () => {
    // A CSS @import blocks first paint until another origin answers.
    expect(css).not.toMatch(/@import\s+url\(\s*['"]?https?:/);
  });

  it('comes from bundled files', () => {
    expect(faces.length).toBeGreaterThan(0);
    for (const face of faces) {
      expect(face.block).toMatch(/font-family:\s*'Inter'/);
      expect(face.src).not.toMatch(/^https?:/);
      expect(existsSync(local(face.src))).toBe(true);
    }
  });

  it.each(['β', 'Δ', '₹', '₩', '€', '£', '¥'])('covers %s, which the UI writes in labels and prices', (char) => {
    expect(faces.some((face) => covers(face, char.codePointAt(0)))).toBe(true);
  });

  it('keeps the weights Google served, so 800 still renders as 700', () => {
    for (const face of faces) expect(face.block).toMatch(/font-weight:\s*400 700;/);
  });

  it('never reflows body text, and swaps in only the rarer subsets', () => {
    // Latin is preloaded, so it is ready by first paint. The others are only
    // requested once a page draws one of their characters, too late for
    // `optional`, which would keep the system font for the whole visit.
    expect(latin.display).toBe('optional');
    for (const face of faces.filter((face) => face !== latin)) expect(face.display).toBe('swap');
  });

  it('preloads exactly the latin file, fetched the way the font is', () => {
    const links = read('../index.html').match(/<link[^>]*rel="preload"[^>]*as="font"[^>]*>/g) ?? [];
    expect(links).toHaveLength(1);
    const [link] = links;
    expect(new URL(link.match(/href="([^"]+)"/)[1], 'http://x/').pathname)
      .toBe(new URL(latin.src, 'http://x/src/index.css').pathname);
    expect(link).toMatch(/type="font\/woff2"/);
    // Fonts load in anonymous CORS mode; any other mode leaves the preload unused.
    expect(link).toMatch(/\scrossorigin(="anonymous")?[\s/>]/);
  });

  it('ships its licence with the app', () => {
    expect(existsSync(local('../public/fonts/Inter-OFL.txt'))).toBe(true);
  });

  it('is cached as immutable like every other hashed asset', () => {
    // The font lands in /assets/ with a content hash; without ^~ nginx picks
    // the later woff2 regex block (30 days, no immutable, no nosniff).
    expect(read('../nginx.conf')).toMatch(/location \^~ \/assets\/ \{[^}]*immutable/);
  });

  it('needs no Google origins in the content security policy', () => {
    const policy = read('../nginx-security-headers.conf').match(/Content-Security-Policy "([^"]+)"/)[1];
    expect(policy).not.toMatch(/fonts\.googleapis\.com|fonts\.gstatic\.com/);
  });
});

describe('nginx security headers', () => {
  // Any add_header in a location drops every server-level add_header, so a
  // block that sets its own must include the security headers again.
  const conf = read('../nginx.conf');
  const include = 'include /etc/nginx/snippets/security-headers.conf;';
  const locations = [...conf.matchAll(/\n\s*location ([^{]+)\{([^}]*)\}/g)];

  it('are set at server level from the snippet the image installs', () => {
    expect(conf.split(/\n\s*location /)[0]).toContain(include);
    expect(read('../Dockerfile')).toContain('nginx-security-headers.conf /etc/nginx/snippets/security-headers.conf');
    expect(read('../nginx-security-headers.conf')).toMatch(/add_header Content-Security-Policy .* always;/);
    expect(locations.length).toBeGreaterThan(3);
  });

  it.each(locations.map((m) => [m[1].trim(), m[2]]))('%s keeps them', (_, body) => {
    if (/add_header/.test(body)) expect(body).toContain(include);
  });
});
