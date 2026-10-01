import {spawnSync} from 'node:child_process';
import {
  mkdirSync,
  mkdtempSync,
  readFileSync,
  rmSync,
  writeFileSync
} from 'node:fs';
import {homedir} from 'node:os';
import {extname, join} from 'node:path';
import {fileURLToPath} from 'node:url';

const root = fileURLToPath(new URL('..', import.meta.url));
process.chdir(root);
mkdirSync('dist', {recursive : true});

const browserTmp = mkdtempSync(join(homedir(), 'gfx1250-marp-'));
const env = {
  ...process.env,
  TMPDIR : browserTmp
};
process.on('exit', () => rmSync(browserTmp, {recursive : true, force : true}));

const mimeTypes = {
  '.jpg' : 'image/jpeg',
  '.png' : 'image/png',
  '.svg' : 'image/svg+xml',
  '.woff' : 'font/woff',
};

// Embed the theme assets for both exports, including fonts on hosts without
// Arial.
function assetUrl(path) {
  const mime = mimeTypes[extname(path)];
  if (!mime)
    throw new Error(`Unsupported slide asset: ${path}`);
  return `data:${mime};base64,${
      readFileSync(join(root, path)).toString('base64')}`;
}

function embedCssAssets(css) {
  return css.replace(/url\((['"]?)(assets\/[^'"\)]+)\1\)/g,
                     (_, _quote, path) => `url("${assetUrl(path)}")`);
}

const theme = embedCssAssets(readFileSync('theme.css', 'utf8'));
const fontRules = (theme.match(/@font-face\s*\{[^}]*\}/g) ?? []).join('\n');
const exportTheme = join(browserTmp, 'theme.css');
writeFileSync(exportTheme, theme);

// SVG images cannot inherit the document's fonts. Embed the same font faces
// inside each image so diagrams also survive moving the exported HTML/PDF.
function diagramUrl(path) {
  const svg = readFileSync(join(root, path), 'utf8')
                  .replace(/(<svg\b[^>]*>)/, `$1<style>${fontRules}</style>`);
  return `data:image/svg+xml;base64,${Buffer.from(svg).toString('base64')}`;
}

const source = readFileSync('slides.md', 'utf8')
                   .replace(/src="(assets\/[a-z0-9-]+\.svg)"/g,
                            (_, path) => `src="${diagramUrl(path)}"`);
const exportSource = join(browserTmp, 'slides.md');
writeFileSync(exportSource, source);

const cli = `${root}/node_modules/@marp-team/marp-cli/marp-cli.js`;
const common = [
  cli, exportSource, '--theme-set', exportTheme, '--html',
  '--allow-local-files', '--browser', 'firefox'
];
for (const [format, flags] of [[ 'html', [] ],
                               [ 'pdf', [ '--pdf', '--pdf-outlines' ] ]]) {
  const result =
      spawnSync(process.execPath,
                [...common, ...flags, '-o', `dist/gfx1250-draft.${format}` ], {
                  stdio : 'inherit',
                  env,
                });
  if (result.error)
    throw result.error;
  if (result.status !== 0) {
    console.error('See README.md for Bun and Firefox requirements.');
    process.exit(result.status ?? 1);
  }
}
