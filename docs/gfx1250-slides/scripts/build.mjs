import {spawnSync} from 'node:child_process';
import {mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync} from 'node:fs';
import {homedir} from 'node:os';
import {join} from 'node:path';
import {fileURLToPath} from 'node:url';

const root = fileURLToPath(new URL('..', import.meta.url));
process.chdir(root);
mkdirSync('dist', {recursive : true});

const browserTmp = mkdtempSync(join(homedir(), 'gfx1250-marp-'));
const env = {...process.env, TMPDIR: browserTmp};
process.on('exit', () => rmSync(browserTmp, {recursive: true, force: true}));

const cli = `${root}/node_modules/@marp-team/marp-cli/marp-cli.js`;
const common = [
  cli, 'slides.md', '--theme-set', 'theme.css', '--html', '--allow-local-files',
  '--browser', 'firefox'
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
    console.error(
        'See README.md for Bun and Firefox requirements.');
    process.exit(result.status ?? 1);
  }
  if (format === 'html') {
    const output = 'dist/gfx1250-draft.html';
    const html = readFileSync(output, 'utf8')
                     .replace(/src="(assets\/[a-z0-9-]+\.svg)"/g, (_, path) => {
                       return `src="data:image/svg+xml;base64,${
                           readFileSync(path).toString('base64')}"`;
                     });
    writeFileSync(output, html);
  }
}
