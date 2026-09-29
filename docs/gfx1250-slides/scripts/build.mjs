import {getInstalledBrowsers} from '@puppeteer/browsers';
import {spawnSync} from 'node:child_process';
import {mkdirSync, readFileSync, writeFileSync} from 'node:fs';
import {fileURLToPath} from 'node:url';

const root = fileURLToPath(new URL('..', import.meta.url));
process.chdir(root);
mkdirSync('dist', {recursive : true});

const env = {...process.env};
if (!env.CHROME_PATH) {
  const browsers =
      await getInstalledBrowsers({cacheDir : `${root}/.cache/browser`});
  const local =
      browsers.find((browser) => browser.browser === 'chrome-headless-shell');
  if (local)
    env.CHROME_PATH = local.executablePath;
}

const cli = `${root}/node_modules/@marp-team/marp-cli/marp-cli.js`;
const common = [
  cli, 'slides.md', '--theme-set', 'theme.css', '--html', '--allow-local-files'
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
        'See README.md for browser installation and host sandbox options.');
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
