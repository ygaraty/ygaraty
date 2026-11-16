// Compile in a temporary directory and compare every generated module.
import {execFileSync} from 'node:child_process';
import {mkdtempSync, readFileSync, readdirSync, rmSync} from 'node:fs';
import {tmpdir} from 'node:os';
import {join, dirname} from 'node:path';
import {fileURLToPath} from 'node:url';
import {strictEqual} from 'node:assert';

const root = dirname(dirname(dirname(fileURLToPath(import.meta.url))));
const output = mkdtempSync(join(tmpdir(), 'ygaraty-public-ts-'));
try {
  execFileSync('tsc', ['--project', join(root, 'tsconfig.json'), '--outDir', output], {stdio: 'inherit'});
  for (const name of readdirSync(output)) {
    strictEqual(readFileSync(join(output, name), 'utf8'),
      readFileSync(join(root, 'ygaraty/backends/webgpu/browser', name), 'utf8'), `Stale generated module: ${name}`);
  }
} finally { rmSync(output, {recursive: true, force: true}); }
