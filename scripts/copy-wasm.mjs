// Copies the compiled simulation core to the places that bundle it.
import { copyFileSync, mkdirSync, statSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const src = resolve(root, 'core/target/wasm32-unknown-unknown/release/zonefall_core.wasm');
for (const dst of ['public/core.wasm', 'server/src/core.wasm']) {
  const out = resolve(root, dst);
  mkdirSync(dirname(out), { recursive: true });
  copyFileSync(src, out);
}
console.log(`core.wasm: ${(statSync(src).size / 1024).toFixed(1)} KiB`);
