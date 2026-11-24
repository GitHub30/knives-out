// Copies the compiled simulation core to the places that bundle it.
import { copyFileSync, mkdirSync, statSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const src = resolve(root, 'core/target/wasm32-unknown-unknown/release/zonefall_core.wasm');
// src/core.wasm is imported with ?url, so Vite fingerprints it: after a
// deploy, a reload can never pair the new client with a cached old core.
for (const dst of ['src/core.wasm', 'server/src/core.wasm']) {
  const out = resolve(root, dst);
  mkdirSync(dirname(out), { recursive: true });
  copyFileSync(src, out);
}
console.log(`core.wasm: ${(statSync(src).size / 1024).toFixed(1)} KiB`);
