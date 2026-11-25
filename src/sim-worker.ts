/// <reference lib="webworker" />
// Offline mode: runs the authoritative Rust server inside a Web Worker so the
// client code path is identical to online play.

interface SrvExports {
  memory: WebAssembly.Memory;
  in_ptr(): number;
  srv_init(seed: number, slots: number, flags: number, s0: number, s1: number, s2: number, s3: number): void;
  srv_add(len: number): number;
  srv_msg(pid: number, len: number): void;
  srv_step(behind: number): number;
  srv_welcome(pid: number): number;
  srv_snap(pid: number): number;
  srv_out(): number;
  srv_flush(): void;
}

const ctx = self as unknown as DedicatedWorkerGlobalScope;
let ex: SrvExports | null = null;
let pid = -1;
let paused = false;
let next = 0;
let running = false;
const TICK_MS = 1000 / 30;

function out(len: number) {
  if (!ex || len <= 0) return;
  const copy = new Uint8Array(ex.memory.buffer, ex.srv_out(), len).slice();
  ctx.postMessage(copy.buffer, [copy.buffer]);
}

function tick(behind: boolean) {
  if (!ex) return;
  const r = ex.srv_step(behind ? 1 : 0);
  if (r & 1) {
    out(ex.srv_snap(pid));
    ex.srv_flush();
  }
  if (r & 2) {
    running = false;
    ctx.postMessage({ t: 'ended' });
  }
}

function loop() {
  if (!running) return;
  const now = performance.now();
  if (paused) {
    next = now + TICK_MS;
  } else {
    let n = 0;
    while (now >= next && n < 6) {
      // catching up: at least one more tick is already due after this one
      tick(now - next >= TICK_MS);
      next += TICK_MS;
      n++;
    }
    if (now - next > 250) next = now + TICK_MS;
  }
  setTimeout(loop, Math.max(1, next - performance.now()));
}

ctx.onmessage = async (e: MessageEvent) => {
  const m = e.data;
  try {
    if (m.t === 'init') {
      const inst = await WebAssembly.instantiate(m.module as WebAssembly.Module, {});
      ex = inst.exports as unknown as SrvExports;
      const sec = crypto.getRandomValues(new Uint32Array(4));
      ex.srv_init(m.seed >>> 0, m.slots | 0, 1, sec[0], sec[1], sec[2], sec[3]);
      const name = new TextEncoder().encode(String(m.name)).slice(0, 48);
      new Uint8Array(ex.memory.buffer, ex.in_ptr(), name.length).set(name);
      pid = ex.srv_add(name.length);
      if (pid < 0) throw new Error('could not join local match');
      out(ex.srv_welcome(pid));
      running = true;
      next = performance.now() + TICK_MS;
      loop();
    } else if (m.t === 'msg' && ex) {
      const d = new Uint8Array(m.data as ArrayBuffer);
      if (d.length === 0 || d.length > 1024) return;
      if (d[0] === 3) {
        const pong = d.slice();
        pong[0] = 0x83;
        ctx.postMessage(pong.buffer, [pong.buffer]);
        return;
      }
      new Uint8Array(ex.memory.buffer, ex.in_ptr(), d.length).set(d);
      ex.srv_msg(pid, d.length);
    } else if (m.t === 'pause') {
      paused = !!m.v;
    }
  } catch (err) {
    ctx.postMessage({ t: 'error', msg: String(err) });
  }
};
