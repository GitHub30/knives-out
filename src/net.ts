// Transports: WebSocket (online, Cloudflare Durable Object) and Worker (offline local server).

export interface Transport {
  send(data: Uint8Array<ArrayBuffer>): void;
  close(): void;
  setPaused?(paused: boolean): void;
  onMessage: (data: ArrayBuffer) => void;
  onClose: (reason: string) => void;
  readonly kind: 'online' | 'offline';
}

export const DEFAULT_SERVER = 'wss://zonefall-server.6u.workers.dev';

export function serverBase(): string {
  const q = new URLSearchParams(location.search).get('server');
  if (q && /^wss?:\/\/[a-z0-9.:-]+$/i.test(q)) return q;
  if (location.hostname === 'localhost' || location.hostname === '127.0.0.1') {
    const dev = new URLSearchParams(location.search).get('dev');
    if (dev) return `ws://${location.hostname}:8787`;
  }
  return DEFAULT_SERVER;
}

export class WsTransport implements Transport {
  readonly kind = 'online' as const;
  onMessage: (data: ArrayBuffer) => void = () => {};
  onClose: (reason: string) => void = () => {};
  private ws: WebSocket;
  private closed = false;

  constructor(url: string) {
    this.ws = new WebSocket(url);
    this.ws.binaryType = 'arraybuffer';
    this.ws.onmessage = (e) => {
      if (e.data instanceof ArrayBuffer) this.onMessage(e.data);
    };
    this.ws.onclose = (e) => this.finish(e.reason || `切断されました (${e.code})`);
    this.ws.onerror = () => this.finish('サーバーに接続できませんでした');
  }

  opened(): Promise<void> {
    return new Promise((resolve, reject) => {
      if (this.ws.readyState === WebSocket.OPEN) return resolve();
      const t = setTimeout(() => reject(new Error('接続がタイムアウトしました')), 10000);
      this.ws.addEventListener('open', () => {
        clearTimeout(t);
        resolve();
      });
      this.ws.addEventListener('close', () => {
        clearTimeout(t);
        reject(new Error('サーバーに接続できませんでした'));
      });
    });
  }

  private finish(reason: string) {
    if (this.closed) return;
    this.closed = true;
    this.onClose(reason);
  }

  send(data: Uint8Array<ArrayBuffer>) {
    if (this.ws.readyState === WebSocket.OPEN) this.ws.send(data);
  }

  close() {
    this.closed = true;
    try {
      this.ws.close(1000, 'bye');
    } catch {
      /* ignore */
    }
  }
}

export class WorkerTransport implements Transport {
  readonly kind = 'offline' as const;
  onMessage: (data: ArrayBuffer) => void = () => {};
  onClose: (reason: string) => void = () => {};
  private worker: Worker;

  constructor(module: WebAssembly.Module, seed: number, slots: number, name: string) {
    this.worker = new Worker(new URL('./sim-worker.ts', import.meta.url), { type: 'module' });
    this.worker.onmessage = (e: MessageEvent) => {
      if (e.data instanceof ArrayBuffer) this.onMessage(e.data);
      else if (e.data && e.data.t === 'error') this.onClose(String(e.data.msg));
      else if (e.data && e.data.t === 'ended') this.onClose('ended');
    };
    this.worker.onerror = (e) => this.onClose(`ローカルサーバーエラー: ${e.message}`);
    this.worker.postMessage({ t: 'init', module, seed, slots, name });
  }

  send(data: Uint8Array<ArrayBuffer>) {
    const copy = data.slice().buffer;
    this.worker.postMessage({ t: 'msg', data: copy }, [copy]);
  }

  setPaused(paused: boolean) {
    this.worker.postMessage({ t: 'pause', v: paused });
  }

  close() {
    this.worker.terminate();
  }
}
