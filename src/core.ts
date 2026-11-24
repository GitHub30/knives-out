// Thin typed wrapper around the Rust/Wasm simulation core (client-side API).

export interface WDef {
  ammo: number;
  dmg: number;
  interval: number;
  mag: number;
  reload: number;
  hip: number;
  ads: number;
  pellets: number;
  range: number;
  speed: number;
  zoom: number;
  recoil: number;
}

export interface TownInfo {
  x: number;
  z: number;
  r: number;
  name: number;
  kind: number;
  y: number;
}

interface CoreExports {
  memory: WebAssembly.Memory;
  in_ptr(): number;
  proto_version(): number;
  cli_init(seed: number): number;
  cli_mesh_ptr(kind: number): number;
  cli_mesh_len(kind: number): number;
  cli_free_meshes(): void;
  cli_height(x: number, z: number): number;
  cli_ground(x: number, z: number, feet: number): number;
  cli_ray(ox: number, oy: number, oz: number, dx: number, dy: number, dz: number, maxt: number): number;
  cli_ray_kind(): number;
  cli_building_at(x: number, z: number): number;
  cli_pred_ptr(): number;
  cli_pred_step(): void;
}

let modulePromise: Promise<WebAssembly.Module> | null = null;

/** Compile once; the same Module is shared with the offline server worker. */
export function coreModule(): Promise<WebAssembly.Module> {
  if (!modulePromise) {
    const url = new URL('core.wasm', document.baseURI).href;
    modulePromise = (async () => {
      const res = await fetch(url);
      if (!res.ok) throw new Error(`core.wasm: HTTP ${res.status}`);
      if ('compileStreaming' in WebAssembly && res.headers.get('content-type')?.includes('application/wasm')) {
        return WebAssembly.compileStreaming(res);
      }
      return WebAssembly.compile(await res.arrayBuffer());
    })();
    modulePromise.catch(() => (modulePromise = null));
  }
  return modulePromise;
}

export class Core {
  private ex: CoreExports;
  seed = 0;
  checksum = 0;
  wdefs: WDef[] = [];
  towns: TownInfo[] = [];

  private constructor(ex: CoreExports) {
    this.ex = ex;
  }

  static async create(): Promise<Core> {
    const mod = await coreModule();
    const inst = await WebAssembly.instantiate(mod, {});
    return new Core(inst.exports as unknown as CoreExports);
  }

  get proto() {
    return this.ex.proto_version();
  }

  /** Generates the world for `seed`. Returns the world checksum. */
  init(seed: number): number {
    this.seed = seed >>> 0;
    this.checksum = this.ex.cli_init(this.seed) >>> 0;
    const w = this.f32(6);
    this.wdefs = [];
    for (let i = 0; i + 18 <= w.length; i += 18) {
      this.wdefs.push({
        ammo: w[i], dmg: w[i + 1], interval: w[i + 2], mag: w[i + 3], reload: w[i + 4], hip: w[i + 5], ads: w[i + 6],
        pellets: w[i + 10], range: w[i + 12], speed: w[i + 13], zoom: w[i + 14], recoil: w[i + 16],
      });
    }
    const t = this.f32(7);
    this.towns = [];
    for (let i = 0; i + 6 <= t.length; i += 6) this.towns.push({ x: t[i], z: t[i + 1], r: t[i + 2], name: t[i + 3], kind: t[i + 4], y: t[i + 5] });
    return this.checksum;
  }

  /** Copy of a mesh/output buffer (safe across memory growth). */
  bytes(kind: number): Uint8Array {
    const ptr = this.ex.cli_mesh_ptr(kind);
    const len = this.ex.cli_mesh_len(kind);
    return new Uint8Array(this.ex.memory.buffer, ptr, len).slice();
  }
  f32(kind: number): Float32Array {
    const b = this.bytes(kind);
    return new Float32Array(b.buffer, b.byteOffset, b.byteLength >> 2);
  }
  freeMeshes() {
    this.ex.cli_free_meshes();
  }

  height(x: number, z: number) {
    return this.ex.cli_height(x, z);
  }
  ground(x: number, z: number, feet: number) {
    return this.ex.cli_ground(x, z, feet);
  }
  /** Distance to first hit or -1. */
  ray(o: ArrayLike<number>, d: ArrayLike<number>, maxt: number) {
    return this.ex.cli_ray(o[0], o[1], o[2], d[0], d[1], d[2], maxt);
  }
  rayKind() {
    return this.ex.cli_ray_kind();
  }
  buildingAt(x: number, z: number) {
    return this.ex.cli_building_at(x, z);
  }

  /** Shared prediction IO block (see lib.rs for layout). */
  pred(): Float32Array {
    return new Float32Array(this.ex.memory.buffer, this.ex.cli_pred_ptr(), 32);
  }
  predStep() {
    this.ex.cli_pred_step();
  }
}
