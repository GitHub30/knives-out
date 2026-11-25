// Raw WebGPU renderer: an HDR scene pass (static world, instanced shapes, glTF
// models, sky, water, zone wall) lit by a sun with two shadow cascades, then
// bloom + tonemap + FXAA into the swap chain.

import { AO_WGSL, BLOOM_WGSL, BOX_SHADOW_WGSL, BOX_WGSL, LIT_SHADOW_WGSL, LIT_WGSL, MIP_WGSL, MODEL_SHADOW_WGSL, MODEL_WGSL, POST_WGSL, PUFF_WGSL, SKY_WGSL, WATER_WGSL, ZONE_WGSL } from './shaders';
import { aabbVisible, cross, frustumPlanes, lookAt, mul, norm, ortho, qAxis, qMul, qRot, type M4, type Q, type V3 } from './math';
import { Instances, INSTANCE_STRIDE, MAX_SHAPES } from './instances';
import { pill, rod, roundedBox, sphere } from './shapes';
import { loadGlb, type GltfModel } from './gltf';

export interface Quality {
  shadowSize: number; // 0 disables shadows
  scale: number; // render resolution multiplier (applied on top of capped DPR)
  maxDpr: number;
  /** Upper bound of rendered pixels (the internal resolution never exceeds it). */
  maxPixels: number;
  drawDist: number;
  bloom: boolean;
  ao: boolean;
}

export const QUALITY_PRESETS: Record<'low' | 'medium' | 'high', Quality> = {
  low: { shadowSize: 1024, scale: 0.8, maxDpr: 1.25, maxPixels: 1280 * 720, drawDist: 380, bloom: false, ao: false },
  medium: { shadowSize: 2048, scale: 1, maxDpr: 1.5, maxPixels: 1600 * 900, drawDist: 520, bloom: true, ao: true },
  // 1920 x 1080 internal resolution at most
  high: { shadowSize: 2048, scale: 1, maxDpr: 2, maxPixels: 1920 * 1080, drawDist: 640, bloom: true, ao: true },
};

export interface WorldMeshes {
  terrainVerts: Uint8Array;
  terrainIdx: Uint8Array;
  terrainChunks: Float32Array;
  propVerts: Uint8Array;
  propChunks: Float32Array;
  heights: Float32Array; // 257 x 257
  mapSize: number;
  waterLevel: number;
  /** Parked cars: x, y, z, yaw per car, and the vehicle kind of each. */
  cars?: Float32Array;
  carKinds?: Uint8Array;
}

export interface FrameView {
  viewProj: M4;
  invViewProj: M4;
  camPos: V3;
  time: number;
  shadowCenter: V3;
  zone: { cx: number; cz: number; r: number; visible: boolean };
  hurt: number;
  /** Camera target used to see through foliage (null disables). */
  focus: V3 | null;
  boxes: Instances;
  glow: Instances;
  smoke: Instances;
  /** Soft smoke and fire particles (optional). */
  puffs?: Puffs;
  /** Window panes as thin boxes (optional). */
  glass?: Instances;
}

/** Billboard particle kinds. */
export const PUFF_SMOKE = 0;
export const PUFF_FLAME = 1;
export const PUFF_FIREBALL = 2;

/** Soft particles of a frame: 32 bytes each (pos, size, rgba8, rotation, seed, kind). */
export class Puffs {
  private buf = new ArrayBuffer(256 * 32);
  private f = new Float32Array(this.buf);
  private u = new Uint32Array(this.buf);
  count = 0;
  reset() {
    this.count = 0;
  }
  /** `color`: packed like the box instances (see `rgba`), alpha = opacity / heat. */
  push(p: ArrayLike<number>, size: number, color: number, rot: number, seed: number, kind: number) {
    if (this.count * 32 >= this.buf.byteLength) {
      const nb = new ArrayBuffer(this.buf.byteLength * 2);
      new Uint8Array(nb).set(new Uint8Array(this.buf));
      this.buf = nb;
      this.f = new Float32Array(nb);
      this.u = new Uint32Array(nb);
    }
    const o = this.count * 8;
    this.f[o] = p[0];
    this.f[o + 1] = p[1];
    this.f[o + 2] = p[2];
    this.f[o + 3] = size;
    this.u[o + 4] = color;
    this.f[o + 5] = rot;
    this.f[o + 6] = seed;
    this.f[o + 7] = kind;
    this.count++;
  }
  /** Far to near, so the blending layers up correctly. */
  sorted(cam: ArrayLike<number>): ArrayBuffer {
    const n = this.count;
    const order = new Uint32Array(n);
    const dist = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      order[i] = i;
      const o = i * 8;
      dist[i] = (this.f[o] - cam[0]) ** 2 + (this.f[o + 1] - cam[1]) ** 2 + (this.f[o + 2] - cam[2]) ** 2;
    }
    order.sort((a, b) => dist[b] - dist[a]);
    const out = new Uint32Array(n * 8);
    for (let i = 0; i < n; i++) out.set(this.u.subarray(order[i] * 8, order[i] * 8 + 8), i * 8);
    return out.buffer;
  }
}

const SUN = norm([-0.42, 0.78, -0.46]);
const HDR: GPUTextureFormat = 'rgba16float';
const CASCADE_R = [38, 190];
const U_FLOATS = 116;

interface Chunk {
  first: number;
  count: number;
  min: number[];
  max: number[];
}

function readChunks(t: Float32Array, stride: number): Chunk[] {
  const out: Chunk[] = [];
  for (let i = 0; i + stride <= t.length; i += stride) {
    const count = t[i + 1];
    if (count <= 0) continue;
    out.push({ first: t[i], count, min: [t[i + 2], t[i + 3], t[i + 4]], max: [t[i + 5], t[i + 6], t[i + 7]] });
  }
  return out;
}

function cubeVerts(): Float32Array {
  const faces: [V3, V3, V3, V3][] = [];
  const P = (x: number, y: number, z: number): V3 => [x - 0.5, y - 0.5, z - 0.5];
  faces.push([P(0, 1, 0), P(0, 1, 1), P(1, 1, 1), P(1, 1, 0)]); // +y
  faces.push([P(0, 0, 0), P(1, 0, 0), P(1, 0, 1), P(0, 0, 1)]); // -y
  faces.push([P(1, 0, 0), P(1, 1, 0), P(1, 1, 1), P(1, 0, 1)]); // +x
  faces.push([P(0, 0, 0), P(0, 0, 1), P(0, 1, 1), P(0, 1, 0)]); // -x
  faces.push([P(0, 0, 1), P(1, 0, 1), P(1, 1, 1), P(0, 1, 1)]); // +z
  faces.push([P(0, 0, 0), P(0, 1, 0), P(1, 1, 0), P(1, 0, 0)]); // -z
  const out: number[] = [];
  for (const [a, b, c, d] of faces) {
    const n = norm(cross([b[0] - a[0], b[1] - a[1], b[2] - a[2]], [c[0] - a[0], c[1] - a[1], c[2] - a[2]]));
    for (const v of [a, b, c, a, c, d]) out.push(v[0], v[1], v[2], n[0], n[1], n[2]);
  }
  return new Float32Array(out);
}

function cylinderVerts(seg: number): Float32Array {
  const out: number[] = [];
  const y0 = -40, y1 = 320;
  for (let i = 0; i < seg; i++) {
    const a0 = (i / seg) * Math.PI * 2, a1 = ((i + 1) / seg) * Math.PI * 2;
    const x0 = Math.cos(a0), z0 = Math.sin(a0), x1 = Math.cos(a1), z1 = Math.sin(a1);
    out.push(x0, y0, z0, x1, y0, z1, x1, y1, z1, x0, y0, z0, x1, y1, z1, x0, y1, z0);
  }
  return new Float32Array(out);
}

/** Floats per car handed to `setCars`. */
export const CAR_STRIDE = 10;

/** Engine axes at the intake lips in the airliner's frame, and the fan speed (rad/s). */
const FAN_AXES: V3[] = [[9.75, -3.0, 11.0], [-9.75, -3.0, 11.0]];
const FAN_SPEED = 9;

interface ModelGpu {
  vb: GPUBuffer;
  ib: GPUBuffer;
  parts: { first: number; count: number; blend: boolean; bg: GPUBindGroup }[];
}

export class Renderer {
  readonly canvas: HTMLCanvasElement;
  readonly device: GPUDevice;
  readonly ctx: GPUCanvasContext | null;
  readonly format: GPUTextureFormat;
  quality: Quality;
  lost = false;
  /** Triangles submitted in the last frame's main pass (debug overlay / tests). */
  stats = { tris: 0, draws: 0, width: 0, height: 0, scale: 1 };

  private uniform: GPUBuffer;
  private uniformSh1: GPUBuffer;
  private uData = new Float32Array(U_FLOATS);
  private bglMain: GPUBindGroupLayout;
  private bglShadow: GPUBindGroupLayout;
  private bglWater: GPUBindGroupLayout;
  private bglMat: GPUBindGroupLayout;
  private bglTex: GPUBindGroupLayout;
  private bglPost: GPUBindGroupLayout;
  private bglAo: GPUBindGroupLayout;
  private bgAo: GPUBindGroup | null = null;
  private aoTex: GPUTexture | null = null;
  private bgShadow: GPUBindGroup[] = [];
  private bgMain!: GPUBindGroup;
  private bgWater!: GPUBindGroup;
  private bgPost: GPUBindGroup | null = null;
  private bgBloom: GPUBindGroup[] = [];
  private shadowTex!: GPUTexture;
  private shadowViews: GPUTextureView[] = [];
  private shadowSampler: GPUSampler;
  private linear: GPUSampler;
  private aniso: GPUSampler;
  private depthTex: GPUTexture | null = null;
  private bglDepth!: GPUBindGroupLayout;
  private bgDepth: GPUBindGroup | null = null;
  private hdrTex: GPUTexture | null = null;
  private bloomTex: GPUTexture[] = [];
  private offTex: GPUTexture | null = null;
  private pipes!: Record<string, GPURenderPipeline>;
  private shapeVB: { buf: GPUBuffer; count: number }[] = [];
  private cubeVB: GPUBuffer;
  private zoneVB: GPUBuffer;
  private zoneCount: number;
  private instBufs: Record<string, { buf: GPUBuffer; size: number }> = {};
  private terrainVB: GPUBuffer | null = null;
  private terrainIB: GPUBuffer | null = null;
  private propVB: GPUBuffer | null = null;
  private tChunks: Chunk[] = [];
  private pChunks: Chunk[] = [];
  private heightTex: GPUTexture | null = null;
  private mapSize = 1024;
  private waterLevel = 0;
  private width = 0;
  private height = 0;
  private captureReq: ((img: { w: number; h: number; rgba: Uint8Array }) => void) | null = null;
  /** One model per vehicle kind (0 police car, 1 fire truck). */
  private carModels: (ModelGpu | null)[] = [null, null];
  /** Wheels per vehicle kind, grouped by wheel mesh: where each sits on the car. */
  private carWheels: { m: ModelGpu; at: { pos: V3; quat: Q; front: boolean }[]; data: Float32Array }[][] = [[], []];
  private carKinds: Uint8Array = new Uint8Array(0);
  /** Per car (CAR_STRIDE floats): x, y, z, quaternion, tint, wheel spin, steering angle. */
  private cars: Float32Array = new Float32Array(0);
  private carData: Float32Array[] = [new Float32Array(0), new Float32Array(0)];
  private planeModel: ModelGpu | null = null;
  private planePose: Float32Array | null = null;
  private planeInst: GPUBuffer | null = null;
  private fanModel: ModelGpu | null = null;
  private fanPose = new Float32Array(16);
  private fanInst: GPUBuffer | null = null;
  // dynamic resolution
  private dyn = 1;
  private lastT = 0;
  private slowT = 0;
  private fastT = 0;
  private avgMs = 16;

  /** Debug: render the next frame into a readable texture and return RGBA pixels. */
  capture(): Promise<{ w: number; h: number; rgba: Uint8Array }> {
    return new Promise((resolve) => (this.captureReq = resolve));
  }

  /** `offscreen` renders to a texture instead of the canvas (headless testing). */
  static async create(canvas: HTMLCanvasElement, quality: Quality, offscreen = false): Promise<Renderer> {
    if (!('gpu' in navigator) || !navigator.gpu) throw new Error('WebGPU unavailable');
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
    if (!adapter) throw new Error('No WebGPU adapter');
    const device = await adapter.requestDevice();
    return new Renderer(canvas, device, quality, offscreen);
  }

  private constructor(canvas: HTMLCanvasElement, device: GPUDevice, quality: Quality, offscreen: boolean) {
    this.canvas = canvas;
    this.device = device;
    this.quality = quality;
    if (offscreen) {
      this.ctx = null;
      this.format = 'rgba8unorm';
    } else {
      const ctx = canvas.getContext('webgpu');
      if (!ctx) throw new Error('Could not get WebGPU context');
      this.ctx = ctx;
      this.format = navigator.gpu.getPreferredCanvasFormat();
      ctx.configure({ device, format: this.format, alphaMode: 'opaque' });
    }
    device.lost.then(() => {
      this.lost = true;
    });
    let reported = 0;
    device.addEventListener('uncapturederror', (e) => {
      if (reported++ < 10) console.error('WebGPU error:', (e as GPUUncapturedErrorEvent).error.message);
    });

    const usize = U_FLOATS * 4;
    this.uniform = device.createBuffer({ size: usize, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.uniformSh1 = device.createBuffer({ size: usize, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    const VF = GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT;
    this.bglMain = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: VF, buffer: { type: 'uniform' } },
        { binding: 1, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'depth', viewDimension: '2d-array' } },
        { binding: 2, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'comparison' } },
      ],
    });
    this.bglShadow = device.createBindGroupLayout({ entries: [{ binding: 0, visibility: VF, buffer: { type: 'uniform' } }] });
    this.bglWater = device.createBindGroupLayout({
      entries: [{ binding: 0, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'unfilterable-float' } }],
    });
    this.bglMat = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
        { binding: 1, visibility: GPUShaderStage.FRAGMENT, sampler: {} },
        { binding: 2, visibility: GPUShaderStage.FRAGMENT, texture: {} },
        { binding: 3, visibility: GPUShaderStage.FRAGMENT, texture: {} },
        { binding: 4, visibility: GPUShaderStage.FRAGMENT, texture: {} },
        { binding: 5, visibility: GPUShaderStage.FRAGMENT, texture: {} },
      ],
    });
    this.bglTex = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.FRAGMENT, texture: {} },
        { binding: 1, visibility: GPUShaderStage.FRAGMENT, sampler: {} },
      ],
    });
    this.bglPost = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
        { binding: 1, visibility: GPUShaderStage.FRAGMENT, texture: {} },
        { binding: 2, visibility: GPUShaderStage.FRAGMENT, sampler: {} },
        { binding: 3, visibility: GPUShaderStage.FRAGMENT, texture: {} },
        { binding: 4, visibility: GPUShaderStage.FRAGMENT, texture: {} },
        { binding: 5, visibility: GPUShaderStage.FRAGMENT, texture: {} },
        { binding: 6, visibility: GPUShaderStage.FRAGMENT, texture: {} },
        { binding: 7, visibility: GPUShaderStage.FRAGMENT, texture: {} },
      ],
    });
    this.bglAo = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
        { binding: 1, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'depth' } },
      ],
    });
    this.bgShadow = [this.uniform, this.uniformSh1].map((buffer) => device.createBindGroup({ layout: this.bglShadow, entries: [{ binding: 0, resource: { buffer } }] }));
    this.shadowSampler = device.createSampler({ compare: 'less', magFilter: 'linear', minFilter: 'linear' });
    this.linear = device.createSampler({ magFilter: 'linear', minFilter: 'linear', addressModeU: 'clamp-to-edge', addressModeV: 'clamp-to-edge' });
    this.aniso = device.createSampler({ magFilter: 'linear', minFilter: 'linear', mipmapFilter: 'linear', addressModeU: 'repeat', addressModeV: 'repeat', maxAnisotropy: 4 });

    const mkVB = (data: Float32Array) => {
      const buf = device.createBuffer({ size: data.byteLength, usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST });
      device.queue.writeBuffer(buf, 0, data);
      return buf;
    };
    this.cubeVB = mkVB(cubeVerts());
    for (const mesh of [roundedBox(), sphere(), pill(), rod()]) this.shapeVB.push({ buf: mkVB(mesh), count: mesh.length / 6 });
    const cyl = cylinderVerts(160);
    this.zoneCount = cyl.length / 3;
    this.zoneVB = mkVB(cyl);

    this.makeShadowTex();
    this.buildPipelines();
    this.resize();
  }

  private makeShadowTex() {
    this.shadowTex?.destroy();
    const size = Math.max(1, this.quality.shadowSize);
    this.shadowTex = this.device.createTexture({
      size: [size, size, 2],
      format: 'depth32float',
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
    });
    this.shadowViews = [0, 1].map((k) => this.shadowTex.createView({ dimension: '2d', baseArrayLayer: k, arrayLayerCount: 1 }));
    this.bgMain = this.device.createBindGroup({
      layout: this.bglMain,
      entries: [
        { binding: 0, resource: { buffer: this.uniform } },
        { binding: 1, resource: this.shadowTex.createView({ dimension: '2d-array' }) },
        { binding: 2, resource: this.shadowSampler },
      ],
    });
  }

  setQuality(q: Quality) {
    const reshadow = q.shadowSize !== this.quality.shadowSize;
    this.quality = q;
    this.dyn = 1;
    if (reshadow) this.makeShadowTex();
    this.resize(true);
  }

  private buildPipelines() {
    const d = this.device;
    const mod = (code: string) => d.createShaderModule({ code });
    const lit = mod(LIT_WGSL), litSh = mod(LIT_SHADOW_WGSL), box = mod(BOX_WGSL), boxSh = mod(BOX_SHADOW_WGSL);
    const model = mod(MODEL_WGSL), modelSh = mod(MODEL_SHADOW_WGSL);
    const sky = mod(SKY_WGSL), water = mod(WATER_WGSL), zone = mod(ZONE_WGSL);
    const puff = mod(PUFF_WGSL);
    const bloom = mod(BLOOM_WGSL), post = mod(POST_WGSL), mip = mod(MIP_WGSL), ao = mod(AO_WGSL);
    const layoutAo = d.createPipelineLayout({ bindGroupLayouts: [this.bglAo] });
    const layoutMain = d.createPipelineLayout({ bindGroupLayouts: [this.bglMain] });
    this.bglDepth = d.createBindGroupLayout({ entries: [{ binding: 0, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'depth' } }] });
    const layoutPuff = d.createPipelineLayout({ bindGroupLayouts: [this.bglMain, this.bglDepth] });
    const layoutShadow = d.createPipelineLayout({ bindGroupLayouts: [this.bglShadow] });
    const layoutWater = d.createPipelineLayout({ bindGroupLayouts: [this.bglMain, this.bglWater] });
    const layoutModel = d.createPipelineLayout({ bindGroupLayouts: [this.bglMain, this.bglMat] });
    const layoutTex = d.createPipelineLayout({ bindGroupLayouts: [this.bglTex] });
    const layoutPost = d.createPipelineLayout({ bindGroupLayouts: [this.bglPost] });
    const litVB: GPUVertexBufferLayout = {
      arrayStride: 20,
      attributes: [
        { shaderLocation: 0, offset: 0, format: 'float32x3' },
        { shaderLocation: 1, offset: 12, format: 'snorm8x4' },
        { shaderLocation: 2, offset: 16, format: 'unorm8x4' },
      ],
    };
    const boxVB: GPUVertexBufferLayout[] = [
      { arrayStride: 24, attributes: [{ shaderLocation: 0, offset: 0, format: 'float32x3' }, { shaderLocation: 1, offset: 12, format: 'float32x3' }] },
      {
        arrayStride: INSTANCE_STRIDE,
        stepMode: 'instance',
        attributes: [
          { shaderLocation: 2, offset: 0, format: 'float32x3' },
          { shaderLocation: 3, offset: 12, format: 'float32x4' },
          { shaderLocation: 4, offset: 28, format: 'float32x3' },
          { shaderLocation: 5, offset: 40, format: 'unorm8x4' },
          { shaderLocation: 6, offset: 44, format: 'unorm8x4' },
        ],
      },
    ];
    const modelVB: GPUVertexBufferLayout[] = [
      {
        arrayStride: 48,
        attributes: [
          { shaderLocation: 0, offset: 0, format: 'float32x3' },
          { shaderLocation: 1, offset: 12, format: 'float32x3' },
          { shaderLocation: 2, offset: 24, format: 'float32x4' },
          { shaderLocation: 3, offset: 40, format: 'float32x2' },
        ],
      },
      {
        arrayStride: 32,
        stepMode: 'instance',
        attributes: [
          { shaderLocation: 4, offset: 0, format: 'float32x3' },
          { shaderLocation: 5, offset: 12, format: 'float32x4' },
          { shaderLocation: 6, offset: 28, format: 'float32' },
        ],
      },
    ];
    const depthMain = (write: boolean, compare: GPUCompareFunction = 'greater'): GPUDepthStencilState => ({ format: 'depth32float', depthWriteEnabled: write, depthCompare: compare });
    const target = (blend?: GPUBlendState): GPUColorTargetState[] => [{ format: HDR, blend }];
    const alpha: GPUBlendState = {
      color: { srcFactor: 'src-alpha', dstFactor: 'one-minus-src-alpha', operation: 'add' },
      alpha: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' },
    };
    const additive: GPUBlendState = {
      color: { srcFactor: 'src-alpha', dstFactor: 'one', operation: 'add' },
      alpha: { srcFactor: 'zero', dstFactor: 'one', operation: 'add' },
    };
    const shadowDepth: GPUDepthStencilState = { format: 'depth32float', depthWriteEnabled: true, depthCompare: 'less', depthBias: 2, depthBiasSlopeScale: 2.0 };
    const tri: GPUPrimitiveState = { topology: 'triangle-list', cullMode: 'back', frontFace: 'ccw' };
    const fullscreen = (module: GPUShaderModule, entryPoint: string, layout: GPUPipelineLayout, format: GPUTextureFormat) =>
      d.createRenderPipeline({ layout, vertex: { module, entryPoint: 'vs' }, fragment: { module, entryPoint, targets: [{ format }] }, primitive: { topology: 'triangle-list' } });
    this.pipes = {
      terrain: d.createRenderPipeline({
        layout: layoutMain,
        vertex: { module: lit, entryPoint: 'vs', buffers: [litVB] },
        fragment: { module: lit, entryPoint: 'fs_terrain', targets: target() },
        primitive: tri,
        depthStencil: depthMain(true),
      }),
      props: d.createRenderPipeline({
        layout: layoutMain,
        vertex: { module: lit, entryPoint: 'vs', buffers: [litVB] },
        fragment: { module: lit, entryPoint: 'fs_props', targets: target() },
        primitive: tri,
        depthStencil: depthMain(true),
      }),
      litShadow: d.createRenderPipeline({
        layout: layoutShadow,
        vertex: { module: litSh, entryPoint: 'vs', buffers: [litVB] },
        primitive: { topology: 'triangle-list', cullMode: 'none' },
        depthStencil: shadowDepth,
      }),
      box: d.createRenderPipeline({
        layout: layoutMain,
        vertex: { module: box, entryPoint: 'vs', buffers: boxVB },
        fragment: { module: box, entryPoint: 'fs', targets: target() },
        primitive: tri,
        depthStencil: depthMain(true),
      }),
      boxShadow: d.createRenderPipeline({
        layout: layoutShadow,
        vertex: { module: boxSh, entryPoint: 'vs', buffers: boxVB },
        primitive: { topology: 'triangle-list', cullMode: 'none' },
        depthStencil: shadowDepth,
      }),
      model: d.createRenderPipeline({
        layout: layoutModel,
        vertex: { module: model, entryPoint: 'vs', buffers: modelVB },
        fragment: { module: model, entryPoint: 'fs', targets: target() },
        primitive: { topology: 'triangle-list', cullMode: 'none' },
        depthStencil: depthMain(true),
      }),
      modelBlend: d.createRenderPipeline({
        layout: layoutModel,
        vertex: { module: model, entryPoint: 'vs', buffers: modelVB },
        fragment: { module: model, entryPoint: 'fs_blend', targets: target(alpha) },
        primitive: { topology: 'triangle-list', cullMode: 'none' },
        depthStencil: depthMain(false),
      }),
      modelShadow: d.createRenderPipeline({
        layout: layoutShadow,
        vertex: { module: modelSh, entryPoint: 'vs', buffers: modelVB },
        primitive: { topology: 'triangle-list', cullMode: 'none' },
        depthStencil: shadowDepth,
      }),
      glow: d.createRenderPipeline({
        layout: layoutMain,
        vertex: { module: box, entryPoint: 'vs', buffers: boxVB },
        fragment: { module: box, entryPoint: 'fs_glow', targets: target(additive) },
        primitive: { topology: 'triangle-list', cullMode: 'none' },
        depthStencil: depthMain(false),
      }),
      smoke: d.createRenderPipeline({
        layout: layoutMain,
        vertex: { module: box, entryPoint: 'vs', buffers: boxVB },
        fragment: { module: box, entryPoint: 'fs_smoke', targets: target(alpha) },
        primitive: tri,
        depthStencil: depthMain(false),
      }),
      glass: d.createRenderPipeline({
        layout: layoutMain,
        vertex: { module: box, entryPoint: 'vs', buffers: boxVB },
        fragment: { module: box, entryPoint: 'fs_glass', targets: target(alpha) },
        primitive: { topology: 'triangle-list', cullMode: 'none' },
        depthStencil: depthMain(false),
      }),
      puff: d.createRenderPipeline({
        layout: layoutPuff,
        vertex: {
          module: puff,
          entryPoint: 'vs',
          buffers: [{
            arrayStride: 32,
            stepMode: 'instance',
            attributes: [
              { shaderLocation: 0, offset: 0, format: 'float32x3' },
              { shaderLocation: 1, offset: 12, format: 'float32' },
              { shaderLocation: 2, offset: 16, format: 'unorm8x4' },
              { shaderLocation: 3, offset: 20, format: 'float32x3' },
            ],
          }],
        },
        // premultiplied: smoke covers what is behind it, fire adds light
        fragment: {
          module: puff,
          entryPoint: 'fs',
          targets: target({ color: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' }, alpha: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' } }),
        },
        primitive: { topology: 'triangle-list', cullMode: 'none' },
        depthStencil: depthMain(false),
      }),
      sky: d.createRenderPipeline({
        layout: layoutMain,
        vertex: { module: sky, entryPoint: 'vs' },
        fragment: { module: sky, entryPoint: 'fs', targets: target() },
        primitive: { topology: 'triangle-list' },
        depthStencil: depthMain(false, 'always'),
      }),
      water: d.createRenderPipeline({
        layout: layoutWater,
        vertex: { module: water, entryPoint: 'vs' },
        fragment: { module: water, entryPoint: 'fs', targets: target(alpha) },
        primitive: { topology: 'triangle-list', cullMode: 'none' },
        depthStencil: depthMain(false),
      }),
      zone: d.createRenderPipeline({
        layout: layoutMain,
        vertex: { module: zone, entryPoint: 'vs', buffers: [{ arrayStride: 12, attributes: [{ shaderLocation: 0, offset: 0, format: 'float32x3' }] }] },
        fragment: { module: zone, entryPoint: 'fs', targets: target(alpha) },
        primitive: { topology: 'triangle-list', cullMode: 'none' },
        depthStencil: depthMain(false),
      }),
      bloomFirst: fullscreen(bloom, 'fs_first', layoutTex, HDR),
      bloomDown: fullscreen(bloom, 'fs_down', layoutTex, HDR),
      post: fullscreen(post, 'fs', layoutPost, this.format),
      ao: fullscreen(ao, 'fs', layoutAo, 'r8unorm'),
      mipLinear: fullscreen(mip, 'fs', layoutTex, 'rgba8unorm'),
      mipSrgb: fullscreen(mip, 'fs', layoutTex, 'rgba8unorm-srgb'),
    };
  }

  uploadWorld(w: WorldMeshes) {
    const d = this.device;
    const mk = (data: Uint8Array, usage: number) => {
      const size = Math.max(4, Math.ceil(data.byteLength / 4) * 4);
      const b = d.createBuffer({ size, usage: usage | GPUBufferUsage.COPY_DST });
      d.queue.writeBuffer(b, 0, data.buffer, data.byteOffset, data.byteLength);
      return b;
    };
    this.terrainVB?.destroy();
    this.terrainIB?.destroy();
    this.propVB?.destroy();
    this.terrainVB = mk(w.terrainVerts, GPUBufferUsage.VERTEX);
    this.terrainIB = mk(w.terrainIdx, GPUBufferUsage.INDEX);
    this.propVB = mk(w.propVerts, GPUBufferUsage.VERTEX);
    this.tChunks = readChunks(w.terrainChunks, 8);
    this.pChunks = readChunks(w.propChunks, 8);
    this.mapSize = w.mapSize;
    this.waterLevel = w.waterLevel;
    // parked poses until the game says where the cars are now
    const c4 = w.cars ?? new Float32Array(0);
    this.cars = new Float32Array((c4.length >> 2) * CAR_STRIDE);
    for (let i = 0, o = 0; i + 4 <= c4.length; i += 4, o += CAR_STRIDE) {
      this.cars.set([c4[i], c4[i + 1], c4[i + 2], 0, Math.sin(c4[i + 3] * 0.5), 0, Math.cos(c4[i + 3] * 0.5), 1, 0, 0], o);
    }
    this.carKinds = w.carKinds ?? new Uint8Array(c4.length >> 2);
    this.heightTex?.destroy();
    this.heightTex = d.createTexture({ size: [257, 257], format: 'r32float', usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST });
    d.queue.writeTexture({ texture: this.heightTex }, w.heights.buffer, { offset: w.heights.byteOffset, bytesPerRow: 257 * 4, rowsPerImage: 257 }, [257, 257]);
    this.bgWater = d.createBindGroup({ layout: this.bglWater, entries: [{ binding: 0, resource: this.heightTex.createView() }] });
  }

  // ------------------------------------------------------------------ models
  private texture(img: ImageBitmap | null, srgb: boolean, fallback: [number, number, number, number]): GPUTexture {
    const d = this.device;
    const format: GPUTextureFormat = srgb ? 'rgba8unorm-srgb' : 'rgba8unorm';
    if (!img) {
      const t = d.createTexture({ size: [1, 1], format, usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST });
      d.queue.writeTexture({ texture: t }, new Uint8Array(fallback), {}, [1, 1]);
      return t;
    }
    const mips = Math.floor(Math.log2(Math.max(img.width, img.height))) + 1;
    const t = d.createTexture({
      size: [img.width, img.height],
      format,
      mipLevelCount: mips,
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT,
    });
    d.queue.copyExternalImageToTexture({ source: img }, { texture: t }, [img.width, img.height]);
    const enc = d.createCommandEncoder();
    const pipe = srgb ? this.pipes.mipSrgb : this.pipes.mipLinear;
    for (let l = 1; l < mips; l++) {
      const bg = d.createBindGroup({
        layout: this.bglTex,
        entries: [
          { binding: 0, resource: t.createView({ baseMipLevel: l - 1, mipLevelCount: 1 }) },
          { binding: 1, resource: this.linear },
        ],
      });
      const p = enc.beginRenderPass({ colorAttachments: [{ view: t.createView({ baseMipLevel: l, mipLevelCount: 1 }), loadOp: 'clear', storeOp: 'store' }] });
      p.setPipeline(pipe);
      p.setBindGroup(0, bg);
      p.draw(3);
      p.end();
    }
    d.queue.submit([enc.finish()]);
    return t;
  }

  private uploadModel(m: GltfModel): ModelGpu {
    const d = this.device;
    const vb = d.createBuffer({ size: m.verts.byteLength, usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST });
    d.queue.writeBuffer(vb, 0, m.verts);
    const ib = d.createBuffer({ size: m.indices.byteLength, usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST });
    d.queue.writeBuffer(ib, 0, m.indices);
    const parts = m.parts.map((p) => {
      const mat = m.materials[p.material];
      const textured = !!mat.baseTex;
      const u = d.createBuffer({ size: 48, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
      d.queue.writeBuffer(u, 0, new Float32Array([...mat.base, ...mat.emissive, 0, mat.metallic, mat.roughness, textured ? 1 : 0, 0]));
      const bg = d.createBindGroup({
        layout: this.bglMat,
        entries: [
          { binding: 0, resource: { buffer: u } },
          { binding: 1, resource: this.aniso },
          { binding: 2, resource: this.texture(mat.baseTex, true, [255, 255, 255, 255]).createView() },
          { binding: 3, resource: this.texture(mat.normalTex, false, [128, 128, 255, 255]).createView() },
          { binding: 4, resource: this.texture(mat.ormTex, false, [255, 255, 255, 255]).createView() },
          { binding: 5, resource: this.texture(mat.emissiveTex, true, [0, 0, 0, 255]).createView() },
        ],
      });
      return { first: p.first, count: p.count, blend: mat.blend, bg };
    });
    // opaque parts first
    parts.sort((a, b) => Number(a.blend) - Number(b.blend));
    return { vb, ib, parts };
  }

  /** Register extra instanced shapes (pos + normal, unindexed). Returns the first new shape id. */
  addShapes(meshes: Float32Array[]): number {
    const first = this.shapeVB.length;
    if (first + meshes.length > MAX_SHAPES) throw new Error('too many shapes');
    for (const m of meshes) {
      const data = m.length ? m : new Float32Array(18);
      const buf = this.device.createBuffer({ size: data.byteLength, usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST });
      this.device.queue.writeBuffer(buf, 0, data);
      this.shapeVB.push({ buf, count: m.length / 6 });
    }
    return first;
  }

  /** Load the parked-car model (drawn at the world's car spots once it is ready). */
  /** The cars this frame, CAR_STRIDE floats each: x, y, z, quaternion, tint (1 = intact), wheel spin and steering (rad). */
  setCars(poses: Float32Array) {
    this.cars = poses;
  }

  /** Load the airliner (the transport everyone jumps from) and its engine fans. */
  async loadPlaneModel(url: string, fanUrl: string): Promise<void> {
    const [m, f] = await Promise.all([loadGlb(url), loadGlb(fanUrl)]);
    if (this.lost) return;
    this.fanModel = this.uploadModel(f);
    this.planeModel = this.uploadModel(m);
  }
  get hasPlaneModel() {
    return !!this.planeModel;
  }
  /** Draw the airliner in the next frame: nose along fwd(yaw), banked by `roll`, fans turning with `time`. */
  setPlane(p: ArrayLike<number>, yaw: number, roll = 0, time = 0) {
    const q = qMul(qAxis(0, 1, 0, yaw), qAxis(0, 0, 1, roll));
    this.planePose = new Float32Array([p[0], p[1], p[2], q[0], q[1], q[2], q[3], 1]);
    // the fans sit on the engine axes (see scripts/build-b777.py) and spin about them
    FAN_AXES.forEach((at, k) => {
      const o = qRot(q, at);
      const fq = qMul(q, qAxis(0, 0, 1, -time * FAN_SPEED));
      this.fanPose.set([p[0] + o[0], p[1] + o[1], p[2] + o[2], fq[0], fq[1], fq[2], fq[3], 1], k * 8);
    });
  }

  async loadCarModel(url: string, kind = 0): Promise<void> {
    const m = await loadGlb(url, 'Wheel');
    if (this.lost) return;
    // wheels are drawn on their own so they can turn
    const groups = new Map<GltfModel, { m: ModelGpu; at: { pos: V3; quat: Q; front: boolean }[]; data: Float32Array }>();
    for (const w of m.wheels) {
      let g = groups.get(w.model);
      if (!g) {
        g = { m: this.uploadModel(w.model), at: [], data: new Float32Array(0) };
        groups.set(w.model, g);
      }
      g.at.push({ pos: w.pos, quat: w.quat, front: w.pos[2] > 0 });
    }
    this.carWheels[kind] = [...groups.values()];
    this.carModels[kind] = this.uploadModel(m);
  }

  // ------------------------------------------------------------------ targets
  resize(force = false) {
    const dpr = Math.min(window.devicePixelRatio || 1, this.quality.maxDpr) * this.quality.scale;
    let w = Math.max(1, this.canvas.clientWidth * dpr);
    let h = Math.max(1, this.canvas.clientHeight * dpr);
    // never above the pixel budget (1920 x 1080 on high), then the dynamic scale
    const k = Math.min(1, Math.sqrt(this.quality.maxPixels / (w * h))) * this.dyn;
    w = Math.max(1, Math.floor(w * k));
    h = Math.max(1, Math.floor(h * k));
    if (!force && w === this.width && h === this.height && this.depthTex) return;
    this.width = w;
    this.height = h;
    this.canvas.width = w;
    this.canvas.height = h;
    const d = this.device;
    this.depthTex?.destroy();
    this.hdrTex?.destroy();
    this.offTex?.destroy();
    for (const t of this.bloomTex) t.destroy();
    this.offTex = this.ctx ? null : d.createTexture({ size: [w, h], format: this.format, usage: GPUTextureUsage.RENDER_ATTACHMENT });
    this.depthTex = d.createTexture({ size: [w, h], format: 'depth32float', usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING });
    this.aoTex?.destroy();
    this.aoTex = d.createTexture({ size: [Math.max(1, w >> 1), Math.max(1, h >> 1)], format: 'r8unorm', usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING });
    this.bgDepth = d.createBindGroup({ layout: this.bglDepth, entries: [{ binding: 0, resource: this.depthTex.createView() }] });
    this.bgAo = d.createBindGroup({ layout: this.bglAo, entries: [{ binding: 0, resource: { buffer: this.uniform } }, { binding: 1, resource: this.depthTex.createView() }] });
    this.hdrTex = d.createTexture({ size: [w, h], format: HDR, usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING });
    this.bloomTex = [];
    for (let l = 1; l <= 4; l++) {
      this.bloomTex.push(d.createTexture({ size: [Math.max(1, w >> l), Math.max(1, h >> l)], format: HDR, usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING }));
    }
    const texBg = (t: GPUTexture) =>
      d.createBindGroup({ layout: this.bglTex, entries: [{ binding: 0, resource: t.createView() }, { binding: 1, resource: this.linear }] });
    this.bgBloom = [texBg(this.hdrTex), texBg(this.bloomTex[0]), texBg(this.bloomTex[1]), texBg(this.bloomTex[2])];
    this.bgPost = d.createBindGroup({
      layout: this.bglPost,
      entries: [
        { binding: 0, resource: { buffer: this.uniform } },
        { binding: 1, resource: this.hdrTex.createView() },
        { binding: 2, resource: this.linear },
        ...this.bloomTex.map((t, i) => ({ binding: 3 + i, resource: t.createView() })),
        { binding: 7, resource: this.aoTex.createView() },
      ],
    });
  }

  get aspect() {
    return this.width / Math.max(1, this.height);
  }

  /** Hold 30 fps: lower the internal resolution when frames run long, raise it back when there is room. */
  private adapt() {
    const now = performance.now();
    const ms = this.lastT ? now - this.lastT : 16;
    this.lastT = now;
    if (!this.ctx || ms > 500) return;
    this.avgMs += (ms - this.avgMs) * 0.1;
    const dt = ms / 1000;
    if (this.avgMs > 36) {
      this.slowT += dt;
      this.fastT = 0;
    } else if (this.avgMs < 22) {
      this.fastT += dt;
      this.slowT = 0;
    } else {
      this.slowT = this.fastT = 0;
    }
    if (this.slowT > 1.5 && this.dyn > 0.5) {
      this.dyn = Math.max(0.5, this.dyn * 0.87);
      this.slowT = 0;
      this.resize();
    } else if (this.fastT > 6 && this.dyn < 1) {
      this.dyn = Math.min(1, this.dyn / 0.93);
      this.fastT = 0;
      this.resize();
    }
  }

  private instBuffer(name: string, data: ArrayBuffer, bytes: number): GPUBuffer | null {
    if (bytes === 0) return null;
    let e = this.instBufs[name];
    if (!e || e.size < bytes) {
      e?.buf.destroy();
      const size = Math.ceil((Math.max(bytes, 16 * 1024) * 1.5) / 4) * 4;
      e = { buf: this.device.createBuffer({ size, usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST }), size };
      this.instBufs[name] = e;
    }
    this.device.queue.writeBuffer(e.buf, 0, data, 0, bytes);
    return e.buf;
  }

  private shapeBuffers(name: string, inst: Instances): (GPUBuffer | null)[] {
    return inst.lists.map((l, k) => (k < this.shapeVB.length ? this.instBuffer(`${name}${k}`, l.buf, l.count * INSTANCE_STRIDE) : null));
  }

  private lightMatrix(center: V3, R: number): M4 {
    const res = Math.max(1, this.quality.shadowSize);
    const up: V3 = [0, 1, 0];
    // snap the center to shadow texels to avoid shimmering
    const z = SUN;
    const x = norm(cross(up, z));
    const y = cross(z, x);
    const texel = (2 * R) / res;
    const px = center[0] * x[0] + center[1] * x[1] + center[2] * x[2];
    const py = center[0] * y[0] + center[1] * y[1] + center[2] * y[2];
    const sx = Math.round(px / texel) * texel - px;
    const sy = Math.round(py / texel) * texel - py;
    const c: V3 = [center[0] + x[0] * sx + y[0] * sy, center[1] + x[1] * sx + y[1] * sy, center[2] + x[2] * sx + y[2] * sy];
    const eye: V3 = [c[0] + SUN[0] * 500, c[1] + SUN[1] * 500, c[2] + SUN[2] * 500];
    const view = lookAt(eye, c, up);
    const proj = ortho(-R, R, -R, R, 10, 1000);
    return mul(proj, view);
  }

  private inflight = false;

  render(v: FrameView) {
    if (this.lost || !this.depthTex || !this.hdrTex || !this.terrainVB || !this.terrainIB || !this.propVB) return;
    // Offscreen (test) mode has no swap-chain back-pressure: keep one frame in flight.
    if (!this.ctx && this.inflight && !this.captureReq) return;
    this.adapt();
    this.resize();
    const d = this.device;
    const shadowsOn = this.quality.shadowSize > 0;
    const lightVP = [this.lightMatrix(v.shadowCenter, CASCADE_R[0]), this.lightMatrix(v.shadowCenter, CASCADE_R[1])];
    const u = this.uData;
    u.set(v.viewProj, 0);
    u.set(lightVP[0], 16);
    u.set(lightVP[1], 32);
    u.set(v.invViewProj, 48);
    u.set([v.camPos[0], v.camPos[1], v.camPos[2], v.time], 64);
    u.set([SUN[0], SUN[1], SUN[2], shadowsOn ? 1 : 0], 68);
    const lin = (c: number) => Math.pow(c, 2.2);
    u.set([3.6, 3.3, 2.85, 1.0], 72); // sun radiance (HDR), ambient scale
    u.set([lin(0.66), lin(0.75), lin(0.86), 1 / 1500], 76);
    u.set([lin(0.2), lin(0.42), lin(0.82), 1], 80);
    u.set([lin(0.62), lin(0.76), lin(0.92), 1], 84);
    u.set([1 / Math.max(1, this.quality.shadowSize), this.mapSize, v.hurt, this.waterLevel], 88);
    u.set([v.zone.cx, v.zone.cz, Math.max(0.01, v.zone.r), v.zone.visible ? 1 : 0], 92);
    u.set(v.focus ? [v.focus[0], v.focus[1], v.focus[2], 1] : [0, 0, 0, 0], 96);
    u.set([1 / this.width, 1 / this.height, this.quality.bloom ? 0.55 : 0, 0.62], 100);
    // view rays for the sky, worked out here in double precision (the shader
    // would lose them to cancellation next to the camera)
    {
      const m = v.invViewProj;
      const at = (x: number, y: number): V3 => {
        const z = 0.001;
        const w = m[3] * x + m[7] * y + m[11] * z + m[15];
        return [(m[0] * x + m[4] * y + m[8] * z + m[12]) / w - v.camPos[0], (m[1] * x + m[5] * y + m[9] * z + m[13]) / w - v.camPos[1], (m[2] * x + m[6] * y + m[10] * z + m[14]) / w - v.camPos[2]];
      };
      const c = at(0, 0), rx = at(1, 0), ry = at(0, 1);
      const l = Math.hypot(c[0], c[1], c[2]) || 1;
      u.set([c[0] / l, c[1] / l, c[2] / l, 0], 104);
      u.set([(rx[0] - c[0]) / l, (rx[1] - c[1]) / l, (rx[2] - c[2]) / l, 0], 108);
      u.set([(ry[0] - c[0]) / l, (ry[1] - c[1]) / l, (ry[2] - c[2]) / l, 0], 112);
    }
    d.queue.writeBuffer(this.uniform, 0, u);
    // the far cascade's passes read their matrix from the same slot
    const u1 = u.slice();
    u1.set(lightVP[1], 16);
    d.queue.writeBuffer(this.uniformSh1, 0, u1);

    const boxBufs = this.shapeBuffers('box', v.boxes);
    const glowBufs = this.shapeBuffers('glow', v.glow);
    const smokeBufs = this.shapeBuffers('smoke', v.smoke);

    // cars near the camera
    const cam = v.camPos;
    const carDraws: { m: ModelGpu; inst: GPUBuffer; n: number }[] = [];
    const nCars = Math.floor(this.cars.length / CAR_STRIDE);
    for (let kind = 0; kind < this.carModels.length; kind++) {
      const m = this.carModels[kind];
      if (!m || !nCars) continue;
      if (this.carData[kind].length < nCars * 8) this.carData[kind] = new Float32Array(nCars * 8);
      const data = this.carData[kind];
      const wheels = this.carWheels[kind];
      for (const g of wheels) if (g.data.length < nCars * g.at.length * 8) g.data = new Float32Array(nCars * g.at.length * 8);
      let n = 0;
      for (let k = 0; k < nCars; k++) {
        if ((this.carKinds[k] ?? 0) !== kind) continue;
        const i = k * CAR_STRIDE, c = this.cars;
        const dx = c[i] - cam[0], dz = c[i + 2] - cam[2];
        if (dx * dx + dz * dz > 260 * 260) continue;
        for (let j = 0; j < 8; j++) data[n * 8 + j] = c[i + j];
        // each wheel: on its hub, rolled about the axle, the front pair steered
        const q: Q = [c[i + 3], c[i + 4], c[i + 5], c[i + 6]];
        const roll = qAxis(1, 0, 0, c[i + 8]), steer = qAxis(0, 1, 0, c[i + 9]);
        for (const g of wheels) {
          g.at.forEach((w, wi) => {
            const o = qRot(q, w.pos);
            const turned = qMul(roll, w.quat);
            const wq = qMul(q, w.front ? qMul(steer, turned) : turned);
            g.data.set([c[i] + o[0], c[i + 1] + o[1], c[i + 2] + o[2], wq[0], wq[1], wq[2], wq[3], c[i + 7]], (n * g.at.length + wi) * 8);
          });
        }
        n++;
      }
      if (!n) continue;
      const inst = this.instBuffer(`cars${kind}`, data.buffer as ArrayBuffer, n * 32);
      if (inst) carDraws.push({ m, inst, n });
      wheels.forEach((g, gi) => {
        const wi = this.instBuffer(`wheels${kind}_${gi}`, g.data.buffer as ArrayBuffer, n * g.at.length * 32);
        if (wi) carDraws.push({ m: g.m, inst: wi, n: n * g.at.length });
      });
    }
    const planeOn = !!(this.planeModel && this.planePose);
    if (planeOn) {
      this.planeInst = this.instBuffer('plane', this.planePose!.buffer as ArrayBuffer, 32);
      this.fanInst = this.instBuffer('fans', this.fanPose.buffer as ArrayBuffer, 64);
    }
    this.planePose = null;
    const enc = d.createCommandEncoder();

    // ---------------- shadow passes (near and far cascade)
    if (shadowsOn) {
      for (let k = 0; k < 2; k++) {
        const sp = enc.beginRenderPass({
          colorAttachments: [],
          depthStencilAttachment: { view: this.shadowViews[k], depthClearValue: 1, depthLoadOp: 'clear', depthStoreOp: 'store' },
        });
        const lp = frustumPlanes(lightVP[k]);
        sp.setBindGroup(0, this.bgShadow[k]);
        sp.setPipeline(this.pipes.litShadow);
        sp.setVertexBuffer(0, this.terrainVB);
        sp.setIndexBuffer(this.terrainIB, 'uint32');
        for (const c of this.tChunks) if (aabbVisible(lp, c.min, c.max)) sp.drawIndexed(c.count, 1, c.first);
        sp.setVertexBuffer(0, this.propVB);
        for (const c of this.pChunks) if (aabbVisible(lp, c.min, c.max)) sp.draw(c.count, 1, c.first);
        sp.setPipeline(this.pipes.boxShadow);
        for (let s = 0; s < this.shapeVB.length; s++) {
          const b = boxBufs[s];
          if (!b || (s >= 4 && this.shapeVB[s].count === 0)) continue;
          // plain cubes are enough for box shadows
          const mesh = s === 0 ? { buf: this.cubeVB, count: 36 } : this.shapeVB[s];
          sp.setVertexBuffer(0, mesh.buf);
          sp.setVertexBuffer(1, b);
          sp.draw(mesh.count, v.boxes.lists[s].count);
        }
        for (const c of carDraws) {
          sp.setPipeline(this.pipes.modelShadow);
          sp.setVertexBuffer(0, c.m.vb);
          sp.setVertexBuffer(1, c.inst);
          sp.setIndexBuffer(c.m.ib, 'uint32');
          for (const p of c.m.parts) if (!p.blend) sp.drawIndexed(p.count, c.n, p.first);
        }
        sp.end();
      }
    }

    // ---------------- scene pass (linear HDR)
    let tris = 0, draws = 0;
    const pass = enc.beginRenderPass({
      colorAttachments: [{ view: this.hdrTex.createView(), clearValue: { r: 0.3, g: 0.5, b: 0.8, a: 1 }, loadOp: 'clear', storeOp: 'store' }],
      depthStencilAttachment: { view: this.depthTex.createView(), depthClearValue: 0, depthLoadOp: 'clear', depthStoreOp: 'store' },
    });
    pass.setBindGroup(0, this.bgMain);
    pass.setPipeline(this.pipes.sky);
    pass.draw(3);

    const planes = frustumPlanes(v.viewProj);
    const far = this.quality.drawDist;
    const near = (c: Chunk) => {
      const dx = Math.max(c.min[0] - cam[0], 0, cam[0] - c.max[0]);
      const dz = Math.max(c.min[2] - cam[2], 0, cam[2] - c.max[2]);
      return dx * dx + dz * dz < far * far;
    };
    pass.setPipeline(this.pipes.terrain);
    pass.setVertexBuffer(0, this.terrainVB);
    pass.setIndexBuffer(this.terrainIB, 'uint32');
    for (const c of this.tChunks) {
      if (near(c) && aabbVisible(planes, c.min, c.max)) {
        pass.drawIndexed(c.count, 1, c.first);
        tris += c.count / 3;
        draws++;
      }
    }
    pass.setPipeline(this.pipes.props);
    pass.setVertexBuffer(0, this.propVB);
    for (const c of this.pChunks) {
      if (near(c) && aabbVisible(planes, c.min, c.max)) {
        pass.draw(c.count, 1, c.first);
        tris += c.count / 3;
        draws++;
      }
    }
    const models: { m: ModelGpu; inst: GPUBuffer; n: number }[] = [];
    models.push(...carDraws);
    if (planeOn && this.planeInst) models.push({ m: this.planeModel!, inst: this.planeInst, n: 1 });
    if (planeOn && this.fanModel && this.fanInst) models.push({ m: this.fanModel, inst: this.fanInst, n: 2 });
    for (const { m, inst, n } of models) {
      pass.setPipeline(this.pipes.model);
      pass.setVertexBuffer(0, m.vb);
      pass.setVertexBuffer(1, inst);
      pass.setIndexBuffer(m.ib, 'uint32');
      for (const p of m.parts) {
        if (p.blend) continue;
        pass.setBindGroup(1, p.bg);
        pass.drawIndexed(p.count, n, p.first);
        tris += (p.count / 3) * n;
        draws++;
      }
    }
    const drawShapes = (pipe: GPURenderPipeline, bufs: (GPUBuffer | null)[], inst: Instances, cubes = false) => {
      pass.setPipeline(pipe);
      for (let s = 0; s < this.shapeVB.length; s++) {
        const b = bufs[s];
        if (!b || (s >= 4 && this.shapeVB[s].count === 0)) continue;
        const mesh = cubes ? { buf: this.cubeVB, count: 36 } : this.shapeVB[s];
        pass.setVertexBuffer(0, mesh.buf);
        pass.setVertexBuffer(1, b);
        pass.draw(mesh.count, inst.lists[s].count);
        tris += (mesh.count / 3) * inst.lists[s].count;
        draws++;
      }
    };
    drawShapes(this.pipes.box, boxBufs, v.boxes);
    if (this.bgWater) {
      pass.setPipeline(this.pipes.water);
      pass.setBindGroup(1, this.bgWater);
      pass.draw(6);
    }
    for (const c of carDraws) {
      pass.setPipeline(this.pipes.modelBlend);
      pass.setVertexBuffer(0, c.m.vb);
      pass.setVertexBuffer(1, c.inst);
      pass.setIndexBuffer(c.m.ib, 'uint32');
      for (const p of c.m.parts) {
        if (!p.blend) continue;
        pass.setBindGroup(1, p.bg);
        pass.drawIndexed(p.count, c.n, p.first);
        draws++;
      }
    }
    drawShapes(this.pipes.smoke, smokeBufs, v.smoke, true);
    if (v.glass && v.glass.count) drawShapes(this.pipes.glass, this.shapeBuffers('glass', v.glass), v.glass, true);
    if (v.zone.visible) {
      pass.setPipeline(this.pipes.zone);
      pass.setVertexBuffer(0, this.zoneVB);
      pass.draw(this.zoneCount);
    }
    drawShapes(this.pipes.glow, glowBufs, v.glow, true);
    pass.end();
    // ---------------- soft particles: their own pass, so they can read the
    // scene depth and fade out where they meet the ground and the walls
    if (v.puffs && v.puffs.count && this.bgDepth) {
      const pb = this.instBuffer('puffs', v.puffs.sorted(v.camPos), v.puffs.count * 32);
      if (pb) {
        const pp = enc.beginRenderPass({
          colorAttachments: [{ view: this.hdrTex.createView(), loadOp: 'load', storeOp: 'store' }],
          depthStencilAttachment: { view: this.depthTex.createView(), depthReadOnly: true },
        });
        pp.setBindGroup(0, this.bgMain);
        pp.setBindGroup(1, this.bgDepth);
        pp.setPipeline(this.pipes.puff);
        pp.setVertexBuffer(0, pb);
        pp.draw(6, v.puffs.count);
        pp.end();
        draws++;
      }
    }
    this.stats = { tris: Math.round(tris), draws, width: this.width, height: this.height, scale: this.dyn };

    // ---------------- ambient occlusion (white when off)
    {
      const ap = enc.beginRenderPass({ colorAttachments: [{ view: this.aoTex!.createView(), clearValue: { r: 1, g: 1, b: 1, a: 1 }, loadOp: 'clear', storeOp: 'store' }] });
      if (this.quality.ao) {
        ap.setPipeline(this.pipes.ao);
        ap.setBindGroup(0, this.bgAo!);
        ap.draw(3);
      }
      ap.end();
    }

    // ---------------- bloom chain
    if (this.quality.bloom) {
      for (let l = 0; l < 4; l++) {
        const bp = enc.beginRenderPass({ colorAttachments: [{ view: this.bloomTex[l].createView(), loadOp: 'clear', storeOp: 'store' }] });
        bp.setPipeline(l === 0 ? this.pipes.bloomFirst : this.pipes.bloomDown);
        bp.setBindGroup(0, this.bgBloom[l]);
        bp.draw(3);
        bp.end();
      }
    }

    // ---------------- tonemap + FXAA into the swap chain
    const capture = this.captureReq;
    this.captureReq = null;
    let capTex: GPUTexture | null = null;
    if (capture) {
      capTex = d.createTexture({ size: [this.width, this.height], format: this.format, usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC });
    }
    const swap = capTex ? capTex.createView() : this.ctx ? this.ctx.getCurrentTexture().createView() : this.offTex!.createView();
    const pp = enc.beginRenderPass({ colorAttachments: [{ view: swap, loadOp: 'clear', storeOp: 'store' }] });
    pp.setPipeline(this.pipes.post);
    pp.setBindGroup(0, this.bgPost!);
    pp.draw(3);
    pp.end();

    if (capTex && capture) {
      const bpr = Math.ceil((this.width * 4) / 256) * 256;
      const buf = d.createBuffer({ size: bpr * this.height, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
      enc.copyTextureToBuffer({ texture: capTex }, { buffer: buf, bytesPerRow: bpr }, [this.width, this.height]);
      d.queue.submit([enc.finish()]);
      const w = this.width, h = this.height, bgra = this.format === 'bgra8unorm';
      void buf.mapAsync(GPUMapMode.READ).then(() => {
        const src = new Uint8Array(buf.getMappedRange());
        const rgba = new Uint8Array(w * h * 4);
        for (let y = 0; y < h; y++) {
          for (let x = 0; x < w; x++) {
            const i = y * bpr + x * 4, o = (y * w + x) * 4;
            rgba[o] = src[i + (bgra ? 2 : 0)];
            rgba[o + 1] = src[i + 1];
            rgba[o + 2] = src[i + (bgra ? 0 : 2)];
            rgba[o + 3] = 255;
          }
        }
        buf.unmap();
        buf.destroy();
        capTex!.destroy();
        capture({ w, h, rgba });
      });
      return;
    }
    d.queue.submit([enc.finish()]);
    if (!this.ctx) {
      this.inflight = true;
      void d.queue.onSubmittedWorkDone().then(() => (this.inflight = false));
    }
  }
}
