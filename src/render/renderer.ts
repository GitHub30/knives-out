// Raw WebGPU renderer: shadow-mapped static world, instanced boxes, sky, water, zone wall.

import { BOX_SHADOW_WGSL, BOX_WGSL, LIT_SHADOW_WGSL, LIT_WGSL, SKY_WGSL, WATER_WGSL, ZONE_WGSL } from './shaders';
import { aabbVisible, cross, frustumPlanes, lookAt, mul, norm, ortho, type M4, type V3 } from './math';
import { Instances, INSTANCE_STRIDE } from './instances';

export interface Quality {
  shadowSize: number; // 0 disables shadows
  msaa: 1 | 4;
  scale: number; // render resolution multiplier (applied on top of capped DPR)
  maxDpr: number;
  drawDist: number;
}

export const QUALITY_PRESETS: Record<'low' | 'medium' | 'high', Quality> = {
  low: { shadowSize: 0, msaa: 1, scale: 0.75, maxDpr: 1.5, drawDist: 420 },
  medium: { shadowSize: 1024, msaa: 1, scale: 1, maxDpr: 1.75, drawDist: 560 },
  high: { shadowSize: 2048, msaa: 4, scale: 1, maxDpr: 2, drawDist: 760 },
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
}

const SUN = norm([-0.42, 0.78, -0.46]);

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

export class Renderer {
  readonly canvas: HTMLCanvasElement;
  readonly device: GPUDevice;
  readonly ctx: GPUCanvasContext | null;
  readonly format: GPUTextureFormat;
  quality: Quality;
  lost = false;

  private uniform: GPUBuffer;
  private uData = new Float32Array(88);
  private bglMain: GPUBindGroupLayout;
  private bglShadow: GPUBindGroupLayout;
  private bglWater: GPUBindGroupLayout;
  private bgShadow: GPUBindGroup;
  private bgMain!: GPUBindGroup;
  private bgWater!: GPUBindGroup;
  private shadowTex!: GPUTexture;
  private shadowSampler: GPUSampler;
  private depthTex: GPUTexture | null = null;
  private msaaTex: GPUTexture | null = null;
  private offTex: GPUTexture | null = null;
  private pipes!: Record<string, GPURenderPipeline>;
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

    this.uniform = device.createBuffer({ size: 352, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.bglMain = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
        { binding: 1, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'depth' } },
        { binding: 2, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'comparison' } },
      ],
    });
    this.bglShadow = device.createBindGroupLayout({
      entries: [{ binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } }],
    });
    this.bglWater = device.createBindGroupLayout({
      entries: [{ binding: 0, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'unfilterable-float' } }],
    });
    this.bgShadow = device.createBindGroup({ layout: this.bglShadow, entries: [{ binding: 0, resource: { buffer: this.uniform } }] });
    this.shadowSampler = device.createSampler({ compare: 'less', magFilter: 'linear', minFilter: 'linear' });

    const cube = cubeVerts();
    this.cubeVB = device.createBuffer({ size: cube.byteLength, usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(this.cubeVB, 0, cube);
    const cyl = cylinderVerts(160);
    this.zoneCount = cyl.length / 3;
    this.zoneVB = device.createBuffer({ size: cyl.byteLength, usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(this.zoneVB, 0, cyl);

    this.makeShadowTex();
    this.buildPipelines();
    this.resize();
  }

  private makeShadowTex() {
    this.shadowTex?.destroy();
    const size = Math.max(1, this.quality.shadowSize);
    this.shadowTex = this.device.createTexture({
      size: [size, size],
      format: 'depth32float',
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
    });
    this.bgMain = this.device.createBindGroup({
      layout: this.bglMain,
      entries: [
        { binding: 0, resource: { buffer: this.uniform } },
        { binding: 1, resource: this.shadowTex.createView() },
        { binding: 2, resource: this.shadowSampler },
      ],
    });
  }

  setQuality(q: Quality) {
    const rebuild = q.msaa !== this.quality.msaa;
    const reshadow = q.shadowSize !== this.quality.shadowSize;
    this.quality = q;
    if (reshadow) this.makeShadowTex();
    if (rebuild) this.buildPipelines();
    this.resize(true);
  }

  private buildPipelines() {
    const d = this.device;
    const sample = this.quality.msaa;
    const mod = (code: string) => d.createShaderModule({ code });
    const lit = mod(LIT_WGSL), litSh = mod(LIT_SHADOW_WGSL), box = mod(BOX_WGSL), boxSh = mod(BOX_SHADOW_WGSL);
    const sky = mod(SKY_WGSL), water = mod(WATER_WGSL), zone = mod(ZONE_WGSL);
    const layoutMain = d.createPipelineLayout({ bindGroupLayouts: [this.bglMain] });
    const layoutShadow = d.createPipelineLayout({ bindGroupLayouts: [this.bglShadow] });
    const layoutWater = d.createPipelineLayout({ bindGroupLayouts: [this.bglMain, this.bglWater] });
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
        ],
      },
    ];
    const depthMain = (write: boolean, compare: GPUCompareFunction = 'greater'): GPUDepthStencilState => ({ format: 'depth32float', depthWriteEnabled: write, depthCompare: compare });
    const target = (blend?: GPUBlendState): GPUColorTargetState[] => [{ format: this.format, blend }];
    const alpha: GPUBlendState = {
      color: { srcFactor: 'src-alpha', dstFactor: 'one-minus-src-alpha', operation: 'add' },
      alpha: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' },
    };
    const additive: GPUBlendState = {
      color: { srcFactor: 'src-alpha', dstFactor: 'one', operation: 'add' },
      alpha: { srcFactor: 'zero', dstFactor: 'one', operation: 'add' },
    };
    const ms = { count: sample };
    const shadowDepth: GPUDepthStencilState = { format: 'depth32float', depthWriteEnabled: true, depthCompare: 'less', depthBias: 3, depthBiasSlopeScale: 2.5 };
    this.pipes = {
      terrain: d.createRenderPipeline({
        layout: layoutMain,
        vertex: { module: lit, entryPoint: 'vs', buffers: [litVB] },
        fragment: { module: lit, entryPoint: 'fs_terrain', targets: target() },
        primitive: { topology: 'triangle-list', cullMode: 'back', frontFace: 'ccw' },
        depthStencil: depthMain(true),
        multisample: ms,
      }),
      props: d.createRenderPipeline({
        layout: layoutMain,
        vertex: { module: lit, entryPoint: 'vs', buffers: [litVB] },
        fragment: { module: lit, entryPoint: 'fs_props', targets: target() },
        primitive: { topology: 'triangle-list', cullMode: 'back', frontFace: 'ccw' },
        depthStencil: depthMain(true),
        multisample: ms,
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
        primitive: { topology: 'triangle-list', cullMode: 'back', frontFace: 'ccw' },
        depthStencil: depthMain(true),
        multisample: ms,
      }),
      boxShadow: d.createRenderPipeline({
        layout: layoutShadow,
        vertex: { module: boxSh, entryPoint: 'vs', buffers: boxVB },
        primitive: { topology: 'triangle-list', cullMode: 'none' },
        depthStencil: shadowDepth,
      }),
      glow: d.createRenderPipeline({
        layout: layoutMain,
        vertex: { module: box, entryPoint: 'vs', buffers: boxVB },
        fragment: { module: box, entryPoint: 'fs_glow', targets: target(additive) },
        primitive: { topology: 'triangle-list', cullMode: 'none' },
        depthStencil: depthMain(false),
        multisample: ms,
      }),
      smoke: d.createRenderPipeline({
        layout: layoutMain,
        vertex: { module: box, entryPoint: 'vs', buffers: boxVB },
        fragment: { module: box, entryPoint: 'fs_smoke', targets: target(alpha) },
        primitive: { topology: 'triangle-list', cullMode: 'back', frontFace: 'ccw' },
        depthStencil: depthMain(false),
        multisample: ms,
      }),
      sky: d.createRenderPipeline({
        layout: layoutMain,
        vertex: { module: sky, entryPoint: 'vs' },
        fragment: { module: sky, entryPoint: 'fs', targets: target() },
        primitive: { topology: 'triangle-list' },
        depthStencil: depthMain(false, 'always'),
        multisample: ms,
      }),
      water: d.createRenderPipeline({
        layout: layoutWater,
        vertex: { module: water, entryPoint: 'vs' },
        fragment: { module: water, entryPoint: 'fs', targets: target(alpha) },
        primitive: { topology: 'triangle-list', cullMode: 'none' },
        depthStencil: depthMain(false),
        multisample: ms,
      }),
      zone: d.createRenderPipeline({
        layout: layoutMain,
        vertex: { module: zone, entryPoint: 'vs', buffers: [{ arrayStride: 12, attributes: [{ shaderLocation: 0, offset: 0, format: 'float32x3' }] }] },
        fragment: { module: zone, entryPoint: 'fs', targets: target(alpha) },
        primitive: { topology: 'triangle-list', cullMode: 'none' },
        depthStencil: depthMain(false),
        multisample: ms,
      }),
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
    this.heightTex?.destroy();
    this.heightTex = d.createTexture({ size: [257, 257], format: 'r32float', usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST });
    d.queue.writeTexture({ texture: this.heightTex }, w.heights.buffer, { offset: w.heights.byteOffset, bytesPerRow: 257 * 4, rowsPerImage: 257 }, [257, 257]);
    this.bgWater = d.createBindGroup({ layout: this.bglWater, entries: [{ binding: 0, resource: this.heightTex.createView() }] });
  }

  resize(force = false) {
    const dpr = Math.min(window.devicePixelRatio || 1, this.quality.maxDpr) * this.quality.scale;
    const w = Math.max(1, Math.floor(this.canvas.clientWidth * dpr));
    const h = Math.max(1, Math.floor(this.canvas.clientHeight * dpr));
    if (!force && w === this.width && h === this.height && this.depthTex) return;
    this.width = w;
    this.height = h;
    this.canvas.width = w;
    this.canvas.height = h;
    this.depthTex?.destroy();
    this.msaaTex?.destroy();
    this.offTex?.destroy();
    this.offTex = this.ctx ? null : this.device.createTexture({ size: [w, h], format: this.format, usage: GPUTextureUsage.RENDER_ATTACHMENT });
    const sc = this.quality.msaa;
    this.depthTex = this.device.createTexture({ size: [w, h], format: 'depth32float', sampleCount: sc, usage: GPUTextureUsage.RENDER_ATTACHMENT });
    this.msaaTex = sc > 1 ? this.device.createTexture({ size: [w, h], format: this.format, sampleCount: sc, usage: GPUTextureUsage.RENDER_ATTACHMENT }) : null;
  }

  get aspect() {
    return this.width / Math.max(1, this.height);
  }

  private instBuffer(name: string, inst: Instances): GPUBuffer | null {
    if (inst.count === 0) return null;
    const bytes = inst.count * INSTANCE_STRIDE;
    let e = this.instBufs[name];
    if (!e || e.size < bytes) {
      e?.buf.destroy();
      const size = Math.max(bytes, 64 * 1024) * 1.5;
      e = { buf: this.device.createBuffer({ size: Math.ceil(size / 4) * 4, usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST }), size: Math.ceil(size / 4) * 4 };
      this.instBufs[name] = e;
    }
    this.device.queue.writeBuffer(e.buf, 0, inst.buf, 0, bytes);
    return e.buf;
  }

  private lightMatrix(center: V3): M4 {
    const R = 75;
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
    const eye: V3 = [c[0] + SUN[0] * 400, c[1] + SUN[1] * 400, c[2] + SUN[2] * 400];
    const view = lookAt(eye, c, up);
    const proj = ortho(-R, R, -R, R, 10, 800);
    return mul(proj, view);
  }

  private inflight = false;

  render(v: FrameView) {
    if (this.lost || !this.depthTex || !this.terrainVB || !this.terrainIB || !this.propVB) return;
    // Offscreen (test) mode has no swap-chain back-pressure: keep one frame in flight.
    if (!this.ctx && this.inflight && !this.captureReq) return;
    this.resize();
    const d = this.device;
    const shadowsOn = this.quality.shadowSize > 0;
    const lightVP = this.lightMatrix(v.shadowCenter);
    const u = this.uData;
    u.set(v.viewProj, 0);
    u.set(lightVP, 16);
    u.set(v.invViewProj, 32);
    u.set([v.camPos[0], v.camPos[1], v.camPos[2], v.time], 48);
    u.set([SUN[0], SUN[1], SUN[2], shadowsOn ? 1 : 0], 52);
    const lin = (c: number) => Math.pow(c, 2.2);
    u.set([1.25, 1.17, 1.05, 0.9], 56);
    u.set([lin(0.7), lin(0.78), lin(0.86), 1 / 1100], 60);
    u.set([lin(0.27), lin(0.5), lin(0.84), 1], 64);
    u.set([lin(0.74), lin(0.82), lin(0.9), 1], 68);
    u.set([1 / Math.max(1, this.quality.shadowSize), this.mapSize, v.hurt, this.waterLevel], 72);
    u.set([v.zone.cx, v.zone.cz, Math.max(0.01, v.zone.r), v.zone.visible ? 1 : 0], 76);
    u.set(v.focus ? [v.focus[0], v.focus[1], v.focus[2], 1] : [0, 0, 0, 0], 80);
    d.queue.writeBuffer(this.uniform, 0, u);

    const boxBuf = this.instBuffer('box', v.boxes);
    const glowBuf = this.instBuffer('glow', v.glow);
    const smokeBuf = this.instBuffer('smoke', v.smoke);
    const enc = d.createCommandEncoder();

    // ---------------- shadow pass
    if (shadowsOn) {
      const sp = enc.beginRenderPass({
        colorAttachments: [],
        depthStencilAttachment: { view: this.shadowTex.createView(), depthClearValue: 1, depthLoadOp: 'clear', depthStoreOp: 'store' },
      });
      const lp = frustumPlanes(lightVP);
      sp.setBindGroup(0, this.bgShadow);
      sp.setPipeline(this.pipes.litShadow);
      sp.setVertexBuffer(0, this.terrainVB);
      sp.setIndexBuffer(this.terrainIB, 'uint32');
      for (const c of this.tChunks) if (aabbVisible(lp, c.min, c.max)) sp.drawIndexed(c.count, 1, c.first);
      sp.setVertexBuffer(0, this.propVB);
      for (const c of this.pChunks) if (aabbVisible(lp, c.min, c.max)) sp.draw(c.count, 1, c.first);
      if (boxBuf) {
        sp.setPipeline(this.pipes.boxShadow);
        sp.setVertexBuffer(0, this.cubeVB);
        sp.setVertexBuffer(1, boxBuf);
        sp.draw(36, v.boxes.count);
      }
      sp.end();
    }

    // ---------------- main pass
    const capture = this.captureReq;
    this.captureReq = null;
    let capTex: GPUTexture | null = null;
    if (capture) {
      capTex = d.createTexture({ size: [this.width, this.height], format: this.format, usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC });
    }
    const swap = capTex ? capTex.createView() : this.ctx ? this.ctx.getCurrentTexture().createView() : this.offTex!.createView();
    const msaa = this.msaaTex;
    const pass = enc.beginRenderPass({
      colorAttachments: [
        msaa
          ? { view: msaa.createView(), resolveTarget: swap, clearValue: { r: 0.7, g: 0.8, b: 0.9, a: 1 }, loadOp: 'clear', storeOp: 'discard' }
          : { view: swap, clearValue: { r: 0.7, g: 0.8, b: 0.9, a: 1 }, loadOp: 'clear', storeOp: 'store' },
      ],
      depthStencilAttachment: { view: this.depthTex.createView(), depthClearValue: 0, depthLoadOp: 'clear', depthStoreOp: 'discard' },
    });
    pass.setBindGroup(0, this.bgMain);
    pass.setPipeline(this.pipes.sky);
    pass.draw(3);

    const planes = frustumPlanes(v.viewProj);
    const cam = v.camPos;
    const far = this.quality.drawDist;
    const near = (c: Chunk) => {
      const dx = Math.max(c.min[0] - cam[0], 0, cam[0] - c.max[0]);
      const dz = Math.max(c.min[2] - cam[2], 0, cam[2] - c.max[2]);
      return dx * dx + dz * dz < far * far;
    };
    pass.setPipeline(this.pipes.terrain);
    pass.setVertexBuffer(0, this.terrainVB);
    pass.setIndexBuffer(this.terrainIB, 'uint32');
    for (const c of this.tChunks) if (near(c) && aabbVisible(planes, c.min, c.max)) pass.drawIndexed(c.count, 1, c.first);
    pass.setPipeline(this.pipes.props);
    pass.setVertexBuffer(0, this.propVB);
    for (const c of this.pChunks) if (near(c) && aabbVisible(planes, c.min, c.max)) pass.draw(c.count, 1, c.first);
    if (boxBuf) {
      pass.setPipeline(this.pipes.box);
      pass.setVertexBuffer(0, this.cubeVB);
      pass.setVertexBuffer(1, boxBuf);
      pass.draw(36, v.boxes.count);
    }
    if (this.bgWater) {
      pass.setPipeline(this.pipes.water);
      pass.setBindGroup(1, this.bgWater);
      pass.draw(6);
    }
    if (smokeBuf) {
      pass.setPipeline(this.pipes.smoke);
      pass.setVertexBuffer(0, this.cubeVB);
      pass.setVertexBuffer(1, smokeBuf);
      pass.draw(36, v.smoke.count);
    }
    if (v.zone.visible) {
      pass.setPipeline(this.pipes.zone);
      pass.setVertexBuffer(0, this.zoneVB);
      pass.draw(this.zoneCount);
    }
    if (glowBuf) {
      pass.setPipeline(this.pipes.glow);
      pass.setVertexBuffer(0, this.cubeVB);
      pass.setVertexBuffer(1, glowBuf);
      pass.draw(36, v.glow.count);
    }
    pass.end();
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
