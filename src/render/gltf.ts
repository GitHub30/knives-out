// Minimal GLB loader for static PBR models: node transforms are baked into the
// vertices, primitives are grouped by material. Vertex layout (48 bytes):
// pos f32x3, normal f32x3, tangent f32x4, uv f32x2.

export interface GltfMaterial {
  base: [number, number, number, number];
  emissive: [number, number, number];
  metallic: number;
  roughness: number;
  blend: boolean;
  baseTex: ImageBitmap | null;
  normalTex: ImageBitmap | null;
  ormTex: ImageBitmap | null;
  emissiveTex: ImageBitmap | null;
}

export interface GltfPart {
  material: number;
  first: number; // first index
  count: number;
}

export interface GltfModel {
  verts: Float32Array;
  indices: Uint32Array;
  parts: GltfPart[];
  materials: GltfMaterial[];
  min: [number, number, number];
  max: [number, number, number];
  /** Nodes split off by name (wheels): their mesh about its own origin and where it sits. */
  wheels: GltfWheel[];
}

export interface GltfWheel {
  /** Shared between the wheels that use the same mesh. */
  model: GltfModel;
  pos: [number, number, number];
  quat: [number, number, number, number];
}

/** Rotation of a (scale-free) column-major matrix as a quaternion. */
function quatOf(m: M4): [number, number, number, number] {
  const m00 = m[0], m10 = m[1], m20 = m[2], m01 = m[4], m11 = m[5], m21 = m[6], m02 = m[8], m12 = m[9], m22 = m[10];
  const tr = m00 + m11 + m22;
  let q: [number, number, number, number];
  if (tr > 0) {
    const k = Math.sqrt(tr + 1) * 2;
    q = [(m21 - m12) / k, (m02 - m20) / k, (m10 - m01) / k, k / 4];
  } else if (m00 > m11 && m00 > m22) {
    const k = Math.sqrt(1 + m00 - m11 - m22) * 2;
    q = [k / 4, (m01 + m10) / k, (m02 + m20) / k, (m21 - m12) / k];
  } else if (m11 > m22) {
    const k = Math.sqrt(1 + m11 - m00 - m22) * 2;
    q = [(m01 + m10) / k, k / 4, (m12 + m21) / k, (m02 - m20) / k];
  } else {
    const k = Math.sqrt(1 + m22 - m00 - m11) * 2;
    q = [(m02 + m20) / k, (m12 + m21) / k, k / 4, (m10 - m01) / k];
  }
  const l = Math.hypot(...q) || 1;
  return [q[0] / l, q[1] / l, q[2] / l, q[3] / l];
}

type M4 = number[];
const IDENT: M4 = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];
function mulM(a: M4, b: M4): M4 {
  const o = new Array<number>(16);
  for (let c = 0; c < 4; c++) for (let r = 0; r < 4; r++) o[c * 4 + r] = a[r] * b[c * 4] + a[4 + r] * b[c * 4 + 1] + a[8 + r] * b[c * 4 + 2] + a[12 + r] * b[c * 4 + 3];
  return o;
}
function trs(n: { translation?: number[]; rotation?: number[]; scale?: number[]; matrix?: number[] }): M4 {
  if (n.matrix) return n.matrix;
  const [x, y, z, w] = n.rotation ?? [0, 0, 0, 1];
  const [sx, sy, sz] = n.scale ?? [1, 1, 1];
  const [tx, ty, tz] = n.translation ?? [0, 0, 0];
  return [
    (1 - 2 * (y * y + z * z)) * sx, 2 * (x * y + z * w) * sx, 2 * (x * z - y * w) * sx, 0,
    2 * (x * y - z * w) * sy, (1 - 2 * (x * x + z * z)) * sy, 2 * (y * z + x * w) * sy, 0,
    2 * (x * z + y * w) * sz, 2 * (y * z - x * w) * sz, (1 - 2 * (x * x + y * y)) * sz, 0,
    tx, ty, tz, 1,
  ];
}

const COMPONENTS: Record<string, number> = { SCALAR: 1, VEC2: 2, VEC3: 3, VEC4: 4 };

/** `split`: nodes whose name starts with it are returned as `wheels` instead of being baked in. */
export async function loadGlb(url: string, split?: string): Promise<GltfModel> {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`model ${url}: ${res.status}`);
  const buf = await res.arrayBuffer();
  const dv = new DataView(buf);
  if (dv.getUint32(0, true) !== 0x46546c67) throw new Error('not a GLB file');
  const jsonLen = dv.getUint32(12, true);
  const json = JSON.parse(new TextDecoder().decode(new Uint8Array(buf, 20, jsonLen)));
  const binStart = 20 + jsonLen + 8;
  const view = (i: number) => {
    const bv = json.bufferViews[i];
    return { offset: binStart + (bv.byteOffset ?? 0), length: bv.byteLength as number, stride: (bv.byteStride ?? 0) as number };
  };
  const accessor = (i: number): { data: number[][]; count: number } => {
    const a = json.accessors[i];
    const bv = view(a.bufferView);
    const n = COMPONENTS[a.type];
    const size = a.componentType === 5126 || a.componentType === 5125 ? 4 : a.componentType === 5123 ? 2 : 1;
    const stride = bv.stride || n * size;
    const base = bv.offset + (a.byteOffset ?? 0);
    const data: number[][] = [];
    for (let k = 0; k < a.count; k++) {
      const row: number[] = [];
      for (let c = 0; c < n; c++) {
        const o = base + k * stride + c * size;
        row.push(a.componentType === 5126 ? dv.getFloat32(o, true) : a.componentType === 5125 ? dv.getUint32(o, true) : a.componentType === 5123 ? dv.getUint16(o, true) : dv.getUint8(o));
      }
      data.push(row);
    }
    return { data, count: a.count };
  };

  // images (decoded once, shared by the textures that use them)
  const bitmaps: Promise<ImageBitmap | null>[] = (json.images ?? []).map((im: { bufferView?: number; mimeType?: string }) => {
    if (im.bufferView === undefined) return Promise.resolve(null);
    const bv = view(im.bufferView);
    const blob = new Blob([new Uint8Array(buf, bv.offset, bv.length)], { type: im.mimeType ?? 'image/jpeg' });
    return createImageBitmap(blob, { colorSpaceConversion: 'none', premultiplyAlpha: 'none' }).catch(() => null);
  });
  const tex = async (t?: { index: number }) => (t ? await bitmaps[json.textures[t.index].source] : null);
  const materials: GltfMaterial[] = [];
  for (const m of json.materials ?? []) {
    const p = m.pbrMetallicRoughness ?? {};
    materials.push({
      base: p.baseColorFactor ?? [1, 1, 1, 1],
      emissive: m.emissiveFactor ?? [0, 0, 0],
      metallic: p.metallicFactor ?? 1,
      roughness: p.roughnessFactor ?? 1,
      blend: m.alphaMode === 'BLEND',
      baseTex: await tex(p.baseColorTexture),
      normalTex: await tex(m.normalTexture),
      ormTex: await tex(p.metallicRoughnessTexture ?? m.occlusionTexture),
      emissiveTex: await tex(m.emissiveTexture),
    });
  }

  interface Acc {
    verts: number[];
    byMat: Map<number, number[]>;
    min: [number, number, number];
    max: [number, number, number];
  }
  const acc = (): Acc => ({ verts: [], byMat: new Map(), min: [1e9, 1e9, 1e9], max: [-1e9, -1e9, -1e9] });
  const addMesh = (mi: number, m: M4, A: Acc) => {
    for (const prim of json.meshes[mi].primitives) {
      if ((prim.mode ?? 4) !== 4 || prim.indices === undefined) continue;
      const pos = accessor(prim.attributes.POSITION).data;
      const nor = prim.attributes.NORMAL !== undefined ? accessor(prim.attributes.NORMAL).data : null;
      const tan = prim.attributes.TANGENT !== undefined ? accessor(prim.attributes.TANGENT).data : null;
      const uv = prim.attributes.TEXCOORD_0 !== undefined ? accessor(prim.attributes.TEXCOORD_0).data : null;
      const base = A.verts.length / 12;
      const dir = (v: number[]) => {
        const x = m[0] * v[0] + m[4] * v[1] + m[8] * v[2], y = m[1] * v[0] + m[5] * v[1] + m[9] * v[2], z = m[2] * v[0] + m[6] * v[1] + m[10] * v[2];
        const l = Math.hypot(x, y, z) || 1;
        return [x / l, y / l, z / l];
      };
      for (let k = 0; k < pos.length; k++) {
        const p = pos[k];
        const w = [m[0] * p[0] + m[4] * p[1] + m[8] * p[2] + m[12], m[1] * p[0] + m[5] * p[1] + m[9] * p[2] + m[13], m[2] * p[0] + m[6] * p[1] + m[10] * p[2] + m[14]];
        for (let c = 0; c < 3; c++) {
          A.min[c] = Math.min(A.min[c], w[c]);
          A.max[c] = Math.max(A.max[c], w[c]);
        }
        const n = nor ? dir(nor[k]) : [0, 1, 0];
        const t = tan ? [...dir(tan[k]), tan[k][3]] : [1, 0, 0, 1];
        const u = uv ? uv[k] : [0, 0];
        A.verts.push(w[0], w[1], w[2], n[0], n[1], n[2], t[0], t[1], t[2], t[3], u[0], u[1]);
      }
      const list = A.byMat.get(prim.material ?? 0) ?? [];
      for (const row of accessor(prim.indices).data) list.push(base + row[0]);
      A.byMat.set(prim.material ?? 0, list);
    }
  };
  const finish = (A: Acc, wheels: GltfWheel[]): GltfModel => {
    const indices: number[] = [];
    const parts: GltfPart[] = [];
    for (const [material, list] of A.byMat) {
      parts.push({ material, first: indices.length, count: list.length });
      for (const i of list) indices.push(i);
    }
    return { verts: new Float32Array(A.verts), indices: new Uint32Array(indices), parts, materials, min: A.min, max: A.max, wheels };
  };

  const main = acc();
  const split_: { mesh: number; m: M4 }[] = [];
  const visit = (ni: number, parent: M4) => {
    const node = json.nodes[ni];
    const m = mulM(parent, trs(node));
    if (node.mesh !== undefined) {
      if (split && typeof node.name === 'string' && node.name.startsWith(split)) split_.push({ mesh: node.mesh, m });
      else addMesh(node.mesh, m, main);
    }
    for (const c of node.children ?? []) visit(c, m);
  };
  for (const root of json.scenes[json.scene ?? 0].nodes) visit(root, IDENT);

  const wheelModels = new Map<number, GltfModel>();
  const wheels: GltfWheel[] = split_.map(({ mesh, m }) => {
    let model = wheelModels.get(mesh);
    if (!model) {
      const A = acc();
      addMesh(mesh, IDENT, A);
      model = finish(A, []);
      wheelModels.set(mesh, model);
    }
    return { model, pos: [m[12], m[13], m[14]], quat: quatOf(m) };
  });
  return finish(main, wheels);
}
