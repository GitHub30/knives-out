#!/usr/bin/env python3
"""Build the Unitree G1 character parts from the MuJoCo Menagerie model.

usage: build-g1.py <mujoco_menagerie/unitree_g1> <out.bin>

The G1's links (rest pose of g1.xml) are grouped into the game character's
rigid segments (torso, head, upper arms, forearms, thighs, shins), scaled by
1.4 (the G1 is 1.32 m tall, the game character 1.85 m) and expressed in each
segment's joint frame, so the client can hang them on the same joints the
server's hit volumes use. Meshes are decimated by vertex clustering.

Output: u32 magic 'G1P1', u32 count, count x u32 vertex count, then for each
mesh its vertices as f32 x 6 (position, normal), unindexed triangles.
Mesh order: for each segment in SEGMENTS, the dark part then the light part.
"""
import struct, sys
import xml.etree.ElementTree as ET
import numpy as np

src, dst = sys.argv[1], sys.argv[2]
SCALE = 1.4
SEGMENTS = ['torso', 'head', 'arm_l', 'arm_r', 'fore_l', 'fore_r', 'thigh_l', 'thigh_r', 'shin_l', 'shin_r']
TARGET = {'torso': 2600, 'head': 900, 'arm': 700, 'fore': 900, 'thigh': 900, 'shin': 1000}

def qmat(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])

def vec(s, n, default):
    return np.array([float(v) for v in s.split()]) if s else np.array(default, float)

def read_stl(path):
    data = open(path, 'rb').read()
    n = struct.unpack('<I', data[80:84])[0]
    rec = np.frombuffer(data, dtype=np.dtype([('n', '<f4', 3), ('v', '<f4', (3, 3)), ('a', '<u2')]), count=n, offset=84)
    return rec['v'].astype(np.float64)  # (n, 3, 3)

root = ET.parse(f'{src}/g1.xml').getroot()
files = {(m.get('name') or m.get('file').rsplit('.', 1)[0]): m.get('file') for m in root.find('asset').iter('mesh')}

def segment(body):
    side = 'l' if body.startswith('left') else 'r'
    if any(k in body for k in ('hip_pitch', 'hip_roll', 'hip_yaw')):
        return f'thigh_{side}'
    if any(k in body for k in ('knee', 'ankle')):
        return f'shin_{side}'
    if 'shoulder' in body:
        return f'arm_{side}'
    if any(k in body for k in ('elbow', 'wrist')):
        return f'fore_{side}'
    return 'torso'

tris = {s: {'dark': [], 'light': []} for s in SEGMENTS}
joints = {}

def walk(node, R, p):
    name = node.get('name')
    if name in ('left_hip_pitch_link', 'right_hip_pitch_link', 'left_knee_link', 'right_knee_link', 'left_shoulder_pitch_link', 'right_shoulder_pitch_link', 'left_elbow_link', 'right_elbow_link'):
        joints[segment(name)] = p.copy()
    for g in node.findall('geom'):
        if g.get('class') != 'visual':
            continue
        mesh = g.get('mesh')
        seg = 'head' if mesh == 'head_link' else segment(name)
        gp, gq = vec(g.get('pos'), 3, [0, 0, 0]), vec(g.get('quat'), 4, [1, 0, 0, 0])
        v = read_stl(f"{src}/assets/{files[mesh]}")
        v = v @ qmat(gq).T + gp
        v = v @ R.T + p
        tris[seg]['dark' if g.get('material') == 'black' else 'light'].append(v)
    for b in node.findall('body'):
        walk(b, R @ qmat(vec(b.get('quat'), 4, [1, 0, 0, 0])), p + R @ vec(b.get('pos'), 3, [0, 0, 0]))

pelvis = root.find('worldbody').find('body')
walk(pelvis, np.eye(3), vec(pelvis.get('pos'), 3, [0, 0, 0]))

hip_z = joints['thigh_l'][2]
# joint the segment hangs on, in G1 coordinates
joints['torso'] = np.array([0, 0, hip_z + 0.02 / SCALE])           # game torso frame: 2 cm above the hips
joints['head'] = np.array([0, 0, hip_z + 0.62 / SCALE])            # head pivot: 0.6 up the torso

def to_game(v, seg):
    v = (v - joints[seg]) * SCALE
    # G1: x forward, y left, z up  ->  game: x left, y up, z forward
    out = np.stack([v[..., 1], v[..., 2], v[..., 0]], axis=-1)
    if seg.startswith('fore'):
        # the G1's forearm points forward at rest; the game's hangs down its frame
        out = np.stack([out[..., 0], -out[..., 2], out[..., 1]], axis=-1)
        out[..., 1] *= 0.78  # ... and is shorter (hand included)
    if seg == 'torso':
        out[..., 0] *= 1.42  # the game character is broader across the shoulders
    if seg.startswith('arm'):
        # the elbow goes on the segment's axis (the G1's sits a little outward)
        side = 1.0 if seg.endswith('_l') else -1.0
        elbow = (joints['fore' + seg[3:]] - joints[seg]) * SCALE
        out[..., 0] -= side * abs(elbow[1])
        out[..., 2] -= elbow[0] * np.clip(out[..., 1] / elbow[2], 0.0, 1.2)
    if seg.startswith('thigh'):
        # put the knee end on the segment's axis: the game's shin hangs straight
        # below the hip joint (the G1's knee sits outward and a little back)
        side = 1.0 if seg.endswith('_l') else -1.0
        knee = (joints['shin' + seg[5:]] - joints[seg]) * SCALE
        t = np.clip(out[..., 1] / knee[2], 0.0, 1.2)
        out[..., 0] -= side * abs(knee[1]) * np.maximum(t, 0.6)
        out[..., 2] -= knee[0] * t
    return out

def decimate(v, target):
    """Vertex clustering to about `target` triangles; smooth normals."""
    if len(v) == 0:
        return np.zeros((0, 6), np.float32)
    lo, hi = 1e-4, 0.2
    best = None
    for _ in range(18):
        cell = (lo * hi) ** 0.5
        key = np.floor(v / cell).astype(np.int64)
        flat = key.reshape(-1, 3)
        uniq, inv = np.unique(flat, axis=0, return_inverse=True)
        idx = inv.reshape(-1, 3)
        ok = (idx[:, 0] != idx[:, 1]) & (idx[:, 1] != idx[:, 2]) & (idx[:, 0] != idx[:, 2])
        n = int(ok.sum())
        best = (cell, uniq, idx[ok], flat, inv)
        if n > target:
            lo = cell
        else:
            hi = cell
    cell, uniq, idx, flat, inv = best
    # cluster position: mean of its vertices
    pos = np.zeros((len(uniq), 3))
    cnt = np.zeros(len(uniq))
    np.add.at(pos, inv, v.reshape(-1, 3))
    np.add.at(cnt, inv, 1)
    pos /= cnt[:, None]
    a, b, c = pos[idx[:, 0]], pos[idx[:, 1]], pos[idx[:, 2]]
    fn = np.cross(b - a, c - a)
    nrm = np.zeros_like(pos)
    for k in range(3):
        np.add.at(nrm, idx[:, k], fn)
    # keep creases: a corner uses the face normal when it is far from the smooth one
    nl = np.linalg.norm(nrm, axis=1, keepdims=True)
    nrm = nrm / np.maximum(nl, 1e-12)
    fl = np.linalg.norm(fn, axis=1, keepdims=True)
    fnu = fn / np.maximum(fl, 1e-12)
    out = np.zeros((len(idx), 3, 6), np.float32)
    for k in range(3):
        sm = nrm[idx[:, k]]
        sharp = (sm * fnu).sum(axis=1, keepdims=True) < 0.55
        out[:, k, :3] = pos[idx[:, k]]
        out[:, k, 3:] = np.where(sharp, fnu, sm)
    return out.reshape(-1, 6)

meshes = []
for seg in SEGMENTS:
    for tone in ('dark', 'light'):
        parts = tris[seg][tone]
        v = to_game(np.concatenate(parts), seg) if parts else np.zeros((0, 3, 3))
        total = sum(len(x) for x in tris[seg].values() for x in [np.concatenate(x) if x else []])
        share = len(v) / max(total, 1)
        m = decimate(v, max(60, int(TARGET[seg.split('_')[0]] * share)))
        meshes.append(m)
        if len(m):
            p = m[:, :3]
            print(f'{seg:8} {tone:5} {len(v):7} -> {len(m) // 3:5} tris  bounds {p.min(0).round(2)} {p.max(0).round(2)}')
with open(dst, 'wb') as f:
    f.write(b'G1P1')
    f.write(struct.pack('<I', len(meshes)))
    for m in meshes:
        f.write(struct.pack('<I', len(m)))
    for m in meshes:
        f.write(m.astype('<f4').tobytes())
print('total tris', sum(len(m) for m in meshes) // 3, 'bytes', 8 + 4 * len(meshes) + sum(m.nbytes for m in meshes))
