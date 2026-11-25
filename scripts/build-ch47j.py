#!/usr/bin/env python3
"""Build the CH-47J transport helicopter (JASDF camouflage) as GLB files.

Everything is generated here: the airframe is lofted from dimensioned
sections (15.5 m fuselage, 3.7 m over the side pods, 18.3 m rotors) and the
livery is painted into textures. No photographs or third-party meshes.

usage: build-ch47j.py out.glb
writes out.glb (everything that does not move), out_rotor.glb (one rotor about
its local Y axis, origin at the hub centre), out_gear_front.glb and
out_gear_rear.glb (the sliding part of one landing-gear unit, origin on the
strut axis at the pod underside with the strut compressed).
Model space: +Z nose, +Y up, +X left; origin on the ground under the middle
of the fuselage, metres. The constants printed at the end are mirrored in
src/render/heli.ts.

The landing gear of the real aircraft is fixed (only the oleo struts extend
when the weight comes off the wheels). The game was asked for a retraction,
so the units can also slide up into wells under the side pods: that part,
like the rocket pods, is a game addition and not on the real transport.
"""
import io, json, math, struct, sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

L = 15.5             # fuselage length
LT = 15.62           # texture length (the rounded tail end included)
H = 2.5              # cabin depth, belly to roof
W = 1.25             # cabin half width
Y0 = 0.55            # belly line above the ground, struts compressed
Z0 = L / 2           # station that becomes z = 0
FONT_JA = '/usr/share/fonts/opentype/ipafont-gothic/ipag.ttf'
FONT_EN = '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'

TAN = (200, 190, 165)
GREEN = (70, 104, 100)
OLIVE = (101, 110, 60)
BROWN = (138, 98, 82)
UNDER = (203, 196, 170)
GLASS = (30, 44, 50)
RED = (200, 34, 44)
INK = (26, 26, 26)
YELLOW = (232, 200, 32)

# rotor hubs (station, height above the belly line, forward tilt of the shaft)
HUB_F = (2.0, 3.95, math.radians(5.0))
HUB_R = (13.9, 5.05, math.radians(2.5))
ROTOR_R = 9.15
# The blades are built straight: the renderer bends them (tip sag at rest,
# slight coning at speed), see BLADE_REST / BLADE_CONE and vs_rotor.
BLADE_ROOT = 0.6     # flap hinge: the bend starts here
BLADE_REST = -1.3    # tip height against the hub, rotor stopped
BLADE_CONE = 0.25    # and at full speed
# landing gear: station, half track, wheel radius, strut travel
GEAR_F = (5.35, 1.4)
GEAR_R = (12.2, 1.3)
WHEEL_R = 0.32
GEAR_TRAVEL = 0.2
# Game addition (the real gear is fixed): stowed, a unit has slid up into its
# pod until the wheels sit half sunk in the wells painted on the underside.
GEAR_SUNK = 0.2      # axle above the pod underside when stowed
POD_BOT = 0.22       # pod underside above the belly line
# rocket pods (game addition): station of the tube mouths, offset, height
LAUNCH = (5.3, 2.16, 0.74)
WINDOWS = [4.0, 5.9, 7.75, 9.6, 11.8]   # cabin window stations (port; starboard has the door for the first)
WIN_Y = 1.55
ROUNDEL = (6.82, 1.6, 0.31, 0.40)   # station, height, red radius, white radius


# ------------------------------------------------------------------ curves
def pchip(pts):
    """Monotone cubic through (x, y) points: no overshoot between them."""
    p = np.asarray(pts, float)
    x, y = p[:, 0], p[:, 1]
    h, d = np.diff(x), np.diff(y) / np.diff(x)
    m = np.zeros(len(x))
    m[0], m[-1] = d[0], d[-1]
    for k in range(1, len(x) - 1):
        if d[k - 1] * d[k] > 0:
            w1, w2 = 2 * h[k] + h[k - 1], h[k] + 2 * h[k - 1]
            m[k] = (w1 + w2) / (w1 / d[k - 1] + w2 / d[k])

    def f(q):
        q = np.asarray(q, float)
        i = np.clip(np.searchsorted(x, q) - 1, 0, len(x) - 2)
        t = np.clip((q - x[i]) / h[i], 0, 1)
        h00, h10, h01, h11 = 2 * t ** 3 - 3 * t ** 2 + 1, t ** 3 - 2 * t ** 2 + t, -2 * t ** 3 + 3 * t ** 2, t ** 3 - t ** 2
        return h00 * y[i] + h10 * h[i] * m[i] + h01 * y[i + 1] + h11 * h[i] * m[i + 1]
    return f


def smooth(a, b, x):
    t = np.clip((np.asarray(x, float) - a) / (b - a), 0, 1)
    return t * t * (3 - 2 * t)


# Side view of the nose: the radome sits low, the cowl rises to the
# windscreen base and the raked windscreen runs up into the roof.
_top = pchip([(0.28, 1.3), (0.45, 1.5), (0.72, 1.62), (1.1, 2.1), (1.4, 2.41), (1.75, H), (LT, H)])
_bot = pchip([(0.28, 0.5), (0.5, 0.3), (0.9, 0.12), (1.4, 0.03), (1.8, 0.0), (LT, 0.0)])
_hw = pchip([(0.28, 0.66), (0.5, 0.95), (0.9, 1.13), (1.4, 1.22), (1.9, W), (12.5, W), (L, 1.12), (LT, 1.12)])


def fus_top(s):
    return _top(s)


def fus_bot(s):
    s = np.asarray(s, float)
    # the belly sweeps up under the rear ramp
    return _bot(s) + 1.38 * np.clip((s - 11.2) / 4.3, 0, 1) ** 1.25


def section(w, top, bot, rt, rb, t):
    """Rounded rectangle walked at even steps of its outline: t = 0 on the
    belly centre line, up the left side, 0.5 on top, down the right side."""
    t = np.asarray(t, float) % 1.0
    u = np.where(t <= 0.5, t * 2, (1 - t) * 2)
    sign = np.where(t <= 0.5, 1.0, -1.0)
    h = top - bot
    rb = np.minimum(np.minimum(rb, w), h / 2)
    rt = np.minimum(np.minimum(rt, w), h / 2)
    a, b, c, d, e = w - rb, math.pi / 2 * rb, h - rb - rt, math.pi / 2 * rt, w - rt
    dist = u * (a + b + c + d + e)
    a2 = (dist - a) / np.maximum(rb, 1e-6)
    a4 = (dist - a - b - c) / np.maximum(rt, 1e-6)
    x = np.where(dist < a, dist, np.where(dist < a + b, w - rb + rb * np.sin(a2), np.where(dist < a + b + c, w, np.where(dist < a + b + c + d, w - rt + rt * np.cos(a4), a + b + c + d + e - dist))))
    y = np.where(dist < a, bot, np.where(dist < a + b, bot + rb - rb * np.cos(a2), np.where(dist < a + b + c, bot + rb + (dist - a - b), np.where(dist < a + b + c + d, top - rt + rt * np.sin(a4), top))))
    return sign * x, y + 0 * x


def fus_point(s, t):
    """Fuselage skin at station s (0.26 .. LT); heights above the belly line."""
    s = np.asarray(s, float)
    top, bot, w = fus_top(s), fus_bot(s), _hw(s)
    k = 0.25 * np.clip((2.2 - s) / 1.9, 0, 1)
    x, y = section(w, top, bot, 0.55 + k, 0.45 + k, t)
    # both ends are closed by shrinking the last rings onto their centres
    f = np.interp(s, [0.26, 0.28, 0.3, L, L + 0.06, L + 0.1, LT], [0.001, 0.6, 1, 1, 0.9, 0.6, 0.001])
    yc = (top + bot) / 2
    return x * f, yc + (y - yc) * f


POD_S0, POD_S1 = 3.4, 12.9
POD_S0_STBD = 3.95   # the starboard pod starts behind the crew door
POD_TAPER = 11.7     # where the tail of the pod starts to close
POD_XC, POD_HW, POD_TOP = 1.31, 0.56, 1.25


def pod_s0(side):
    return np.where(np.asarray(side) > 0, POD_S0, POD_S0_STBD)


def pod_point(side, s, t):
    """Side pod (fuel and landing gear fairing): a rounded box with a
    sloping top shelf, a bullet nose and a tail that follows the upsweep."""
    s = np.asarray(s, float)
    x, y = section(POD_HW, POD_TOP, POD_BOT, 0.3, 0.3, t)
    y = y - 0.14 * np.clip((x + POD_HW) / (2 * POD_HW), 0, 1) * np.clip((y - POD_BOT) / (POD_TOP - POD_BOT), 0, 1)
    d = np.clip((s - pod_s0(side)) / 1.15, 0, 1)
    e = np.clip((s - POD_TAPER) / (POD_S1 - POD_TAPER), 0, 1)
    k = np.sqrt(np.maximum(1 - (1 - d) ** 2, 1e-6)) * (1 - 0.97 * e ** 2.2)
    yc = 0.74 + fus_bot(s) * 0.85
    # shrink towards the cabin wall so the ends stay attached to it
    xs = (W - 0.25) + (POD_XC + x - (W - 0.25)) * np.where(POD_XC + x > W - 0.25, k, 1.0)
    return side * xs, yc + (y - 0.74 + fus_bot(s) * 0.0) * k


def gear_mount(rear):
    """Where a gear unit meets its pod: (station, half track, height of the pod underside there)."""
    gs, gx = GEAR_R if rear else GEAR_F
    lo = min(float(pod_point(1, gs, t)[1]) for t in np.linspace(0.0, 0.12, 7))
    return gs, gx, lo


# rotor pylons: rows of (height, front station, rear station, half width)
PYL_F = [(2.0, 1.4, 4.3, 1.1), (2.5, 1.42, 3.9, 0.95), (2.8, 1.36, 3.35, 0.8), (3.1, 1.32, 2.95, 0.73), (3.35, 1.3, 2.76, 0.705), (3.55, 1.29, 2.71, 0.7)]
PYL_R = [(2.0, 11.2, 15.6, 1.1), (2.5, 11.7, 15.58, 0.85), (2.9, 12.0, 15.55, 0.6), (3.4, 12.2, 15.5, 0.5), (4.0, 12.42, 15.42, 0.46), (4.65, 12.68, 15.28, 0.44)]


def pylon_point(rows, n, hub, v, phi):
    """Pylon skin: v = 0 at the base .. 1 at the top (above 1 the cap closes),
    phi around the section (0 at the rear). Returns x, y, s."""
    r = np.asarray(rows, float)
    y0, y1 = r[0, 0], r[-1, 0]
    v = np.asarray(v, float)
    y = y0 + np.clip(v, 0, 1) * (y1 - y0)
    sf, sr, b = pchip(r[:, [0, 1]])(y), pchip(r[:, [0, 2]])(y), pchip(r[:, [0, 3]])(y)
    k = np.interp(v, [1.0, 1.03, 1.05], [1, 0.9, 0.001])
    y = y + np.interp(v, [1.0, 1.03, 1.05], [0, 0.05, 0.07])
    c, sn = np.cos(phi), np.sin(phi)
    sc, a = (sf + sr) / 2, (sr - sf) / 2
    s = sc + a * k * np.sign(c) * np.abs(c) ** (2 / n)
    x = b * k * np.sign(sn) * np.abs(sn) ** (2 / n)
    # the top is square to the rotor shaft, which leans forward
    y = y + (s - hub[0]) * math.tan(hub[2]) * smooth(0.45, 1.0, v)
    return x + 0 * s, y + 0 * s, s


# ------------------------------------------------------------------ mesh helpers
def P(x, y, s):
    """Airframe coordinates (x left, height above the belly line, station) to model space."""
    return np.stack([np.asarray(x, float) + 0 * np.asarray(s, float), np.asarray(y, float) + Y0 + 0 * np.asarray(s, float), Z0 - np.asarray(s, float) + 0 * np.asarray(x, float)], -1)


class Mesh:
    def __init__(self):
        self.v, self.uv, self.idx, self.uid = [], [], [], []
        self.n = 0
        self.flat = 0

    def add(self, verts, uvs, tris, flat=False):
        v = np.asarray(verts, np.float32).reshape(-1, 3)
        uv = np.asarray(uvs, np.float32).reshape(-1, 2)
        t = np.asarray(tris, np.uint32).reshape(-1, 3)
        if flat:
            # faceted: every triangle gets its own corners
            v, uv = v[t].reshape(-1, 3), uv[t].reshape(-1, 2)
            t = np.arange(len(v), dtype=np.uint32).reshape(-1, 3)
            uid = np.repeat(np.arange(len(t)) + 1 + self.flat, 3)
            self.flat += len(t)
        else:
            uid = np.zeros(len(v), np.int64)
        self.v.append(v)
        self.uv.append(uv)
        self.idx.append(t + self.n)
        self.uid.append(uid)
        self.n += len(v)

    def done(self):
        v = np.concatenate(self.v)
        uv = np.concatenate(self.uv)
        idx = np.concatenate(self.idx)
        uid = np.concatenate(self.uid)
        # smooth normals, welded by position so UV seams do not show
        fn = np.cross(v[idx[:, 1]] - v[idx[:, 0]], v[idx[:, 2]] - v[idx[:, 0]])
        key = np.concatenate([np.round(v * 2000).astype(np.int64), uid[:, None]], 1)
        _, inv = np.unique(key, axis=0, return_inverse=True)
        inv = inv.reshape(-1)
        acc = np.zeros((inv.max() + 1, 3))
        for k in range(3):
            np.add.at(acc, inv[idx[:, k]], fn)
        n = acc[inv]
        ln = np.linalg.norm(n, axis=1, keepdims=True)
        n = np.where(ln > 1e-12, n / np.maximum(ln, 1e-12), [0, 1, 0])
        return v, n.astype(np.float32), uv, idx


def grid(points, uvs=None, outward=None):
    """Triangulate a (rows, cols, 3) grid. Winding is fixed so the surface
    faces away from its own centre (or along `outward`)."""
    p = np.asarray(points, float)
    rows, cols = p.shape[:2]
    a = (np.arange(rows - 1)[:, None] * cols + np.arange(cols - 1)[None, :]).reshape(-1)
    tris = np.concatenate([np.stack([a, a + 1, a + cols + 1], 1), np.stack([a, a + cols + 1, a + cols], 1)])
    v = p.reshape(-1, 3)
    fn = np.cross(v[tris[:, 1]] - v[tris[:, 0]], v[tris[:, 2]] - v[tris[:, 0]])
    cen = v[tris].mean(1)
    ref = cen - v.mean(0) if outward is None else np.broadcast_to(np.asarray(outward, float), cen.shape)
    if (fn * ref).sum() < 0:
        tris = tris[:, ::-1]
    if uvs is None:
        uvs = np.zeros((rows, cols, 2)) + 0.5
    return v, np.asarray(uvs, float).reshape(-1, 2), tris


def frame(axis):
    a = np.asarray(axis, float)
    a = a / np.linalg.norm(a)
    h = np.array([0.0, 1.0, 0.0]) if abs(a[1]) < 0.9 else np.array([1.0, 0.0, 0.0])
    b1 = np.cross(h, a)
    b1 /= np.linalg.norm(b1)
    return a, b1, np.cross(a, b1)


def lathe(profile, p0, axis, seg=16, uv=None):
    """Surface of revolution about `axis` through p0; profile = (distance along the axis, radius)."""
    prof = np.asarray(profile, float)
    a, b1, b2 = frame(axis)
    th = np.linspace(0, 2 * math.pi, seg + 1)
    pts = np.asarray(p0, float) + prof[:, 0, None, None] * a + prof[:, 1, None, None] * (np.cos(th)[None, :, None] * b1 + np.sin(th)[None, :, None] * b2)
    uvs = None if uv is None else uv(prof, th)
    return grid(pts, uvs)


def tube(p0, p1, r0, r1=None, seg=10):
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    r1 = r0 if r1 is None else r1
    ln = float(np.linalg.norm(p1 - p0))
    return lathe([(0, 0.0005), (0, r0), (ln, r1), (ln, 0.0005)], p0, p1 - p0, seg)


def box(c, half, axes=None):
    """Box about c; `axes` = three unit vectors (rows) or None for the model axes."""
    ax = np.eye(3) if axes is None else np.asarray(axes, float)
    v = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], float) * np.asarray(half, float)
    v = np.asarray(c, float) + v @ ax
    q = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]
    tris = [(a, b, cc) for (a, b, cc, d) in q] + [(a, cc, d) for (a, b, cc, d) in q]
    return v, np.zeros((8, 2)) + 0.5, np.array(tris)


def ellipsoid(center, radii, seg=16, rings=10):
    u = np.linspace(0, math.pi, rings + 1)[:, None]
    t = np.linspace(0, 2 * math.pi, seg + 1)[None, :]
    pts = np.stack([center[0] + radii[0] * np.sin(u) * np.sin(t), center[1] - radii[1] * np.sin(u) * np.cos(t), center[2] + radii[2] * np.cos(u) + 0 * t], -1)
    return grid(pts)


def airfoil(n=9, thick=0.1):
    """Closed loop from the trailing edge over the top to the leading edge and
    back underneath: (chord fraction, height fraction)."""
    x = (1 - np.cos(np.linspace(0, math.pi, n))) / 2
    yt = 5 * thick * (0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x ** 2 + 0.2843 * x ** 3 - 0.1036 * x ** 4)
    return np.concatenate([np.stack([x[::-1], yt[::-1]], 1), np.stack([x[1:], -yt[1:]], 1)])


# ------------------------------------------------------------------ painting
def bilinear(arr, px, py):
    h, w = arr.shape[:2]
    px, py = np.clip(px, 0, w - 1.001), np.clip(py, 0, h - 1.001)
    x0, y0 = np.floor(px).astype(np.int32), np.floor(py).astype(np.int32)
    fx, fy = (px - x0)[..., None], (py - y0)[..., None]
    return (arr[y0, x0] * (1 - fx) + arr[y0, x0 + 1] * fx) * (1 - fy) + (arr[y0 + 1, x0] * (1 - fx) + arr[y0 + 1, x0 + 1] * fx) * fy


class Sheet:
    """A flat drawing in metres that is projected onto the skin. Drawn at
    twice the final resolution and reduced, so edges come out smooth."""
    SS = 2

    def __init__(self, u0, u1, v0, v1, ppm, flip=False):
        self.u0, self.u1, self.v0, self.v1, self.ppm, self.flip = u0, u1, v0, v1, ppm, flip
        self.k = ppm * self.SS
        size = (int((u1 - u0) * self.k), int((v1 - v0) * self.k))
        self.im = Image.new('RGBA', size, (0, 0, 0, 0))
        self.gl = Image.new('L', size, 0)
        self.d, self.g = ImageDraw.Draw(self.im), ImageDraw.Draw(self.gl)

    def p(self, u, v):
        return (((self.u1 - u) if self.flip else (u - self.u0)) * self.k, (self.v1 - v) * self.k)

    def poly(self, pts, fill, gloss=False, outline=None, width=0.0):
        pp = [self.p(*q) for q in pts]
        if fill is not None:
            self.d.polygon(pp, fill=fill)
        if gloss:
            self.g.polygon(pp, fill=255)
        if outline is not None:
            self.d.line(pp + [pp[0]], fill=outline, width=max(1, int(width * self.k)), joint='curve')

    def line(self, pts, fill, width):
        self.d.line([self.p(*q) for q in pts], fill=fill, width=max(1, int(round(width * self.k))), joint='curve')

    def disc(self, u, v, r, fill=None, gloss=False, outline=None, width=0.0):
        (x, y), rr = self.p(u, v), r * self.k
        box_ = [x - rr, y - rr, x + rr, y + rr]
        self.d.ellipse(box_, fill=fill, outline=outline, width=max(1, int(round(width * self.k))))
        if gloss:
            self.g.ellipse(box_, fill=255)

    def rrect(self, u0, u1, v0, v1, rad, fill=None, outline=None, width=0.0, gloss=False):
        (xa, ya), (xb, yb) = self.p(u0, v0), self.p(u1, v1)
        box_ = [min(xa, xb), min(ya, yb), max(xa, xb), max(ya, yb)]
        self.d.rounded_rectangle(box_, radius=rad * self.k, fill=fill, outline=outline, width=max(1, int(round(width * self.k))))
        if gloss:
            self.g.rounded_rectangle(box_, radius=rad * self.k, fill=255)

    def text(self, s, u, v, height, font=FONT_EN, fill=INK, squeeze=1.0):
        """Upright text centred on (u, v), `height` metres tall."""
        f = ImageFont.truetype(font, 200)
        l, t, r, b = f.getbbox(s)
        im = Image.new('RGBA', (r - l + 8, b - t + 8), (0, 0, 0, 0))
        ImageDraw.Draw(im).text((4 - l, 4 - t), s, font=f, fill=fill + (255,))
        hh = max(1, int(height * self.k))
        ww = max(1, int(hh * im.width / im.height * squeeze))
        im = im.resize((ww, hh), Image.LANCZOS)
        x, y = self.p(u, v)
        self.im.alpha_composite(im, (int(x - ww / 2), int(y - hh / 2)))

    def bake(self):
        w, h = self.im.width // self.SS, self.im.height // self.SS
        self.rgba = np.asarray(self.im.convert('RGBa').resize((w, h), Image.LANCZOS).convert('RGBA'), np.float32) / 255
        self.gloss = np.asarray(self.gl.resize((w, h), Image.LANCZOS), np.float32)[..., None] / 255
        self.im = self.gl = self.d = self.g = None
        return self

    def sample(self, u, v):
        px = ((self.u1 - u) if self.flip else (u - self.u0)) * self.ppm - 0.5
        py = (self.v1 - v) * self.ppm - 0.5
        return bilinear(self.rgba, px, py), bilinear(self.gloss, px, py)[..., 0]


CAMO_PPM, CAMO_S0, CAMO_Q0, CAMO_S1, CAMO_Q1 = 32, -0.6, -0.6, 16.4, 6.8

# Camouflage blotches per side in (station, girth height): the height above
# the belly, continued over the roof towards the centre line. Corner lists are
# blurred into flowing shapes.
CAMO = {
    1: {  # port
        'G': [[(-0.5, 0.9), (-0.5, 3.3), (2.4, 3.3), (3.3, 3.0), (3.1, 2.0), (2.7, 1.35), (1.6, 1.1), (0.9, 0.7)],
              [(3.3, 0.2), (3.35, 1.4), (4.7, 1.45), (5.3, 0.9), (5.1, 0.2)],
              [(3.2, 3.05), (4.4, 3.6), (4.4, 4.4), (3.0, 4.4)],
              [(6.2, 3.7), (9.6, 3.7), (10.6, 2.75), (9.3, 2.35), (7.8, 2.6)],
              [(9.6, 0.2), (9.4, 1.3), (10.4, 1.75), (11.2, 1.25), (12.9, 1.25), (13.0, 0.5)],
              [(12.9, 0.8), (13.5, 1.55), (16.0, 1.5), (16.0, 0.8)],
              [(12.2, 4.75), (13.7, 4.6), (16.2, 5.2), (16.2, 6.8), (12.2, 6.8)],
              [(14.9, 2.9), (16.2, 2.9), (16.2, 4.1)]],
        'O': [[(4.5, 3.7), (5.9, 3.7), (7.5, 2.25), (9.0, 1.25), (9.0, 0.2), (7.0, 0.2), (6.6, 1.3), (5.6, 2.15), (4.7, 2.95)],
              [(2.9, 3.35), (3.9, 3.2), (3.7, 3.8), (3.0, 4.3)],
              [(12.5, 3.2), (13.9, 3.25), (16.0, 4.7), (16.0, 5.3), (14.6, 5.0)],
              [(11.0, 2.4), (12.4, 2.2), (12.8, 3.3), (11.4, 3.5)]],
        'B': [[(1.5, 0.3), (1.6, 1.05), (2.7, 1.2), (3.35, 0.95), (3.4, 0.3)],
              [(13.7, 1.65), (13.6, 2.5), (16.0, 2.6), (16.0, 1.6)]],
    },
    -1: {  # starboard
        'G': [[(-0.5, 0.9), (-0.5, 3.3), (2.5, 3.3), (3.0, 2.8), (2.9, 1.9), (2.5, 1.35), (1.6, 1.1), (0.9, 0.7)],
              [(3.3, 0.2), (3.35, 1.3), (4.4, 1.35), (5.4, 1.1), (5.3, 0.2)],
              [(3.2, 3.05), (4.4, 3.5), (4.4, 4.4), (3.0, 4.4)],
              [(4.3, 1.5), (4.6, 2.4), (5.8, 3.0), (6.9, 3.8), (5.0, 3.9), (3.9, 2.6)],
              [(8.6, 0.2), (8.9, 1.2), (10.2, 1.3), (11.0, 2.1), (12.6, 2.4), (13.4, 1.6), (12.9, 0.6), (11.5, 0.2)],
              [(12.2, 4.4), (13.7, 4.2), (16.2, 4.6), (16.2, 6.8), (12.2, 6.8)],
              [(7.6, 3.8), (9.8, 3.8), (9.6, 3.0), (8.0, 3.1)]],
        'O': [[(5.9, 0.6), (6.3, 1.5), (7.4, 2.4), (8.8, 3.0), (10.0, 2.6), (9.2, 1.7), (8.3, 1.3), (7.9, 0.5)],
              [(2.9, 3.4), (3.8, 3.3), (3.6, 3.9), (3.0, 4.3)],
              [(12.6, 2.6), (14.2, 2.4), (16.0, 3.4), (16.0, 4.4), (13.6, 3.6)],
              [(13.4, 0.9), (13.9, 1.7), (16.0, 1.9), (16.0, 1.1)]],
        'B': [[(1.5, 0.3), (1.6, 1.05), (2.7, 1.2), (3.3, 0.9), (3.4, 0.3)],
              [(6.4, 0.2), (6.5, 0.75), (7.6, 0.8), (7.8, 0.2)],
              [(12.6, 0.35), (12.9, 1.0), (14.2, 1.5), (14.6, 1.1)]],
    },
}


def camo_map(side):
    """Colour weights (tan, blue-green, olive, brown) over (station, girth)."""
    w, h = int((CAMO_S1 - CAMO_S0) * CAMO_PPM), int((CAMO_Q1 - CAMO_Q0) * CAMO_PPM)
    rng = np.random.default_rng(47 + side)
    fields = [np.full((h, w), 0.5, np.float32)]
    for key in 'GOB':
        im = Image.new('L', (w, h), 0)
        d = ImageDraw.Draw(im)
        for poly in CAMO[side][key]:
            d.polygon([((a - CAMO_S0) * CAMO_PPM, (CAMO_Q1 - b) * CAMO_PPM) for a, b in poly], fill=255)
        f = np.asarray(im.filter(ImageFilter.GaussianBlur(0.2 * CAMO_PPM)), np.float32) / 255
        # a slow wobble so the edges wander like sprayed paint
        nz = Image.fromarray((rng.random((h // 28 + 2, w // 28 + 2)) * 255).astype(np.uint8)).resize((w, h), Image.BICUBIC)
        fields.append(f + (np.asarray(nz, np.float32) / 255 - 0.5) * 0.22)
    f = np.stack(fields, -1)
    e = np.exp((f - f.max(-1, keepdims=True)) * 34)
    return e / e.sum(-1, keepdims=True)


class Paint:
    """Colours any skin texel from its place on the airframe, so the
    camouflage and the markings run across the parts without seams."""

    def __init__(self):
        self.camo = {1: camo_map(1), -1: camo_map(-1)}
        self.side = {1: side_sheet(1), -1: side_sheet(-1)}
        self.top, self.bottom, self.front = top_sheet(), bottom_sheet(), front_sheet()

    def __call__(self, x, y, s, n, plain=False):
        """x, y, s: airframe coordinates; n: outward normals (x, y, s). Returns sRGB and roughness."""
        q = y + 0.8 * (W - np.minimum(np.abs(x), W))
        px, py = (s - CAMO_S0) * CAMO_PPM, (CAMO_Q1 - q) * CAMO_PPM
        wl = smooth(-0.25, 0.25, x)[..., None]
        wts = bilinear(self.camo[1], px, py) * wl + bilinear(self.camo[-1], px, py) * (1 - wl)
        col = wts @ np.array([TAN, GREEN, OLIVE, BROWN], np.float32)
        # pale underside, wrapping up to the middle of the radome at the nose
        line = fus_bot(s) + np.interp(s, [0, 0.5, 1.4, 2.2], [0.5, 0.52, 0.42, 0.36]) + 0.05 * np.sin(1.7 * s + 2.0 * np.sign(x))
        und = np.maximum(smooth(0.05, -0.05, y - line), smooth(-0.45, -0.7, n[..., 1]))[..., None]
        col = col * (1 - und) + np.array(UNDER, np.float32) * und
        # very light mottling: paint that has weathered a little
        col = col * (1 + 0.025 * np.sin(s * 3.1 + y * 5.3) * np.sin(s * 0.7 - x * 2.9 + 1.0))[..., None]
        # exhaust soot along the tail behind the engines, grime under the belly
        soot = smooth(13.75, 14.5, s) * np.exp(-((y - 2.85) / 0.5) ** 2) * (np.abs(x) > 0.35)
        grime = smooth(0.3, 0.8, -n[..., 1]) * smooth(2.5, 5.0, s) * (0.45 + 0.55 * np.sin(x * 7.0 + 2.0 * np.sin(s * 0.9)) ** 2)
        col = col * (1 - 0.5 * soot - 0.13 * grime)[..., None]
        rough = np.full(col.shape[:-1], 0.6, np.float32)
        if not plain:
            # frames every half metre with rows of rivets, and a few stringers
            fr = np.abs((s % 0.508) - 0.254)
            riv = (np.abs(fr - 0.236) < 0.006) & (np.sin(q * 62.0) > 0.55)
            seam = fr > 0.2505
            strg = (np.abs(((q + 0.11) % 0.62) - 0.31) > 0.3065) & (np.sin(s * 70.0) > 0.3)
            col = col * np.where(seam, 0.9, np.where(riv | strg, 0.86, 1.0))[..., None]
        layers = [
            (self.side[1], s, y, smooth(0.38, 0.52, n[..., 0])),
            (self.side[-1], s, y, smooth(0.38, 0.52, -n[..., 0])),
            (self.top, s, x, smooth(0.7, 0.85, n[..., 1]) * (s > 1.8)),
            (self.bottom, s, x, smooth(0.7, 0.85, -n[..., 1])),
            (self.front, x, y, (s < 1.8) * (n[..., 1] > -0.25) * smooth(0.52, 0.38, np.abs(n[..., 0]))),
        ]
        for sh, u, v, wgt in layers:
            rgba, gl = sh.sample(u, v)
            a = (rgba[..., 3] * wgt)[..., None]
            col = col * (1 - a) + rgba[..., :3] * 255 * a / np.maximum(rgba[..., 3:4], 1e-4) * (rgba[..., 3:4] > 0)
            rough = rough * (1 - gl * wgt) + 0.07 * gl * wgt
        # the dark intake in the leading edge of the aft pylon
        intake = (np.abs(x) < 0.17) & (y > 3.5) & (y < 4.05) & (s < 12.75) & (s > 12.0) & (n[..., 2] < -0.6)
        col = np.where(intake[..., None], np.array([14, 15, 16], np.float32), col)
        return np.clip(col, 0, 255), rough


def window(sh, s, y=WIN_Y, r=0.25, bubble=False):
    sh.rrect(s - 0.37, s + 0.37, y - 0.37, y + 0.37, 0.08, outline=(60, 62, 52, 150), width=0.014)
    sh.disc(s, y, r + 0.035, fill=(70, 74, 66, 255))
    sh.disc(s, y, r, fill=(12, 14, 16, 255) if bubble else (64, 30, 32, 255), gloss=True)
    if not bubble:
        # the red cabin lining shows through, darker towards the rim
        sh.disc(s, y, r, outline=(22, 22, 24, 255), width=0.07)


def corners(sh, s, y, half, col=YELLOW + (255,), arm=0.12, width=0.028):
    for a in (-1, 1):
        for b in (-1, 1):
            cs, cy = s + a * half, y + b * half
            sh.line([(cs - a * arm, cy), (cs, cy), (cs, cy - b * arm)], col, width)


def stencil(sh, s, y, lines=2, width=0.3, col=(40, 40, 40, 200)):
    """A block of lettering too small to read."""
    for k in range(lines):
        sh.line([(s, y - k * 0.045), (s + width * (1 - 0.25 * (k % 2)), y - k * 0.045)], col, 0.018)


def side_sheet(side):
    """Everything painted on one side, as seen from outside (port: nose to the left)."""
    sh = Sheet(-0.2, 15.8, -0.2, 5.4, 190, flip=side < 0)
    glass, frame_c = GLASS + (255,), (50, 84, 70, 255)
    # cockpit: chin window, lower door window, upper sliding window with its quarter light
    chin = [(0.6, 0.66), (0.66, 1.18), (0.86, 1.4), (1.24, 1.46), (1.28, 0.62), (0.9, 0.4)]
    low = [(1.4, 0.66), (1.4, 1.44), (2.04, 1.44), (2.04, 0.7)]
    quarter = [(1.02, 1.7), (1.38, 2.22), (1.38, 1.7)]
    upper = [(1.47, 1.7), (1.47, 2.26), (2.12, 2.28), (2.12, 1.7)]
    for poly in (chin, low, quarter, upper):
        sh.poly(poly, glass, gloss=True, outline=frame_c, width=0.035)
    # what shows through the glass: the dark cockpit floor and console low
    # down, a dim red seat, the pilot's shoulders and helmet higher up
    dim, seat = (13, 18, 20, 255), (72, 27, 29, 255)
    sh.poly([(1.4, 0.66), (1.4, 0.98), (2.04, 1.06), (2.04, 0.7)], dim)
    sh.poly([(1.58, 0.9), (1.56, 1.38), (1.8, 1.38), (1.84, 0.92)], seat)
    sh.poly([(1.4, 0.98), (1.4, 1.2), (1.56, 1.06), (1.58, 0.99)], dim)
    sh.poly([(0.66, 0.62), (0.9, 0.42), (1.28, 0.62), (1.27, 0.86), (0.7, 0.9)], dim)
    sh.poly([(1.47, 1.7), (1.47, 1.83), (2.12, 1.86), (2.12, 1.7)], dim)
    sh.poly([(1.72, 1.7), (1.7, 2.02), (1.96, 2.02), (1.98, 1.7)], seat)
    sh.poly([(1.6, 1.7), (1.62, 1.86), (1.72, 1.94), (1.9, 1.94), (1.98, 1.84), (1.99, 1.7)], (58, 66, 50, 255))
    sh.disc(1.79, 2.03, 0.105, fill=(84, 92, 84, 255))
    sh.poly([(1.69, 1.98), (1.7, 2.06), (1.76, 2.07), (1.75, 1.97)], (24, 28, 30, 255))
    for poly in (chin, low, quarter, upper):
        sh.poly(poly, None, outline=frame_c, width=0.035)
    sh.line([(1.33, 0.5), (1.33, 2.32), (2.2, 2.35), (2.2, 0.55), (1.33, 0.5)], (44, 52, 46, 170), 0.014)
    sh.text('非常口', 1.76, 1.57, 0.085, FONT_JA, YELLOW)
    # nose number on the brown patch, with the emergency handle marker below it
    sh.text('487', 2.78, 0.86, 0.5, FONT_EN, INK, squeeze=0.92)
    sh.poly([(2.5, 0.5), (2.72, 0.46), (2.72, 0.54)], YELLOW + (255,))
    sh.text('非常口ハンドル', 2.62, 0.4, 0.04, FONT_JA, INK)
    sh.text('ボタンを押して', 2.62, 0.35, 0.04, FONT_JA, INK)
    sh.text('ハンドルを回せ', 2.62, 0.3, 0.04, FONT_JA, INK)
    sh.text('航空自衛隊', 3.45, 2.2, 0.17, FONT_JA, INK)
    # white ring emblem behind the radome
    sh.disc(0.47, 0.95, 0.075, outline=(236, 236, 230, 255), width=0.014)
    sh.disc(0.47, 0.975, 0.014, fill=(236, 236, 230, 255))
    sh.disc(0.47, 0.925, 0.014, fill=(236, 236, 230, 255))
    # cabin
    stations = WINDOWS if side > 0 else WINDOWS[1:]
    for k, ws in enumerate(stations):
        window(sh, ws, bubble=(ws == WINDOWS[-1]))
    corners(sh, WINDOWS[1], WIN_Y, 0.45)
    corners(sh, WINDOWS[-1], WIN_Y, 0.45)
    if side > 0:
        # escape hatch around the first window
        sh.rrect(3.47, 4.53, 0.98, 2.1, 0.1, outline=(226, 218, 190, 230), width=0.03)
        sh.rrect(3.47, 4.53, 0.98, 2.1, 0.1, outline=(60, 62, 52, 160), width=0.012)
    else:
        # crew door: the upper half opens inwards, the lower half folds down as steps
        sh.rrect(2.98, 3.86, 0.26, 1.96, 0.08, outline=(46, 50, 44, 235), width=0.028)
        sh.line([(2.98, 1.13), (3.86, 1.13)], (46, 50, 44, 235), 0.024)
        sh.rrect(3.2, 3.64, 1.34, 1.78, 0.07, fill=(16, 22, 24, 255), outline=(60, 64, 56, 255), width=0.03, gloss=True)
        sh.line([(2.9, 1.2), (2.9, 1.9)], (232, 232, 226, 255), 0.03)
        # orange lens behind the cockpit, red filler rings
        sh.disc(2.68, 1.98, 0.14, fill=(226, 120, 40, 255), outline=(90, 60, 30, 255), width=0.012)
        sh.disc(4.75, 0.78, 0.1, outline=RED + (255,), width=0.022)
        stencil(sh, 4.62, 0.62, 2, 0.26, RED + (220,))
        sh.disc(7.3, 1.75, 0.06, outline=RED + (255,), width=0.016)
    r = ROUNDEL
    sh.disc(r[0], r[1], r[3], fill=(244, 244, 240, 255))
    sh.disc(r[0], r[1], r[2], fill=RED + (255,))
    # access panels with yellow corner marks, tiny stencils
    corners(sh, 10.75, 1.86, 0.2, arm=0.07, width=0.02)
    for (a, b, n_) in ((2.95, 2.02, 2), (4.8, 2.08, 1), (5.3, 0.95, 2), (8.6, 2.1, 1), (9.9, 0.9, 2), (11.0, 2.2, 2), (3.1, 0.2, 1), (12.9, 3.0, 2), (2.3, 2.9, 2), (2.05, 3.2, 1)):
        stencil(sh, a, b, n_)
    for (a, b) in ((5.1, 1.3), (7.6, 0.75), (10.6, 0.7)):
        sh.rrect(a, a + 0.5, b - 0.42, b, 0.03, outline=(60, 62, 52, 130), width=0.012)
    # rear ramp: hinge line, side edge, the dark panel above it
    sh.line([(12.35, 0.1), (12.5, 1.42), (15.6, 1.58)], (50, 52, 46, 220), 0.022)
    sh.poly([(13.95, 2.14), (13.85, 2.5), (15.7, 2.54), (15.7, 2.1)], (62, 70, 80, 255))
    # aft pylon: unit badge with the serial behind it
    bs, by = 13.42, 4.02
    sh.poly([(bs - 0.21, by + 0.25), (bs + 0.21, by + 0.25), (bs + 0.21, by - 0.05), (bs, by - 0.27), (bs - 0.21, by - 0.05)], (136, 190, 220, 255), outline=(240, 240, 236, 255), width=0.022)
    # a white bird with spread wings, as far as the badge can be made out
    sh.poly([(bs - 0.16, by + 0.15), (bs - 0.05, by + 0.09), (bs, by + 0.17), (bs + 0.05, by + 0.09), (bs + 0.16, by + 0.15), (bs + 0.06, by - 0.02), (bs, by - 0.14), (bs - 0.06, by - 0.02)], (244, 244, 240, 255))
    sh.line([(bs - 0.12, by - 0.33), (bs + 0.12, by - 0.33)], (240, 240, 236, 220), 0.02)
    sh.text('27-4487', 14.5, 4.05, 0.2, FONT_EN, INK, squeeze=0.9)
    return sh.bake()


def top_sheet():
    sh = Sheet(-0.2, 15.8, -2.0, 2.0, 120)
    walk = (30, 30, 30, 255)
    for sd in (1, -1):
        # non-slip walkways behind the forward pylon, heat shields under the exhausts
        sh.rrect(4.0, 5.3, sd * 0.42, sd * 1.0, 0.04, fill=walk)
        sh.rrect(10.2, 11.5, sd * 0.42, sd * 0.95, 0.04, fill=walk)
        sh.poly([(13.5, sd * 0.55), (15.6, sd * 0.5), (15.6, sd * 1.2), (13.7, sd * 1.25)], (34, 34, 36, 255))
        sh.line([(2.4, sd * 0.62), (12.0, sd * 0.62)], (60, 62, 52, 120), 0.016)
    return sh.bake()


def bottom_sheet():
    sh = Sheet(-0.2, 15.8, -2.0, 2.0, 120)
    line = (96, 92, 78, 150)
    for xx in (-0.62, 0.0, 0.62):
        sh.line([(1.6, xx), (12.2, xx)], line, 0.016)
    # cargo hook well in the middle, hook fittings ahead and behind, a roundel
    # (painted to read as a recess: lit far wall, shaded near one, the hook beam across)
    sh.rrect(7.2, 8.3, -0.44, 0.44, 0.06, fill=(150, 144, 124, 255), outline=(84, 80, 68, 255), width=0.03)
    sh.rrect(7.24, 8.26, -0.4, 0.4, 0.05, fill=(12, 12, 12, 255))
    sh.rrect(7.3, 8.26, -0.34, 0.4, 0.04, fill=(34, 33, 30, 255))
    sh.line([(7.75, -0.4), (7.75, 0.4)], (92, 92, 88, 255), 0.07)
    sh.disc(7.75, 0.0, 0.09, fill=(58, 58, 56, 255), outline=(110, 110, 104, 255), width=0.02)
    sh.rrect(5.35, 5.85, -0.2, 0.2, 0.18, fill=(40, 38, 36, 255))
    sh.rrect(4.5, 4.75, -0.1, 0.1, 0.04, fill=(52, 50, 46, 255))
    sh.rrect(9.5, 9.75, -0.1, 0.1, 0.04, fill=(52, 50, 46, 255))
    sh.disc(6.5, 0.0, ROUNDEL[3], fill=(244, 244, 240, 255))
    sh.disc(6.5, 0.0, ROUNDEL[2], fill=RED + (255,))
    # wheel wells (game addition: the units slide up into them)
    for rear in (False, True):
        gs, gx, _ = gear_mount(rear)
        for sd in (1, -1):
            sh.rrect(gs - 0.42, gs + 0.42, sd * gx - 0.36, sd * gx + 0.36, 0.12, fill=(22, 22, 21, 255), outline=(70, 68, 60, 255), width=0.02)
    sh.line([(12.35, -1.2), (12.35, 1.2)], (60, 58, 50, 220), 0.024)
    for a in (13.2, 14.1, 15.0):
        sh.line([(a, -1.0), (a, 1.0)], line, 0.014)
    return sh.bake()


def front_sheet():
    """Windscreen seen from ahead: (x, height)."""
    sh = Sheet(-1.6, 1.6, 0.0, 3.0, 220)
    glass, frame_c = GLASS + (255,), (50, 84, 70, 255)
    mid = [(-0.2, 1.7), (-0.19, 2.28), (0.19, 2.28), (0.2, 1.7)]
    sh.poly(mid, glass, gloss=True)
    sh.poly([(-0.2, 1.7), (-0.2, 1.86), (0.2, 1.86), (0.2, 1.7)], (12, 14, 15, 255))
    sh.poly(mid, None, outline=frame_c, width=0.04)
    for sd in (1, -1):
        pane = [(sd * 0.29, 1.68), (sd * 0.28, 2.29), (sd * 0.86, 2.27), (sd * 1.06, 2.0), (sd * 1.05, 1.68)]
        sh.poly(pane, glass, gloss=True)
        # instrument coaming, seat back, the pilot's shoulders and helmet
        sh.poly([(sd * 0.29, 1.68), (sd * 0.29, 1.84), (sd * 1.05, 1.8), (sd * 1.05, 1.68)], (12, 14, 15, 255))
        sh.poly([(sd * 0.46, 1.8), (sd * 0.47, 2.1), (sd * 0.83, 2.1), (sd * 0.84, 1.8)], (72, 27, 29, 255))
        sh.poly([(sd * 0.42, 1.8), (sd * 0.46, 1.95), (sd * 0.56, 2.0), (sd * 0.74, 2.0), (sd * 0.84, 1.95), (sd * 0.88, 1.8)], (58, 66, 50, 255))
        sh.disc(sd * 0.65, 2.07, 0.105, fill=(84, 92, 84, 255))
        sh.rrect(sd * 0.65 - 0.07, sd * 0.65 + 0.07, 2.02, 2.08, 0.02, fill=(24, 28, 30, 255))
        sh.poly(pane, None, outline=frame_c, width=0.04)
        sh.poly([(sd * 0.3, 2.37), (sd * 0.32, 2.485), (sd * 0.62, 2.47), (sd * 0.82, 2.37)], (16, 30, 26, 255), gloss=True, outline=frame_c, width=0.035)
        # small domes below the outer windscreen corners
        sh.disc(sd * 0.98, 1.57, 0.05, fill=(58, 96, 80, 255), outline=(40, 60, 52, 255), width=0.012)
    return sh.bake()


def skin_texture(paint, pos, n_out, width, height, plain=False):
    """Texture for a lofted part. pos(u, v) -> airframe x, y, s for texture
    coordinates in 0..1; n_out(x, y, s) -> a point inside, to orient normals."""
    u, v = (np.arange(width) + 0.5) / width, (np.arange(height) + 0.5) / height
    U, V = np.meshgrid(u, v)
    x, y, s = pos(U, V)
    p = np.stack([x, y, s], -1)
    eps = 1e-3
    du = np.stack(pos(U + eps, V), -1) - np.stack(pos(U - eps, V), -1)
    dv = np.stack(pos(U, V + eps), -1) - np.stack(pos(U, V - eps), -1)
    n = np.cross(du, dv)
    n /= np.maximum(np.linalg.norm(n, axis=-1, keepdims=True), 1e-9)
    n = np.where(((p - n_out(x, y, s)) * n).sum(-1, keepdims=True) < 0, -n, n)
    return paint(x, y, s, n, plain)


def to_img(a):
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))


def radome_texture(N=256):
    """Gloss black with five white lightning strips, seen from ahead."""
    im = Image.new('RGB', (N * 2, N * 2), (14, 15, 17))
    d = ImageDraw.Draw(im)
    c = N
    white = (232, 232, 228)
    d.line([(c, 0.5 * N), (c, 1.5 * N)], fill=white, width=7)
    for sx in (-1, 1):
        for sy in (-1, 1):
            d.line([(c + sx * 0.3 * N, c + sy * 0.18 * N), (c + sx * 0.62 * N, c + sy * 0.42 * N)], fill=white, width=7)
    return im.resize((N, N), Image.LANCZOS)


def screen_texture(N=256):
    """Engine intake debris screen: olive wire mesh over a dark mouth."""
    yy, xx = np.mgrid[0:N, 0:N]
    wire = (np.abs((xx % 16) - 8) > 6.4) | (np.abs((yy % 16) - 8) > 6.4)
    img = np.where(wire[..., None], np.array([92, 100, 56]), np.array([52, 58, 36])).astype(np.float32)
    rib = np.abs((xx % 64) - 32) > 29.5
    img[rib] = (80, 88, 50)
    return to_img(img)


def blade_texture(Wd=64, Hd=1024):
    """Blade skin: u across the chord from the leading edge, v along the span."""
    img = np.zeros((Hd, Wd, 3), np.float32)
    u = ((np.arange(Wd) + 0.5) / Wd)[None, :]
    v = ((np.arange(Hd) + 0.5) / Hd)[:, None]
    img[:] = (29, 30, 32)
    U, V = np.broadcast_arrays(u, v)
    img[(U < 0.14) & (V > 0.12)] = (52, 54, 58)       # erosion strip
    img[V < 0.12] = (24, 24, 26)                      # root cuff
    img[(np.abs(V - 0.55) < 0.004) & (U > 0.3) & (U < 0.6)] = (190, 160, 40)
    img *= (1 + 0.05 * np.sin(V * 180.0))[..., None]
    return to_img(img)


# ------------------------------------------------------------------ build
def build(paint):
    names = ('fuselage', 'pods', 'pylons', 'radome', 'screen', 'olive', 'green', 'black', 'walk', 'dark', 'hot', 'chrome', 'strut', 'glass', 'steel', 'red_l', 'green_l', 'white_l')
    M = {k: Mesh() for k in names}
    tex = {}

    # fuselage: dense rings through the nose, the ramp and both closed ends
    st = np.unique(np.concatenate([[0.26, 0.28, 0.3], np.linspace(0.3, 2.2, 26), np.linspace(2.2, 11.2, 30), np.linspace(11.2, L, 16), [L + 0.06, L + 0.1, LT]]))
    tt = np.linspace(0, 1, 89)
    S, T = np.meshgrid(st, tt, indexing='ij')
    x, y = fus_point(S, T)
    M['fuselage'].add(*grid(P(x, y, S), np.stack([S / LT, T], -1)))

    def fpos(u, v):
        s = u * LT
        xx, yy = fus_point(s, v)
        return xx, yy, s + 0 * xx
    col, rough = skin_texture(paint, fpos, lambda x_, y_, s_: np.stack([0 * x_, (fus_top(s_) + fus_bot(s_)) / 2, s_], -1), 4096, 1024)
    tex['fuselage'] = jpg(to_img(col), 90)
    orm = np.stack([np.full_like(rough, 255), rough * 255, np.zeros_like(rough)], -1)
    tex['fuselage_orm'] = jpg(to_img(orm).resize((2048, 512), Image.LANCZOS), 88)

    # radome: a wide rounded snout, low on the nose
    d = np.concatenate([[0.0], np.linspace(0.04, 1, 12)])[:, None]
    k = np.sqrt(1 - (1 - d) ** 2) + 1e-4
    rx, ry = section(0.6 * k, 0.87 + 0.37 * k, 0.87 - 0.37 * k, 0.3 * k, 0.3 * k, np.linspace(0, 1, 41)[None, :])
    rs = 0.0 + 0.44 * d + 0 * rx
    M['radome'].add(*grid(P(rx, ry, rs), np.stack([0.5 + rx / 1.3, 0.5 - (ry - 0.87) / 1.3 * 1.6], -1)))
    tex['radome'] = jpg(radome_texture(), 90)

    # side pods
    pt = np.linspace(0, 1, 33)
    for k_, side in enumerate((1, -1)):
        s0 = float(pod_s0(side))
        ps = np.unique(np.concatenate([s0 + 1.15 * (1 - np.cos(np.linspace(0, math.pi / 2, 9))), np.linspace(s0 + 1.15, POD_TAPER, 14), np.linspace(POD_TAPER, POD_S1, 9)]))
        S, T = np.meshgrid(ps, pt, indexing='ij')
        x, y = pod_point(side, S, T)
        M['pods'].add(*grid(P(x, y, S), np.stack([(S - s0) / (POD_S1 - s0), (T + k_) / 2], -1)))

    def ppos(u, v):
        side = np.where(v < 0.5, 1.0, -1.0)
        s = pod_s0(side) + np.clip(u, 0.0005, 0.9995) * (POD_S1 - pod_s0(side))
        xx, yy = pod_point(side, s, (v * 2) % 1.0)
        return xx, yy, s + 0 * xx
    col, _ = skin_texture(paint, ppos, lambda x_, y_, s_: np.stack([np.sign(x_) * POD_XC, 0.74 + 0 * y_, s_], -1), 2048, 512)
    tex['pods'] = jpg(to_img(col), 90)

    # rotor pylons, sharing one texture (forward one in the lower third)
    vv = np.concatenate([np.linspace(0, 1, 13), [1.03, 1.05]])
    # extra columns where the section turns from the flank to the ends: the
    # flanks carry lettering, which coarse columns there would shear
    ph = np.unique(np.concatenate([np.linspace(0, 2 * math.pi, 49)] + [c0 + sg * np.array([0.004, 0.012, 0.03, 0.06, 0.095]) for c0 in (math.pi / 2, 3 * math.pi / 2) for sg in (-1, 1)]))
    Vg, Pg = np.meshgrid(vv, ph, indexing='ij')
    for rows, n_, hub, v0, v1 in ((PYL_F, 2.4, HUB_F, 0.0, 0.34), (PYL_R, 3.2, HUB_R, 0.36, 1.0)):
        x, y, s = pylon_point(rows, n_, hub, Vg, Pg)
        M['pylons'].add(*grid(P(x, y, s), np.stack([Pg / (2 * math.pi), v0 + np.clip(Vg, 0, 1) * (v1 - v0)], -1)))

    def ypos(u, v):
        front = v < 0.35
        a = pylon_point(PYL_F, 2.4, HUB_F, np.clip(v / 0.34, 0, 1), u * 2 * math.pi)
        b = pylon_point(PYL_R, 3.2, HUB_R, np.clip((v - 0.36) / 0.64, 0, 1), u * 2 * math.pi)
        return tuple(np.where(front, p_, q_) for p_, q_ in zip(a, b))
    col, _ = skin_texture(paint, ypos, lambda x_, y_, s_: np.stack([0 * x_, y_, np.where(s_ < 8, 2.3, 13.9)], -1), 2048, 1024)
    tex['pylons'] = jpg(to_img(col), 90)

    def axis_of(hub):
        return np.array([0.0, math.cos(hub[2]), math.sin(hub[2])])

    # black caps under the rotor heads, and the fixed swashplate housings
    for hub in (HUB_F, HUB_R):
        c = P(0, hub[1], hub[0])
        M['black'].add(*lathe([(-0.47, 0.0), (-0.47, 0.64), (-0.4, 0.66), (-0.36, 0.58), (-0.36, 0.36), (-0.3, 0.3)], c, axis_of(hub), 28))

    # spine tunnel over the drive shaft, a walkway on top
    sp = np.linspace(3.0, 12.3, 12)
    S, T = np.meshgrid(sp, np.linspace(0.12, 0.88, 21), indexing='ij')
    x, y = section(0.27, H + 0.2, H - 0.3, 0.1, 0.05, T)
    M['walk'].add(*grid(P(x, y, S)))

    # engines on the shoulders beside the aft pylon
    tex['screen'] = jpg(screen_texture(), 88)
    for side in (1, -1):
        ex, ey = 1.36 * side, 2.8
        c = P(ex, ey, 11.75)
        back = (0, 0, -1)
        suv = lambda prof, th: np.stack([th[None, :] / (2 * math.pi) * 6 + 0 * prof[:, :1], prof[:, :1] * 3 + 0 * th[None, :]], -1)
        M['screen'].add(*lathe([(0.0, 0.03), (0.03, 0.16), (0.1, 0.28), (0.22, 0.38), (0.4, 0.44), (0.56, 0.45), (0.68, 0.42)], c, back, 24, suv))
        M['olive'].add(*lathe([(0.68, 0.42), (0.74, 0.385), (1.3, 0.385), (1.75, 0.37), (1.9, 0.33)], c, back, 24))
        M['hot'].add(*lathe([(1.86, 0.335), (1.9, 0.32), (2.26, 0.31), (2.28, 0.295)], c, back, 24))
        M['dark'].add(*lathe([(2.28, 0.295), (1.95, 0.27), (1.9, 0.0005)], c, back, 24))
        # mount, and the drive fairing running in and forward into the pylon
        M['olive'].add(*box(P(ex * 0.83, ey - 0.3, 12.9), (0.2, 0.2, 0.55)), flat=True)
        M['olive'].add(*tube(P(ex * 0.8, ey, 12.55), P(0.3 * side, ey + 0.12, 12.75), 0.2, 0.17, 12))
        M['olive'].add(*tube(P(ex, ey + 0.1, 13.3), P(0.3 * side, ey + 0.45, 13.5), 0.07, 0.07, 8))

    # rescue hoist over the crew door (starboard), on a tube frame
    hc = P(-1.63, 2.72, 2.85)
    M['green'].add(*lathe([(0.0, 0.07), (0.02, 0.2), (0.1, 0.3), (0.2, 0.33), (0.7, 0.33), (0.8, 0.3), (0.88, 0.2), (0.9, 0.05)], hc, (0, 0, -1), 18))
    M['chrome'].add(*lathe([(-0.012, 0.0005), (-0.012, 0.08), (0.0, 0.08)], hc, (0, 0, -1), 12))
    for (a, b) in (((-1.5, 2.55, 3.0), (-1.05, 2.42, 2.9)), ((-1.5, 2.55, 3.6), (-1.05, 2.42, 3.75)), ((-1.45, 2.9, 3.3), (-0.8, 2.62, 3.3)), ((-1.5, 2.5, 3.3), (-1.22, 1.95, 3.0))):
        M['green'].add(*tube(P(*a), P(*b), 0.028, seg=6))
    M['steel'].add(*tube(P(-1.63, 2.42, 3.2), P(-1.63, 2.26, 3.2), 0.012, seg=6))
    M['chrome'].add(*ellipsoid(P(-1.63, 2.22, 3.2), (0.04, 0.06, 0.04), 8, 6))

    # air-data probes on the nose flanks; the comb of rods on the port side
    for side in (1, -1):
        M['green'].add(*box(P(0.98 * side, 1.3, 0.72), (0.06, 0.035, 0.1)), flat=True)
        M['steel'].add(*tube(P(1.03 * side, 1.3, 0.78), P(1.03 * side, 1.3, 0.1), 0.022, 0.014, 8))
        M['steel'].add(*tube(P(1.03 * side, 1.3, 0.1), P(1.03 * side, 1.3, -0.02), 0.009, 0.007, 6))
        # wiper arms stand a little off the glass
        M['black'].add(*tube(P(0.37 * side, 2.3, 1.25), P(0.38 * side, 1.76, 0.8), 0.012, seg=5))
    for k_ in range(4):
        yk = 1.55 + 0.19 * k_
        M['steel'].add(*tube(P(1.1 + 0.03 * k_, yk, 1.0 + 0.1 * k_), P(1.46, yk, 1.0 + 0.1 * k_), 0.009, seg=5))
    # antenna stand-offs along the port shoulder, the wire between their tips
    tips = []
    for ws in WINDOWS:
        a, b = P(1.2, 2.14, ws - 0.2), P(1.4, 2.14, ws - 0.2)
        M['black'].add(*tube(a, b, 0.012, seg=5))
        M['black'].add(*tube(b, P(1.4, 2.14, ws - 0.58), 0.01, seg=5))
        tips.append(P(1.4, 2.14, ws - 0.58))
    for a, b in zip(tips, tips[1:]):
        M['black'].add(*tube(a, b, 0.004, seg=4))
    # blade antennas under the nose and the belly, a whip and a blade on the roof
    for (s_, y_, hh, ln) in ((1.4, 0.02, -0.3, 0.16), (3.1, 0.0, -0.22, 0.14), (9.0, 0.0, -0.2, 0.14), (5.6, H + 0.2, 0.26, 0.14)):
        M['black'].add(*box(P(0.0 if hh < 0 else 0.0, y_ + hh / 2, s_ + abs(hh) * 0.25), (0.012, abs(hh) / 2, ln), [(1, 0, 0), (0, 0.94, 0.34 * np.sign(hh)), (0, -0.34 * np.sign(hh), 0.94)]), flat=True)
    M['black'].add(*tube(P(0.45, H - 0.02, 2.75), P(0.45, H + 0.55, 2.95), 0.008, seg=5))
    # lights: two landing lights under the nose, pod sides, beacon on the aft pylon
    for side in (1, -1):
        M['dark'].add(*lathe([(0.0, 0.0005), (0.0, 0.085), (0.08, 0.085)], P(0.17 * side, 0.2, 0.78), (0, -0.5, -0.87), 12))
        M['white_l'].add(*lathe([(0.0, 0.0005), (0.01, 0.05), (0.0, 0.075)], P(0.17 * side, 0.2, 0.78), (0, -0.5, 0.87), 12))
        M['red_l' if side > 0 else 'green_l'].add(*ellipsoid(P(1.875 * side, 0.82, 4.75), (0.035, 0.045, 0.09), 8, 6))
    M['red_l'].add(*ellipsoid(P(0, 4.76, 14.95), (0.07, 0.09, 0.07), 8, 6))
    M['red_l'].add(*ellipsoid(P(0, -0.03, 10.4), (0.07, 0.06, 0.07), 8, 6))
    # the aft-most windows are bubbles
    for side in (1, -1):
        M['glass'].add(*ellipsoid(P(W * side - 0.02 * side, WIN_Y, WINDOWS[-1]), (0.13, 0.25, 0.25), 14, 8))
    # grab rail at the crew door, steps on the aft fuselage
    M['chrome'].add(*tube(P(-W - 0.05, 1.2, 2.9), P(-W - 0.05, 1.9, 2.9), 0.014, seg=6))
    # landing gear: the fixed outer cylinders (the sliding parts are their own models)
    for rear, drop in ((False, 0.12), (True, 0.05)):
        gs, gx, lo = gear_mount(rear)
        for side in (1, -1):
            M['strut'].add(*tube(P(gx * side, lo + 0.25, gs), P(gx * side, lo - drop, gs), 0.085, seg=12))
            M['strut'].add(*lathe([(0.0, 0.085), (0.0, 0.1), (0.05, 0.1), (0.05, 0.085)], P(gx * side, lo - drop + 0.05, gs), (0, -1, 0), 12))
    # rocket pods on stub pylons (not on the real aircraft: the game arms it)
    ls, lx, ly = LAUNCH
    for side in (1, -1):
        M['olive'].add(*box(P((lx - 0.2) * side, ly, ls + 0.8), (0.14, 0.05, 0.3)), flat=True)
        for a in (-1, 1):
            for b in (-1, 1):
                c = P(lx * side + a * 0.105, ly + b * 0.105, ls)
                M['olive'].add(*lathe([(0.0, 0.075), (0.0, 0.1), (1.5, 0.1), (1.6, 0.06), (1.6, 0.0005)], c, (0, 0, -1), 10))
                M['dark'].add(*lathe([(0.0, 0.075), (0.25, 0.07), (0.25, 0.0005)], c, (0, 0, -1), 10))
        M['olive'].add(*box(P(lx * side, ly, ls + 0.45), (0.215, 0.215, 0.05)), flat=True)
        M['olive'].add(*box(P(lx * side, ly, ls + 1.2), (0.215, 0.215, 0.05)), flat=True)
    return {k: m.done() for k, m in M.items() if m.n}, tex


def build_rotor():
    """One rotor head with its three blades, about +Y, turning anticlockwise
    seen from above (the aft rotor is the same model turned the other way)."""
    names = ('hub', 'hubmetal', 'rod', 'grey', 'yellow', 'blade', 'disc')
    M = {k: Mesh() for k in names}
    up = np.array([0.0, 1.0, 0.0])
    o = np.zeros(3)
    M['hub'].add(*lathe([(-0.5, 0.15), (-0.2, 0.15), (-0.2, 0.11), (-0.12, 0.11), (-0.12, 0.24), (0.1, 0.24), (0.16, 0.2), (0.2, 0.1), (0.2, 0.0005)], o, up, 18))
    M['grey'].add(*lathe([(-0.36, 0.0005), (-0.36, 0.5), (-0.33, 0.5), (-0.3, 0.3)], o, up, 24))
    M['hub'].add(*lathe([(-0.3, 0.2), (-0.3, 0.36), (-0.23, 0.36), (-0.23, 0.2)], o, up, 20))
    prof = airfoil(9, 1.0)
    for k in range(3):
        a = k * 2 * math.pi / 3
        er = np.array([math.cos(a), 0.0, -math.sin(a)])
        et = np.cross(up, er)          # the way the blade travels
        ax = [er, up, et]
        M['hub'].add(*box(er * 0.36, (0.16, 0.085, 0.11), ax), flat=True)
        M['hubmetal'].add(*box(er * 0.6, (0.1, 0.12, 0.14), ax), flat=True)
        M['hubmetal'].add(*tube(er * 0.6 - up * 0.17, er * 0.6 + up * 0.17, 0.05, seg=8))
        M['hub'].add(*tube(er * 0.68, er * 1.25, 0.095, 0.085, 10))
        M['yellow'].add(*lathe([(0.0, 0.1), (0.05, 0.1)], er * 0.96, er, 10))
        M['hubmetal'].add(*lathe([(0.0, 0.11), (0.0, 0.125), (0.1, 0.125), (0.1, 0.1)], er * 1.16, er, 10))
        # lag damper beside the arm, pitch link down to the swashplate
        M['hubmetal'].add(*tube(er * 0.42 - et * 0.2 + up * 0.02, er * 0.74 - et * 0.17 + up * 0.02, 0.045, seg=7))
        M['rod'].add(*tube(er * 0.74 - et * 0.17 + up * 0.02, er * 1.0 - et * 0.15 + up * 0.02, 0.022, seg=7))
        M['hub'].add(*tube(er * 0.86 + et * 0.2 - up * 0.02, er * 0.3 + et * 0.2 - up * 0.26, 0.018, seg=5))
        M['hub'].add(*box(er * 0.86 + et * 0.13, (0.04, 0.03, 0.09), ax), flat=True)
        # blade: cuff, then the constant-chord blade with a square tip
        rs = np.concatenate([[1.2, 1.32, 1.5, 1.66], np.linspace(1.9, ROTOR_R, 13)])
        rows, uvs = [], []
        for r in rs:
            chord = float(np.interp(r, [1.2, 1.32, 1.66], [0.26, 0.3, 0.81]))
            thick = float(np.interp(r, [1.2, 1.32, 1.66, ROTOR_R], [0.6, 0.45, 0.12, 0.1]))
            tw = math.radians(float(np.interp(r, [1.2, ROTOR_R], [7.0, -3.0])))
            cx = (0.3 - prof[:, 0]) * chord           # + towards the leading edge
            cy = prof[:, 1] * thick * chord
            rows.append(er * r + np.outer(cx * math.cos(tw) - cy * math.sin(tw), et) + np.outer(cx * math.sin(tw) + cy * math.cos(tw), up))
            uvs.append(np.stack([prof[:, 0], np.full(len(prof), (r - 1.2) / (ROTOR_R - 1.2))], 1))
        rows = np.array(rows)
        M['blade'].add(*grid(rows, np.array(uvs)))
        tip = rows[-1]
        M['blade'].add(*grid(np.stack([tip, np.repeat(tip.mean(0)[None], len(tip), 0)]), np.zeros((2, len(tip), 2)) + (0.5, 0.99)))
    # the blur of the turning blades, drawn only at speed
    th = np.linspace(0, 2 * math.pi, 73)
    rr = np.concatenate([[0.9], np.linspace(1.6, ROTOR_R + 0.03, 9)])[:, None]
    disc = np.stack([rr * np.cos(th), 0 * rr + 0 * th, -rr * np.sin(th)], -1)
    M['disc'].add(*grid(disc, np.stack([th / (2 * math.pi) + 0 * rr, rr / ROTOR_R + 0 * th], -1), outward=(0, 1, 0)))
    return {k: m.done() for k, m in M.items() if m.n}, {'blade': jpg(blade_texture(), 88)}


def wheel(M, c, width=0.26):
    """Fat tyre and dished hub about the X axis."""
    hw = width / 2
    M['tyre'].add(*lathe([(-hw, 0.17), (-hw, 0.26), (-hw * 0.72, 0.305), (0, WHEEL_R), (hw * 0.72, 0.305), (hw, 0.26), (hw, 0.17)], c, (1, 0, 0), 22))
    M['hubcap'].add(*lathe([(-hw * 0.45, 0.0005), (-hw * 0.45, 0.08), (-hw * 0.85, 0.13), (-hw * 0.85, 0.17), (hw * 0.85, 0.17), (hw * 0.85, 0.13), (hw * 0.45, 0.08), (hw * 0.45, 0.0005)], c, (1, 0, 0), 16))
    for sx in (-1, 1):
        M['strut'].add(*lathe([(0.0, 0.0005), (0.0, 0.05), (0.03, 0.035)], np.asarray(c, float) + (sx * hw * 0.45, 0, 0), (sx, 0, 0), 10))


def build_gear(rear):
    """The part of one gear unit that slides: piston, torque link, axle beam
    or fork, wheels. Origin on the strut axis at the pod underside, strut
    compressed (the wheels then touch the ground)."""
    M = {k: Mesh() for k in ('chrome', 'strut', 'tyre', 'hubcap')}
    _, _, lo = gear_mount(rear)
    axle_y = -(lo + Y0 - WHEEL_R)
    if not rear:
        M['chrome'].add(*tube((0, 0.24, 0), (0, axle_y + 0.16, 0), 0.055, seg=12))
        # axle housing with the beam between the twin wheels
        M['strut'].add(*tube((0, axle_y + 0.2, 0), (0, axle_y - 0.07, 0), 0.075, seg=12))
        M['strut'].add(*tube((-0.2, axle_y, 0), (0.2, axle_y, 0), 0.06, seg=10))
        # lower arm of the torque link behind the strut, the drag brace ahead
        M['strut'].add(*box((0, axle_y + 0.2, -0.13), (0.035, 0.12, 0.018), [(1, 0, 0), (0, 0.74, 0.67), (0, -0.67, 0.74)]), flat=True)
        M['strut'].add(*tube((-0.05, axle_y + 0.29, -0.2), (0.05, axle_y + 0.29, -0.2), 0.022, seg=6))
        M['strut'].add(*tube((0, axle_y + 0.02, 0.06), (0, axle_y + 0.2, 0.2), 0.024, seg=6))
        for sx in (-1, 1):
            wheel(M, (sx * 0.17, axle_y, 0))
    else:
        top = axle_y + WHEEL_R + 0.04
        M['chrome'].add(*tube((0, 0.24, 0), (0, top + 0.1, 0), 0.055, seg=12))
        # swivel housing, then the heavy fork with the wheel trailing behind
        M['strut'].add(*lathe([(0.0, 0.0005), (0.0, 0.1), (0.14, 0.1), (0.17, 0.07)], (0, top, 0), (0, 1, 0), 12))
        M['strut'].add(*box((0, top + 0.01, -0.04), (0.2, 0.045, 0.12)), flat=True)
        for sx in (-1, 1):
            M['strut'].add(*box((sx * 0.175, (top + axle_y) / 2, -0.07), (0.028, (top - axle_y) / 2 + 0.04, 0.06), [(1, 0, 0), (0, 0.97, 0.24), (0, -0.24, 0.97)]), flat=True)
        M['strut'].add(*tube((-0.2, axle_y, -0.11), (0.2, axle_y, -0.11), 0.04, seg=8))
        M['strut'].add(*box((0, top + 0.2, 0.1), (0.03, 0.1, 0.018), [(1, 0, 0), (0, 0.8, -0.6), (0, 0.6, 0.8)]), flat=True)
        wheel(M, (0, axle_y, -0.11))
    return {k: m.done() for k, m in M.items() if m.n}, {}


def jpg(im, q=90):
    b = io.BytesIO()
    im.convert('RGB').save(b, 'JPEG', quality=q, optimize=True, subsampling=0)
    return b.getvalue()


MATS = {
    'fuselage': dict(rough=1.0, metal=0.0),
    'pods': dict(rough=0.6, metal=0.0),
    'pylons': dict(rough=0.6, metal=0.0),
    'radome': dict(rough=0.16, metal=0.0),
    'screen': dict(rough=0.6, metal=0.2),
    'olive': dict(color=(0.1, 0.118, 0.04), rough=0.5, metal=0.0),
    'green': dict(color=(0.055, 0.15, 0.1), rough=0.55, metal=0.0),
    'black': dict(color=(0.012, 0.012, 0.013), rough=0.55, metal=0.0),
    'walk': dict(color=(0.014, 0.014, 0.014), rough=0.9, metal=0.0),
    'dark': dict(color=(0.006, 0.006, 0.007), rough=0.7, metal=0.0),
    'hot': dict(color=(0.42, 0.37, 0.3), rough=0.34, metal=1.0),
    'chrome': dict(color=(0.82, 0.83, 0.85), rough=0.14, metal=1.0),
    'strut': dict(color=(0.42, 0.43, 0.41), rough=0.5, metal=0.0),
    'glass': dict(color=(0.006, 0.008, 0.01), rough=0.06, metal=0.0),
    'steel': dict(color=(0.2, 0.2, 0.21), rough=0.4, metal=0.8),
    'red_l': dict(color=(0.75, 0.03, 0.03), rough=0.25, metal=0.0, emissive=(1.0, 0.05, 0.03)),
    'green_l': dict(color=(0.03, 0.6, 0.15), rough=0.25, metal=0.0, emissive=(0.03, 1.0, 0.2)),
    'white_l': dict(color=(0.95, 0.95, 0.9), rough=0.2, metal=0.0, emissive=(1.0, 1.0, 0.9)),
    'hub': dict(color=(0.016, 0.016, 0.018), rough=0.5, metal=0.3),
    'hubmetal': dict(color=(0.045, 0.045, 0.05), rough=0.5, metal=0.3),
    'rod': dict(color=(0.6, 0.6, 0.62), rough=0.3, metal=0.9),
    'grey': dict(color=(0.02, 0.02, 0.022), rough=0.6, metal=0.0),
    'yellow': dict(color=(0.75, 0.55, 0.03), rough=0.5, metal=0.0),
    'blade': dict(rough=0.62, metal=0.0),
    'disc': dict(color=(0.02, 0.02, 0.024), alpha=0.5, rough=0.8, metal=0.0),
    'tyre': dict(color=(0.012, 0.012, 0.012), rough=0.85, metal=0.0),
    'hubcap': dict(color=(0.33, 0.34, 0.33), rough=0.5, metal=0.1),
}


def tangents(n):
    """Any unit vector square to the normal (there are no normal maps, but
    the shader builds its frame from it and must not see a degenerate one)."""
    ref = np.where(np.abs(n[:, 2:3]) < 0.9, [0.0, 0.0, 1.0], [1.0, 0.0, 0.0])
    t = ref - n * (ref * n).sum(1, keepdims=True)
    t /= np.maximum(np.linalg.norm(t, axis=1, keepdims=True), 1e-9)
    return np.concatenate([t, np.ones((len(t), 1))], 1).astype(np.float32)


def write(out, meshes, textures, name):
    bin_ = bytearray()
    views, accessors, images, texs, materials, prims = [], [], [], [], [], []

    def view(data, target=None):
        while len(bin_) % 4:
            bin_.append(0)
        v = {'buffer': 0, 'byteOffset': len(bin_), 'byteLength': len(data)}
        if target:
            v['target'] = target
        bin_.extend(data)
        views.append(v)
        return len(views) - 1

    def acc(arr, ctype, kind, target, minmax=False):
        a = {'bufferView': view(arr.tobytes(), target), 'componentType': ctype, 'count': len(arr), 'type': kind}
        if minmax:
            a['min'], a['max'] = arr.min(0).tolist(), arr.max(0).tolist()
        accessors.append(a)
        return len(accessors) - 1

    def texture(data):
        images.append({'bufferView': view(data), 'mimeType': 'image/jpeg'})
        texs.append({'source': len(images) - 1, 'sampler': 0})
        return {'index': len(texs) - 1}

    tris = 0
    for mname, (v, n, uv, idx) in meshes.items():
        m = MATS[mname]
        pbr = {'baseColorFactor': list(m.get('color', (1, 1, 1))) + [m.get('alpha', 1)], 'metallicFactor': m['metal'], 'roughnessFactor': m['rough']}
        if mname in textures:
            pbr['baseColorTexture'] = texture(textures[mname])
        if mname + '_orm' in textures:
            pbr['metallicRoughnessTexture'] = texture(textures[mname + '_orm'])
        mat = {'name': mname, 'pbrMetallicRoughness': pbr}
        if 'alpha' in m:
            mat['alphaMode'] = 'BLEND'
            mat['doubleSided'] = True
        if 'emissive' in m:
            mat['emissiveFactor'] = list(m['emissive'])
        materials.append(mat)
        prims.append({
            'attributes': {
                'POSITION': acc(v.astype(np.float32), 5126, 'VEC3', 34962, True),
                'NORMAL': acc(n.astype(np.float32), 5126, 'VEC3', 34962),
                'TANGENT': acc(tangents(n), 5126, 'VEC4', 34962),
                'TEXCOORD_0': acc(uv.astype(np.float32), 5126, 'VEC2', 34962),
            },
            'indices': acc(idx.reshape(-1).astype(np.uint32), 5125, 'SCALAR', 34963),
            'material': len(materials) - 1,
        })
        tris += len(idx)
    gltf = {
        'asset': {'version': '2.0', 'generator': 'zonefall build-ch47j.py'},
        'scene': 0, 'scenes': [{'nodes': [0]}], 'nodes': [{'mesh': 0, 'name': name}],
        'meshes': [{'primitives': prims}], 'materials': materials,
        'samplers': [{'magFilter': 9729, 'minFilter': 9987, 'wrapS': 10497, 'wrapT': 10497}],
        'accessors': accessors, 'bufferViews': views, 'buffers': [{'byteLength': len(bin_)}],
    }
    if texs:
        gltf['textures'], gltf['images'] = texs, images
    js = json.dumps(gltf, separators=(',', ':')).encode()
    js += b' ' * (-len(js) % 4)
    while len(bin_) % 4:
        bin_.append(0)
    with open(out, 'wb') as f:
        f.write(struct.pack('<III', 0x46546C67, 2, 12 + 8 + len(js) + 8 + len(bin_)))
        f.write(struct.pack('<II', len(js), 0x4E4F534A) + js)
        f.write(struct.pack('<II', len(bin_), 0x004E4942) + bytes(bin_))
    print(f'{out}: {tris} triangles, {sum(len(m[0]) for m in meshes.values())} vertices, {(12 + 16 + len(js) + len(bin_)) / 1e6:.2f} MB')


def main(out):
    write(out, *build(Paint()), 'CH-47J')
    write(out.replace('.glb', '_rotor.glb'), *build_rotor(), 'CH-47J rotor')
    write(out.replace('.glb', '_gear_front.glb'), *build_gear(False), 'CH-47J forward gear')
    write(out.replace('.glb', '_gear_rear.glb'), *build_gear(True), 'CH-47J aft gear')
    r3 = lambda v: [round(float(a), 3) for a in v]
    print('constants for src/render/heli.ts:')
    print(' rotor hubs (x, y, z, forward tilt rad):', [r3(list(P(0, h[1], h[0])) + [h[2]]) for h in (HUB_F, HUB_R)])
    print(' gear (x, y, z):', [r3(P(gx * sd, lo, gs)) for gs, gx, lo in (gear_mount(False), gear_mount(True)) for sd in (1, -1)], 'travel', GEAR_TRAVEL, 'retract', [round(lo + Y0 + GEAR_SUNK - WHEEL_R, 3) for _, _, lo in (gear_mount(False), gear_mount(True))])
    print(' blade bend: root', BLADE_ROOT, 'rest', BLADE_REST, 'cone', BLADE_CONE)
    print(' missile ports:', [r3(P(LAUNCH[1] * sd + a * 0.105, LAUNCH[2] + b * 0.105, LAUNCH[0])) for sd in (1, -1) for a in (-1, 1) for b in (-1, 1)])


if __name__ == '__main__':
    main(sys.argv[1])
