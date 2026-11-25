#!/usr/bin/env python3
"""Build the Boeing 777-300ER model (Japanese government livery) as a GLB.

Everything is generated here: the airframe is lofted from dimensioned
sections (73.9 m long, 64.8 m span, 6.2 m fuselage, GE90-size nacelles) and
the livery is painted into textures. No photographs or third-party meshes.

usage: build-b777.py out.glb
Model space: +Z nose, +Y up, +X left wing; origin on the fuselage axis at
the wing, metres.
"""
import io, json, math, struct, sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

L = 73.86            # fuselage length
R = 3.1              # fuselage radius
ORIGIN_S = 35.0      # station that becomes z = 0
FONT_JA = '/usr/share/fonts/opentype/ipafont-gothic/ipag.ttf'
FONT_EN = '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'

WHITE = (247, 248, 250)
BELLY = (214, 219, 225)
RED = (200, 16, 30)
GOLD = (196, 160, 82)
GLASS = (18, 24, 32)


# ------------------------------------------------------------------ fuselage shape
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


# The nose, from the side view: a blunt radome whose upper line runs straight
# into the windshield (no step), reaching the full section about 10 m back.
NOSE_TIP_Y = -0.62
_nose_top = pchip([(0, NOSE_TIP_Y), (0.05, -0.3), (0.25, 0.02), (0.7, 0.4), (1.4, 0.8), (2.3, 1.2), (3.0, 1.75), (3.6, 2.15), (4.5, 2.56), (5.5, 2.82), (7.0, 2.99), (8.6, 3.085), (10.2, R)])
_nose_bot = pchip([(0, NOSE_TIP_Y), (0.05, -0.96), (0.25, -1.32), (0.7, -1.74), (1.4, -2.14), (2.3, -2.5), (3.5, -2.82), (5.0, -3.01), (6.2, -3.085), (7.2, -R)])
_nose_hw = pchip([(0, 0.0), (0.05, 0.33), (0.25, 0.7), (0.7, 1.1), (1.4, 1.5), (2.3, 1.86), (3.5, 2.32), (5.0, 2.76), (6.5, 3.0), (8.2, R)])


def fus_top(s):
    s = np.asarray(s, float)
    tail = R - 0.55 * np.clip((s - 56.0) / (L - 56.0), 0, 1) ** 1.3
    return np.where(s < 10.2, _nose_top(s), tail)


def fus_bot(s):
    s = np.asarray(s, float)
    tail = -R + 4.75 * np.clip((s - 45.5) / (L - 45.5), 0, 1) ** 1.55
    return np.where(s < 7.2, _nose_bot(s), tail)


def fus_half_width(s):
    s = np.asarray(s, float)
    tail = R * (1 - np.clip((s - 49.0) / (L - 49.0), 0, 1) ** 1.45) + 0.1
    return np.minimum(np.where(s < 8.2, _nose_hw(s), R), tail)


def fus_point(s, th):
    """th = 0 at the belly, pi/2 on the left side, pi on top."""
    top, bot = fus_top(s), fus_bot(s)
    yc, ry = (top + bot) / 2, (top - bot) / 2
    return fus_half_width(s) * np.sin(th), yc - ry * np.cos(th)


def to_model(x, y, s):
    return np.stack([x, y, ORIGIN_S - s], -1)


# ------------------------------------------------------------------ mesh helpers
class Mesh:
    def __init__(self):
        self.v, self.uv, self.idx = [], [], []
        self.n = 0

    def add(self, verts, uvs, tris):
        self.v.append(np.asarray(verts, np.float32).reshape(-1, 3))
        self.uv.append(np.asarray(uvs, np.float32).reshape(-1, 2))
        self.idx.append(np.asarray(tris, np.uint32).reshape(-1, 3) + self.n)
        self.n += len(self.v[-1])

    def done(self):
        v = np.concatenate(self.v)
        uv = np.concatenate(self.uv)
        idx = np.concatenate(self.idx)
        # smooth normals, welded by position so UV seams do not show
        fn = np.cross(v[idx[:, 1]] - v[idx[:, 0]], v[idx[:, 2]] - v[idx[:, 0]])
        _, inv = np.unique(np.round(v * 2000).astype(np.int64), axis=0, return_inverse=True)
        inv = inv.reshape(-1)
        acc = np.zeros((inv.max() + 1, 3))
        for k in range(3):
            np.add.at(acc, inv[idx[:, k]], fn)
        n = acc[inv]
        ln = np.linalg.norm(n, axis=1, keepdims=True)
        n = np.where(ln > 1e-12, n / np.maximum(ln, 1e-12), [0, 1, 0])
        return v, n.astype(np.float32), uv, idx


def grid(points, uvs, outward=None, hard=False):
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
    return v, np.asarray(uvs, float).reshape(-1, 2), tris


def revolve(profile, center, axis_fwd=True, seg=48, uv=None):
    """Surface of revolution about the Z axis through `center`.
    profile: list of (s_offset, radius); s grows towards the tail."""
    prof = np.asarray(profile, float)
    th = np.linspace(0, 2 * math.pi, seg + 1)
    pts = np.zeros((len(prof), seg + 1, 3))
    pts[:, :, 0] = center[0] + prof[:, 1:2] * np.sin(th)
    pts[:, :, 1] = center[1] - prof[:, 1:2] * np.cos(th)
    pts[:, :, 2] = (center[2] - prof[:, 0:1]) + 0 * th
    if uv is None:
        uvs = np.zeros((len(prof), seg + 1, 2)) + 0.5
    else:
        uvs = uv(prof, th)
    return pts, uvs


def ellipsoid(center, radii, seg=28, rings=16):
    u = np.linspace(0, math.pi, rings + 1)[:, None]
    t = np.linspace(0, 2 * math.pi, seg + 1)[None, :]
    pts = np.stack([center[0] + radii[0] * np.sin(u) * np.sin(t), center[1] - radii[1] * np.sin(u) * np.cos(t), center[2] + radii[2] * np.cos(u) + 0 * t], -1)
    return pts, np.zeros(pts.shape[:2] + (2,)) + 0.5


# ------------------------------------------------------------------ lifting surfaces
def airfoil(n=22, thick=0.12, camber=0.012):
    """Closed loop from the trailing edge over the top to the leading edge and
    back underneath: (chord fraction, height fraction)."""
    x = (1 - np.cos(np.linspace(0, math.pi, n))) / 2
    yt = 5 * thick * (0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x ** 2 + 0.2843 * x ** 3 - 0.1036 * x ** 4)
    yc = camber * 4 * x * (1 - x)
    up = np.stack([x[::-1], (yc + yt)[::-1]], 1)
    lo = np.stack([x[1:], (yc - yt)[1:]], 1)
    return np.concatenate([up, lo]), n


def wing_le(x):
    x = np.abs(x)
    return np.where(x < 30.0, 27.4 + 0.675 * (x - 3.0), 27.4 + 0.675 * 27.0 + 1.48 * (x - 30.0))


def wing_chord(x):
    x = np.abs(x)
    return np.interp(x, [0, 3.0, 10.0, 30.0, 32.4], [13.6, 13.4, 8.3, 2.7, 0.45])


def wing_y(x):
    x = np.abs(x)
    return -1.45 + 0.105 * np.maximum(x - 3.0, 0) + 0.0016 * x ** 2


def wing_thick(x):
    return np.interp(np.abs(x), [0, 3, 10, 30, 32.4], [0.135, 0.13, 0.11, 0.09, 0.08])


def surface(stations, le, chord, yfun, thick, side, uv_top, vertical=False, n=22):
    """Loft a wing-like surface. Returns (upper grid, lower grid, tip cap)."""
    prof, npts = airfoil(n, 1.0, 0.0)
    rows = []
    for x in stations:
        c, t = float(chord(x)), float(thick(x))
        s = float(le(x)) + prof[:, 0] * c
        h = prof[:, 1] * t * c + 0.012 * 4 * prof[:, 0] * (1 - prof[:, 0]) * c * (0 if vertical else 1)
        if vertical:
            rows.append(np.stack([h * side, np.full_like(s, x), ORIGIN_S - s], 1))
        else:
            rows.append(np.stack([np.full_like(s, x * side), float(yfun(x)) + h, ORIGIN_S - s], 1))
    p = np.array(rows)
    up, lo = p[:, :npts], p[:, npts - 1:]
    return up, lo, p[-1]


# ------------------------------------------------------------------ textures
def smooth_curve(pts, s):
    pts = np.asarray(pts, float)
    y = np.interp(s, pts[:, 0], pts[:, 1])
    k = max(3, int(len(s) / (L / 3.0)) | 1)
    ker = np.ones(k) / k
    for _ in range(3):
        y = np.convolve(np.pad(y, k // 2, mode='edge'), ker, mode='valid')
    return y


BAND = [  # station, upper edge, lower edge (m above the fuselage axis)
    (7.6, 0.28, 0.28), (9.5, 0.2, 0.05), (13, -0.05, -0.5), (18, -0.45, -1.2), (25, -0.85, -2.0), (33, -1.05, -2.45),
    (42, -0.95, -2.35), (49, -0.55, -1.95), (55, 0.2, -1.2), (60, 1.35, -0.2), (64, 3.0, 1.0), (67, 4.2, 2.0), (L, 4.5, 3.1),
]
DOORS = [7.3, 19.6, 36.4, 49.3, 63.2]
WIN_Y = 0.42


def fuselage_texture(W=4096, H=1024, ss=2):
    w, h = W * ss, H * ss
    s = (np.arange(w) + 0.5) / w * L
    th = (np.arange(h) + 0.5) / h * 2 * math.pi
    top, bot = fus_top(s), fus_bot(s)
    yc, ry = (top + bot) / 2, (top - bot) / 2
    y = yc[None, :] - ry[None, :] * np.cos(th)[:, None]
    up = smooth_curve([(a, b) for a, b, _ in BAND], s)[None, :]
    lo = smooth_curve([(a, c) for a, _, c in BAND], s)[None, :]
    img = np.empty((h, w, 3), np.float32)
    img[:] = WHITE
    below = (y < lo) & (s[None, :] > 9.0)
    img[below] = BELLY
    # soft edges: a few centimetres of blend keeps the band from aliasing
    def cover(d, width=0.02):
        return np.clip(d / width + 0.5, 0, 1)
    band = cover(up - y) * cover(y - lo) * (s[None, :] > 7.6)
    gold = cover(y - up) * cover(up + 0.085 - y) * (s[None, :] > 7.9)
    for c in range(3):
        img[:, :, c] = img[:, :, c] * (1 - band) + RED[c] * band
        img[:, :, c] = img[:, :, c] * (1 - gold) + GOLD[c] * gold
    # faint skin panels: frames every ~1.1 m and a few lap joints
    lines = ((np.abs((s[None, :] % 1.105) - 0.55) > 0.54) & (s[None, :] > 6) & (s[None, :] < 66)) | (np.abs(np.abs(y) - 1.9) < 0.008)
    img[lines & (band < 0.5)] *= 0.955
    # radome joint
    img[:, np.abs(s - 1.55) < 0.012] *= 0.9
    im = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))
    d = ImageDraw.Draw(im)

    def px(sv, yv, side):
        """Texture pixel of the point at station sv, height yv on a side."""
        t, b = float(fus_top(sv)), float(fus_bot(sv))
        c = np.clip(-(yv - (t + b) / 2) / ((t - b) / 2), -1, 1)
        thv = math.acos(c)
        if side < 0:
            thv = 2 * math.pi - thv
        return sv / L * w, thv / (2 * math.pi) * h

    def rrect(s0, s1, y0, y1, side, fill=None, outline=None, width=1, rad=0.3):
        (x0, r0), (x1, r1) = px(s0, y0, side), px(s1, y1, side)
        box = [min(x0, x1), min(r0, r1), max(x0, x1), max(r0, r1)]
        d.rounded_rectangle(box, radius=rad * min(box[2] - box[0], box[3] - box[1]), fill=fill, outline=outline, width=width)

    # cabin windows: a VIP cabin, so the rows are broken into groups
    rng = np.random.default_rng(777)
    wins, sv, run = [], 10.4, 0
    while sv < 65.5:
        if run <= 0:
            run = int(rng.integers(3, 9))
            sv += float(rng.choice([0.0, 0.51, 1.02, 1.53]))
        if all(abs(sv - dd) > 1.05 for dd in DOORS):
            wins.append(sv)
        sv += 0.51
        run -= 1
    for side in (1, -1):
        for sv in wins:
            rrect(sv - 0.165, sv + 0.165, WIN_Y - 0.23, WIN_Y + 0.23, side, fill=(170, 176, 184), rad=0.45)
            rrect(sv - 0.125, sv + 0.125, WIN_Y - 0.185, WIN_Y + 0.185, side, fill=GLASS, rad=0.45)
        for dd in DOORS:
            big = dd != DOORS[2]
            dw, y0, y1 = (0.56, -0.62, 1.32) if big else (0.5, -0.5, 1.2)
            rrect(dd - dw, dd + dw, y0, y1, side, outline=(120, 126, 134), width=max(2, ss * 2), rad=0.16)
            rrect(dd - 0.1, dd + 0.1, WIN_Y - 0.1, WIN_Y + 0.16, side, fill=GLASS, rad=0.5)
            rrect(dd - 0.2, dd + 0.2, -0.3, -0.22, side, fill=(200, 30, 40), rad=0.2)
        # cargo doors (right side of a 777, but outlines both sides read fine)
        for a, b in ((12.6, 15.3), (51.5, 53.6)):
            if side < 0:
                rrect(a, b, -2.55, -0.95, side, outline=(150, 156, 164), width=ss * 2, rad=0.08)
        # static ports / probes near the nose
        for k, (a, yy) in enumerate(((4.9, -0.55), (5.3, -0.9), (5.9, -0.35))):
            (cx, cy) = px(a, yy, side)
            d.ellipse([cx - 3 * ss, cy - 3 * ss, cx + 3 * ss, cy + 3 * ss], fill=(120, 124, 130))

    # cockpit glazing, as in the side view: the sills run level about 0.7 m
    # above the axis, the windshield climbs the nose line to the centre post
    # and the last side pane is cut back at its upper rear corner.
    # Corners are (station, height); None = on the centre line.
    panes = [
        [(2.34, None), (3.46, None), (3.98, 1.74), (3.0, 0.78)],
        [(3.08, 0.76), (4.06, 1.72), (4.72, 1.72), (4.3, 0.74)],
        [(4.38, 0.74), (4.8, 1.72), (5.2, 1.6), (5.74, 1.1), (5.6, 0.74)],
    ]
    for side in (1, -1):
        for pane in panes:
            poly = []
            for (a, yy) in pane:
                if yy is None:
                    poly.append((a / L * w, (0.5 - 0.0045 * side) * h))
                else:
                    poly.append(px(a, yy, side))
            d.polygon(poly, fill=GLASS)
            d.line(poly + [poly[0]], fill=(104, 108, 116), width=ss * 3, joint='curve')

    def paste(text_img, s_mid, y_mid, height_m, side):
        """Stick an upright, outside-view image on a side of the fuselage."""
        length_m = height_m * text_img.width / text_img.height
        (x0, r0) = px(s_mid - length_m / 2, y_mid - height_m / 2, side)
        (x1, r1) = px(s_mid + length_m / 2, y_mid + height_m / 2, side)
        tw, thh = int(round(abs(x1 - x0))), int(round(abs(r1 - r0)))
        t = text_img.resize((max(tw, 1), max(thh, 1)), Image.LANCZOS)
        t = t.transpose(Image.FLIP_TOP_BOTTOM) if side > 0 else t.transpose(Image.FLIP_LEFT_RIGHT)
        im.paste(t, (int(min(x0, x1)), int(min(r0, r1))), t)

    def title(order):
        fj, fe = ImageFont.truetype(FONT_JA, 190), ImageFont.truetype(FONT_EN, 150)
        c = Image.new('RGBA', (1900, 240), (0, 0, 0, 0))
        dd = ImageDraw.Draw(c)
        x = 10
        for part in order:
            if part == 'ja':
                dd.text((x, 20), '日本国', font=fj, fill=(24, 26, 30, 255))
                x += int(dd.textlength('日本国', font=fj)) + 40
            elif part == 'en':
                dd.text((x, 44), 'JAPAN', font=fe, fill=(24, 26, 30, 255))
                x += int(dd.textlength('JAPAN', font=fe)) + 40
            else:  # the flag
                dd.rectangle([x, 48, x + 216, 192], fill=(255, 255, 255, 255), outline=(150, 150, 150, 255), width=3)
                dd.ellipse([x + 108 - 43, 120 - 43, x + 108 + 43, 120 + 43], fill=RED + (255,))
                x += 216 + 40
        return c.crop((0, 0, x, 240))

    # nose-side word first on both sides
    paste(title(['ja', 'flag', 'en']), 13.6, 1.72, 0.95, 1)
    paste(title(['en', 'flag', 'ja']), 13.6, 1.72, 0.95, -1)
    small = Image.new('RGBA', (900, 150), (0, 0, 0, 0))
    ImageDraw.Draw(small).text((6, 10), '航空自衛隊', font=ImageFont.truetype(FONT_JA, 120), fill=(30, 32, 38, 255))
    small = small.crop((0, 0, 620, 150))
    for side in (1, -1):
        paste(small, 58.4, 1.65, 0.34, side)
    return im.resize((W, H), Image.LANCZOS)


def wing_texture(W=2048, H=1024):
    ss = 2
    w, h = W * ss, H * ss
    x = ((np.arange(w) + 0.5) / w * 66.0 - 33.0)[None, :]
    s = (26.0 + (np.arange(h) + 0.5) / h * 26.0)[:, None]
    le, ch = wing_le(x), wing_chord(x)
    f = (s - le) / np.maximum(ch, 0.01)
    img = np.empty((h, w, 3), np.float32)
    img[:] = (196, 201, 207)
    ax = np.abs(x)
    img[(f < 0.11) & (ax > 3.4)] = (222, 225, 229)            # slats
    img[(f > 0.3) & (f < 0.62) & (ax < 22)] = (188, 193, 200)  # wing box, a shade darker
    def line(mask):
        img[mask] *= 0.78
    line(np.abs(f - 0.11) < 0.004)
    line((np.abs(f - 0.68) < 0.004) & (ax > 3.4) & (ax < 29))
    for edge in (4.0, 8.9, 11.2, 14.5, 18.0, 21.5, 25.0, 28.6):
        line((np.abs(ax - edge) < 0.03) & (f > 0.68))
        line((np.abs(ax - edge) < 0.03) & (f < 0.11))
    # spoiler panels ahead of the flaps
    line((np.abs(f - 0.56) < 0.003) & (ax > 4.5) & (ax < 21))
    # walkway lines near the root
    img[(np.abs(ax - 4.6) < 0.05) & (f > 0.14) & (f < 0.66)] = (40, 42, 46)
    img[(np.abs(f - 0.14) < 0.006) & (ax > 3.6) & (ax < 4.65)] = (40, 42, 46)
    im = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))
    d = ImageDraw.Draw(im)
    def pix(xv, sv):
        return ((xv + 33.0) / 66.0 * w, (sv - 26.0) / 26.0 * h)
    for xv in (24.6, -24.6):
        sv = float(wing_le(xv) + 0.5 * wing_chord(xv))
        r = 1.05
        (cx, cy) = pix(xv, sv)
        rx, ry = r / 66.0 * w, r / 26.0 * h
        d.ellipse([cx - rx, cy - ry, cx + rx, cy + ry], fill=RED)
    # JAPAN on the right wing, read from behind
    f_en = ImageFont.truetype(FONT_EN, 200)
    t = Image.new('RGBA', (1000, 240), (0, 0, 0, 0))
    td = ImageDraw.Draw(t)
    td.text((8, 10), 'JAPAN', font=f_en, fill=(34, 36, 42, 255))
    t = t.crop((0, 0, int(td.textlength('JAPAN', font=f_en)) + 16, 240))
    xa, xb = -13.2, -20.8                       # J inboard, N outboard
    height_m = (xa - xb) * t.height / t.width
    sm = float(wing_le(-17.0) + 0.42 * wing_chord(-17.0))
    (x0, y0), (x1, y1) = pix(xb, sm - height_m / 2), pix(xa, sm + height_m / 2)
    tt = t.resize((int(x1 - x0), int(y1 - y0)), Image.LANCZOS).transpose(Image.FLIP_LEFT_RIGHT)
    # the wing is swept: shear the word so it runs along the span
    shear = 0.675 * (26.0 / 66.0) * (w / h) * -1.0
    big = Image.new('RGBA', (tt.width, tt.height + int(abs(shear) * tt.width) + 2), (0, 0, 0, 0))
    big.paste(tt, (0, 0))
    big = big.transform(big.size, Image.AFFINE, (1, 0, 0, shear, 1, 0), Image.BICUBIC)
    im.paste(big, (int(x0), int(y0 - 0)), big)
    return im.resize((W, H), Image.LANCZOS)


FIN_BOX = (55.0, 73.0, 1.0, 13.0)  # s0, s1, y0, y1 of the fin texture


def fin_texture(N=1024):
    ss = 2
    n = N * ss
    im = Image.new('RGB', (n, n), WHITE)
    d = ImageDraw.Draw(im)
    s0, s1, y0, y1 = FIN_BOX
    # rudder hinge line and its split, under the roundel
    def p(sv, yv):
        return ((sv - s0) / (s1 - s0) * n, (1 - (yv - y0) / (y1 - y0)) * n)
    d.line([p(64.3, 2.7), p(69.3, 12.1)], fill=(205, 208, 213), width=ss * 2)
    d.line([p(66.2, 6.6), p(68.9, 6.6)], fill=(205, 208, 213), width=ss * 2)
    # the roundel: about two thirds of the chord across, a little below
    # mid-height and close to the leading edge
    cx, cy = p(65.0, 6.6)
    rx, ry = 1.95 / (s1 - s0) * n, 1.95 / (y1 - y0) * n
    d.ellipse([cx - rx, cy - ry, cx + rx, cy + ry], fill=RED)
    return im.resize((N, N), Image.LANCZOS)


def fan_texture(N=512):
    ss = 3
    n = N * ss
    yy, xx = np.mgrid[0:n, 0:n]
    dx, dy = (xx + 0.5) / n * 2 - 1, (yy + 0.5) / n * 2 - 1
    r, a = np.hypot(dx, dy), np.arctan2(dy, dx)
    img = np.zeros((n, n, 3), np.float32)
    # 22 wide-chord composite blades, swept with radius
    ph = (a + 1.15 * r) * 22 / (2 * math.pi)
    blade = 0.5 + 0.5 * np.cos(ph * 2 * math.pi)
    shade = 12 + 46 * blade ** 0.7 * (0.55 + 0.45 * r)
    img[:] = shade[:, :, None]
    edge = (blade > 0.93) & (r > 0.3)
    img[edge] = (150, 152, 158)  # titanium leading edges
    hub = r < 0.3
    img[hub] = (38, 40, 44)
    # the white swirl on the spinner
    sw = hub & (np.abs(((a + 5.2 * (0.3 - r) / 0.3 * 2.4) % (2 * math.pi)) - math.pi) < 0.34) & (r > 0.035) & (r < 0.24)
    img[sw] = (235, 236, 238)
    img[r > 0.985] = (24, 25, 28)
    return Image.fromarray(np.clip(img, 0, 255).astype(np.uint8)).resize((N, N), Image.LANCZOS)


# ------------------------------------------------------------------ build
def build():
    M = {k: Mesh() for k in ('fuselage', 'wingtop', 'grey', 'fin', 'white', 'chrome', 'fan', 'dark', 'hot', 'red_l', 'green_l', 'white_l')}

    # fuselage: dense rings through the nose and the tail cone
    st = np.unique(np.concatenate([np.linspace(0, 1.2, 14) ** 1.0, np.linspace(1.2, 10.5, 56), np.linspace(10.5, 45, 58), np.linspace(45, L, 64)]))
    st[0] = 0.004
    seg = 72
    th = np.linspace(0, 2 * math.pi, seg + 1)
    S, T = np.meshgrid(st, th, indexing='ij')
    x, y = fus_point(S, T)
    M['fuselage'].add(*grid(to_model(x, y, S), np.stack([S / L, T / (2 * math.pi)], -1)))
    # close the tail: the APU exhaust
    xe, ye = fus_point(np.full(seg + 1, L), th)
    cap = np.stack([to_model(xe, ye, np.full(seg + 1, L)), to_model(xe * 0.0, ye * 0 + ye.mean(), np.full(seg + 1, L + 0.05))])
    M['dark'].add(*grid(cap, np.zeros((2, seg + 1, 2)) + 0.5, outward=(0, 0, -1)))

    # wing-to-body fairing and the satcom radome on the crown
    M['grey'].add(*grid(*ellipsoid((0, -2.55, ORIGIN_S - 35.6), (3.45, 1.35, 10.4), 36, 20)))
    M['white'].add(*grid(*ellipsoid((0, R - 0.08, ORIGIN_S - 23.4), (0.62, 0.4, 2.0), 20, 12)))
    M['white'].add(*grid(*ellipsoid((0, R - 0.05, ORIGIN_S - 46.5), (0.5, 0.3, 1.3), 20, 12)))

    # wings
    stations = [0.0, 1.5, 3.0, 4.5, 6.0, 8.0, 10.0, 12.5, 15.0, 17.5, 20.0, 22.5, 25.0, 27.5, 29.2, 30.0, 30.8, 31.6, 32.1, 32.4]
    for side in (1, -1):
        up, lo, tip = surface(stations, wing_le, wing_chord, wing_y, wing_thick, side, None)
        def wuv(p):
            return np.stack([(p[..., 0] + 33.0) / 66.0, ((ORIGIN_S - p[..., 2]) - 26.0) / 26.0], -1)
        M['wingtop'].add(*grid(up, wuv(up), outward=(0, 1, 0)))
        M['grey'].add(*grid(lo, np.zeros(lo.shape[:2] + (2,)) + 0.5, outward=(0, -1, 0)))
        # navigation lights at the tips
        tipx = 32.3 * side
        M['red_l' if side > 0 else 'green_l'].add(*grid(*ellipsoid((tipx, float(wing_y(32.3)) + 0.02, ORIGIN_S - float(wing_le(32.3)) - 0.1), (0.12, 0.06, 0.22), 10, 6)))
        # flap track fairings
        for fx in (6.2, 12.6, 18.9, 24.6):
            te = float(wing_le(fx) + wing_chord(fx))
            M['grey'].add(*grid(*ellipsoid((fx * side, float(wing_y(fx)) - 0.42, ORIGIN_S - te + 0.9), (0.28, 0.36, 2.5 if fx < 20 else 1.8), 14, 10)))

    # horizontal stabilisers
    hs_le = lambda xx: 61.2 + 0.74 * abs(xx)
    hs_ch = lambda xx: float(np.interp(abs(xx), [0, 10.3, 10.75], [7.4, 2.4, 1.2]))
    hs_y = lambda xx: 0.95 + 0.12 * abs(xx)
    hs_t = lambda xx: 0.1
    for side in (1, -1):
        up, lo, _ = surface([0.0, 1.0, 2.5, 4.5, 6.5, 8.5, 10.0, 10.5, 10.75], hs_le, hs_ch, hs_y, hs_t, side, None, n=16)
        z = np.zeros(up.shape[:2] + (2,)) + 0.5
        M['white'].add(*grid(up, z, outward=(0, 1, 0)))
        M['white'].add(*grid(lo, np.zeros(lo.shape[:2] + (2,)) + 0.5, outward=(0, -1, 0)))

    # fin
    fin_le = lambda yy: 56.6 + 1.02 * (yy - 1.5)
    fin_ch = lambda yy: float(np.interp(yy, [1.5, 11.9, 12.35], [10.0, 3.3, 1.8]))
    s0, s1, y0, y1 = FIN_BOX
    for side in (1, -1):
        up, lo, _ = surface([1.5, 3.0, 4.5, 6.0, 7.5, 9.0, 10.5, 11.6, 12.1, 12.35], fin_le, fin_ch, None, lambda yy: 0.105, side, None, vertical=True, n=18)
        for half in (up,):
            uv = np.stack([((ORIGIN_S - half[..., 2]) - s0) / (s1 - s0), 1 - (half[..., 1] - y0) / (y1 - y0)], -1)
            M['fin'].add(*grid(half, uv, outward=(side, 0, 0)))
    M['white_l'].add(*grid(*ellipsoid((0, float(fus_top(L)) - 0.2, ORIGIN_S - L - 0.02), (0.1, 0.1, 0.12), 10, 6)))
    # beacons
    M['red_l'].add(*grid(*ellipsoid((0, R + 0.05, ORIGIN_S - 30.0), (0.11, 0.1, 0.2), 10, 6)))
    M['red_l'].add(*grid(*ellipsoid((0, -3.92, ORIGIN_S - 36.0), (0.11, 0.08, 0.2), 10, 6)))
    # blade antennas
    for (sv, yy, hh) in ((16.0, R, 0.38), (27.0, R, 0.3), (52.0, R, 0.38), (11.5, -R, -0.34), (20.0, -R, -0.3)):
        M['white'].add(*grid(*ellipsoid((0, yy + hh * 0.5, ORIGIN_S - sv), (0.035, abs(hh) * 0.75, 0.3), 8, 6)))

    # engines
    for side in (1, -1):
        ex = 9.75 * side
        cy = -3.0
        zc = ORIGIN_S - 24.0      # intake lip station
        c = (ex, cy, zc)
        # outer cowl: lip, fan cowl, fan nozzle
        outer = [(0.0, 1.66), (0.05, 1.74), (0.16, 1.8), (0.4, 1.87), (0.9, 1.94), (1.8, 1.99), (2.8, 1.985), (3.8, 1.93), (4.7, 1.82), (5.4, 1.68), (5.75, 1.6)]
        M['white'].add(*grid(*revolve(outer[2:], c, seg=56)))
        M['chrome'].add(*grid(*revolve([(0.34, 1.52), (0.12, 1.555), (0.02, 1.6), (0.0, 1.66), (0.05, 1.74), (0.16, 1.8)], c, seg=56)))
        # intake duct down to the fan
        duct = revolve([(0.34, 1.52), (0.9, 1.55), (1.25, 1.6)], c, seg=56)
        M['dark'].add(*grid(*duct, outward=None))
        M['dark'].idx[-1] = M['dark'].idx[-1][:, ::-1]  # seen from the inside
        # fan nozzle lip, core cowl, turbine nozzle and plug
        M['dark'].add(*grid(*revolve([(5.75, 1.6), (5.72, 1.5), (5.2, 1.42)], c, seg=56), outward=None))
        M['white'].add(*grid(*revolve([(5.0, 1.36), (5.6, 1.3), (6.3, 1.12), (7.0, 0.9), (7.45, 0.78)], c, seg=48)))
        M['hot'].add(*grid(*revolve([(7.45, 0.78), (7.9, 0.7), (8.25, 0.62), (8.22, 0.56), (7.6, 0.5)], c, seg=40)))
        M['hot'].add(*grid(*revolve([(7.7, 0.46), (8.3, 0.4), (8.9, 0.22), (9.3, 0.02)], c, seg=32)))
        # pylon: a thin fairing from the cowl up into the wing
        prof, npts = airfoil(12, 1.0, 0.0)
        rows = []
        for (yy, s_le, chord, tk) in ((cy + 1.7, 25.6, 9.2, 0.07), (cy + 2.15, 26.6, 9.6, 0.075), (float(wing_y(9.75)) - 0.15, 29.6, 8.0, 0.07), (float(wing_y(9.75)) + 0.1, 31.0, 6.5, 0.06)):
            rows.append(np.stack([ex + prof[:, 1] * tk * chord, np.full(len(prof), yy), ORIGIN_S - (s_le + prof[:, 0] * chord)], 1))
        M['white'].add(*grid(np.array(rows), np.zeros((4, len(prof), 2)) + 0.5))

    return {k: m.done() for k, m in M.items() if m.n}


FAN_AT = (9.75, -3.0, ORIGIN_S - 24.0)  # engine axis at the intake lip (left engine)


def build_fan():
    """The fan and spinner on their own, about the Z axis, so the game can spin them."""
    m = Mesh()
    thf = np.linspace(0, 2 * math.pi, 73)
    rr = np.array([0.0, 0.14, 0.3, 0.48, 1.6])[:, None]
    zz = np.array([0.5, 0.6, 0.82, 1.15, 1.22])[:, None]
    fan = np.stack([rr * np.cos(thf), rr * np.sin(thf), -zz + 0 * thf], -1)
    fuv = np.stack([0.5 + 0.5 * rr / 1.6 * np.cos(thf), 0.5 - 0.5 * rr / 1.6 * np.sin(thf)], -1)
    m.add(*grid(fan, fuv, outward=(0, 0, 1)))
    return {'fan': m.done()}


def png(im):
    b = io.BytesIO()
    im.save(b, 'PNG', optimize=True)
    return b.getvalue()


def jpg(im, q=90):
    b = io.BytesIO()
    im.convert('RGB').save(b, 'JPEG', quality=q, optimize=True, subsampling=0)
    return b.getvalue()


MATS = {
    'fuselage': dict(color=(0.86, 0.86, 0.86), rough=0.3, metal=0.0),
    'wingtop': dict(color=(0.84, 0.84, 0.84), rough=0.42, metal=0.1),
    'grey': dict(color=(0.6, 0.62, 0.65), rough=0.42, metal=0.1),
    'fin': dict(color=(0.86, 0.86, 0.86), rough=0.3, metal=0.0),
    'white': dict(color=(0.81, 0.815, 0.825), rough=0.3, metal=0.0),
    'chrome': dict(color=(0.86, 0.87, 0.89), rough=0.16, metal=1.0),
    'fan': dict(rough=0.45, metal=0.5),
    'dark': dict(color=(0.03, 0.032, 0.036), rough=0.6, metal=0.2),
    'hot': dict(color=(0.3, 0.27, 0.24), rough=0.38, metal=1.0),
    'red_l': dict(color=(0.6, 0.02, 0.02), rough=0.3, metal=0.0, emissive=(6.0, 0.15, 0.1)),
    'green_l': dict(color=(0.02, 0.5, 0.1), rough=0.3, metal=0.0, emissive=(0.1, 5.0, 0.6)),
    'white_l': dict(color=(0.9, 0.9, 0.9), rough=0.3, metal=0.0, emissive=(5.0, 5.0, 5.0)),
}


def main(out):
    write(out, build(), {'fuselage': jpg(fuselage_texture()), 'wingtop': jpg(wing_texture(), 88), 'fin': jpg(fin_texture(), 88)})
    write(out.replace('.glb', '_fan.glb'), build_fan(), {'fan': jpg(fan_texture(), 88)})
    print('fan axis (left engine):', FAN_AT)


def write(out, meshes, textures):
    mats = MATS
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

    tris = 0
    for name, (v, n, uv, idx) in meshes.items():
        m = mats[name]
        pbr = {'baseColorFactor': list(m.get('color', (1, 1, 1))) + [1], 'metallicFactor': m['metal'], 'roughnessFactor': m['rough']}
        if name in textures:
            images.append({'bufferView': view(textures[name]), 'mimeType': 'image/jpeg'})
            texs.append({'source': len(images) - 1, 'sampler': 0})
            pbr['baseColorTexture'] = {'index': len(texs) - 1}
        mat = {'name': name, 'pbrMetallicRoughness': pbr}
        if 'emissive' in m:
            mat['emissiveFactor'] = [min(1.0, e) for e in m['emissive']]
            mat['extensions'] = {'KHR_materials_emissive_strength': {'emissiveStrength': max(m['emissive'])}}
        materials.append(mat)
        prims.append({
            'attributes': {
                'POSITION': acc(v.astype(np.float32), 5126, 'VEC3', 34962, True),
                'NORMAL': acc(n.astype(np.float32), 5126, 'VEC3', 34962),
                'TEXCOORD_0': acc(uv.astype(np.float32), 5126, 'VEC2', 34962),
            },
            'indices': acc(idx.reshape(-1).astype(np.uint32), 5125, 'SCALAR', 34963),
            'material': len(materials) - 1,
        })
        tris += len(idx)
    gltf = {
        'asset': {'version': '2.0', 'generator': 'zonefall build-b777.py'},
        'extensionsUsed': ['KHR_materials_emissive_strength'],
        'scene': 0, 'scenes': [{'nodes': [0]}], 'nodes': [{'mesh': 0, 'name': 'B777-300ER'}],
        'meshes': [{'primitives': prims}], 'materials': materials, 'textures': texs, 'images': images,
        'samplers': [{'magFilter': 9729, 'minFilter': 9987, 'wrapS': 33071, 'wrapT': 10497}],
        'accessors': accessors, 'bufferViews': views, 'buffers': [{'byteLength': len(bin_)}],
    }
    js = json.dumps(gltf, separators=(',', ':')).encode()
    js += b' ' * (-len(js) % 4)
    while len(bin_) % 4:
        bin_.append(0)
    with open(out, 'wb') as f:
        f.write(struct.pack('<III', 0x46546C67, 2, 12 + 8 + len(js) + 8 + len(bin_)))
        f.write(struct.pack('<II', len(js), 0x4E4F534A) + js)
        f.write(struct.pack('<II', len(bin_), 0x004E4942) + bytes(bin_))
    print(f'{out}: {tris} triangles, {sum(len(m[0]) for m in meshes.values())} vertices, {(12 + 16 + len(js) + len(bin_)) / 1e6:.2f} MB')


if __name__ == '__main__':
    main(sys.argv[1])
