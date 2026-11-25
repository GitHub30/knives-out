#!/usr/bin/env python3
"""Shrink a GLB for the web: embedded textures are scaled down and re-encoded as JPEG.

usage: optimize-glb.py in.glb out.glb [max_texture_size=1024]
Geometry, materials and node transforms are kept as they are.
"""
import io, json, struct, sys
from PIL import Image

src, dst = sys.argv[1], sys.argv[2]
max_size = int(sys.argv[3]) if len(sys.argv) > 3 else 1024
data = open(src, 'rb').read()
magic, version, _ = struct.unpack('<III', data[:12])
assert magic == 0x46546C67 and version == 2
jlen, _ = struct.unpack('<II', data[12:20])
gltf = json.loads(data[20:20 + jlen])
blen, _ = struct.unpack('<II', data[20 + jlen:28 + jlen])
bin_ = data[28 + jlen:28 + jlen + blen]

image_views = {im['bufferView']: im for im in gltf.get('images', []) if 'bufferView' in im}
out = bytearray()
for i, bv in enumerate(gltf['bufferViews']):
    chunk = bin_[bv.get('byteOffset', 0):bv.get('byteOffset', 0) + bv['byteLength']]
    if i in image_views:
        img = Image.open(io.BytesIO(chunk)).convert('RGB')
        if max(img.size) > max_size:
            k = max_size / max(img.size)
            img = img.resize((round(img.width * k), round(img.height * k)), Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, 'JPEG', quality=84, optimize=True)
        chunk = buf.getvalue()
        image_views[i]['mimeType'] = 'image/jpeg'
    while len(out) % 4:
        out.append(0)
    bv['byteOffset'] = len(out)
    bv['byteLength'] = len(chunk)
    out += chunk
while len(out) % 4:
    out.append(0)
gltf['buffers'] = [{'byteLength': len(out)}]
j = json.dumps(gltf, separators=(',', ':')).encode()
j += b' ' * (-len(j) % 4)
total = 12 + 8 + len(j) + 8 + len(out)
with open(dst, 'wb') as f:
    f.write(struct.pack('<III', 0x46546C67, 2, total))
    f.write(struct.pack('<II', len(j), 0x4E4F534A))
    f.write(j)
    f.write(struct.pack('<II', len(out), 0x004E4942))
    f.write(out)
print(f'{src}: {len(data)} -> {total} bytes')
