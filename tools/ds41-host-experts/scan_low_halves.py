"""Check that every f32 router and compressor kv|gate weight in a .rad container is bf16-valued:
the low 16 bits of every element zero (what RADIANCE_EXPERIMENTAL_DS41_ROUTER_BF16's relayout
requires, or it refuses the load). Reads abi/rad_format.h's layout directly; prints each entry's
name, encoding, shape and the count of nonzero low halves (a bf16 plane has nothing to narrow).
Reads the planes with pread and drops them from the page cache afterwards; about 300 MiB for
DeepSeek-V4.1, so run it outside a GPU window (ionice -c3).

The native MXFP4 container (ds41-native-mxfp4-v3-20261027T121600Z, 29 Oct 2026): all 40 routers
f32 [384, 5120] with 0 nonzero low halves; the three compressors' kv|gate bf16 [1024, 5120].

usage: python3 -I scan_low_halves.py model.rad"""
import os, re, struct, sys

path = sys.argv[1]
fd = os.open(path, os.O_RDONLY)
def pread(off, n):
    b = os.pread(fd, n, off)
    if len(b) != n: raise SystemExit(f"short read at {off}")
    return b

h = pread(0, 256)
magic, version, file_bytes = struct.unpack_from('<IIQ', h, 0)
(str_off, str_bytes, meta_off, meta_count, dir_off, dir_count, plane_off, plane_count,
 enc_off, enc_count) = struct.unpack_from('<10Q', h, 56)
strs = pread(str_off, str_bytes)
def s(off):
    if off == 0: return ''
    e = strs.index(b'\0', off)
    return strs[off:e].decode()
ENTRY = 136
dirb = pread(dir_off, dir_count * ENTRY)
planes = pread(plane_off, plane_count * 16)
ENC = 568
encs = pread(enc_off, enc_count * ENC)
def enc_desc(i):
    e = encs[i * ENC:(i + 1) * ENC]
    scheme = e[0:24].split(b'\0')[0].decode(); transform = e[24:48].split(b'\0')[0].decode()
    n = struct.unpack_from('<i', e, 48)[0]
    ps = []
    for k in range(n):
        p = e[56 + 64 * k: 56 + 64 * (k + 1)]
        role = p[0:24].split(b'\0')[0].decode(); dtype = struct.unpack_from('<I', p, 24)[0]
        ps.append((role, dtype))
    return scheme, transform, ps
pat = re.compile(r'(glm_router\.weight|attn_compressor_kvg\.weight)$')
total_bad = 0; n_ent = 0; nbytes = 0
for i in range(dir_count):
    e = dirb[i * ENTRY:(i + 1) * ENTRY]
    name = s(struct.unpack_from('<Q', e, 0)[0])
    if not pat.search(name): continue
    enc, rank = struct.unpack_from('<II', e, 24)
    shape = struct.unpack_from('<6q', e, 32)[:rank]
    n_planes = struct.unpack_from('<I', e, 92)[0]
    plane_first, offset, size = struct.unpack_from('<3Q', e, 96)
    scheme, transform, ps = enc_desc(enc)
    poff, pbytes = struct.unpack_from('<2Q', planes, plane_first * 16)
    n_ent += 1
    DT = {1: 'f32', 2: 'f16', 3: 'bf16'}
    if n_planes != 1 or ps[0][1] != 1:
        print(f"{name:42s} {scheme}/{transform or '-'} {[(r, DT.get(d, d)) for r, d in ps]} shape "
              f"{list(shape)} {pbytes} bytes: not an f32 plane, nothing to narrow")
        continue
    data = pread(poff, pbytes)
    os.posix_fadvise(fd, poff, pbytes, os.POSIX_FADV_DONTNEED)
    # little-endian f32: bytes 0 and 1 of each word are the low half
    lo = data[0::4] + data[1::4]
    bad = len(lo) - lo.count(0)
    nz_words = sum(1 for k in range(0, len(data), 4) if data[k] or data[k + 1]) if bad else 0
    total_bad += nz_words; nbytes += pbytes
    print(f"{name:42s} {scheme}/{transform or '-'} [('codes', 'f32')] shape {list(shape)} "
          f"{pbytes} bytes: {nz_words} elements with a nonzero low half")
print(f"{n_ent} entries, {nbytes / 2**20:.1f} MiB read, {total_bad} nonzero low halves in all")
os.close(fd)
