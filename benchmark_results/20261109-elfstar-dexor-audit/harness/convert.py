# .blocks (int64 raw) -> .f64: int32 nblocks, then per block int32 count + LE doubles = raw/10^dec (IEEE division)
import struct, sys, os
src, dst = sys.argv[1], sys.argv[2]
for f in sorted(os.listdir(src)):
    if not f.endswith('.blocks'): continue
    b = open(os.path.join(src, f), 'rb').read()
    dec, nb = struct.unpack_from('<qq', b, 0); off = 16
    div = float(10**dec)
    out = [struct.pack('<i', nb)]
    for _ in range(nb):
        bid, cnt = struct.unpack_from('<qq', b, off); off += 16
        raw = struct.unpack_from('<%dq' % cnt, b, off); off += 8*cnt
        out.append(struct.pack('<i', cnt)); out.append(struct.pack('<%dd' % cnt, *[float(r)/div for r in raw]))
    open(os.path.join(dst, f[:-7] + '.f64'), 'wb').write(b''.join(out))
