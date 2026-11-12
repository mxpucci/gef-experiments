# Runs on server: emit per dataset a sample of 1000-value blocks as raw int64 + decimals.
# Blocks: first 10 blocks + 10 blocks evenly spaced (block-aligned) over the dataset.
import os, struct, sys
d = sys.argv[1]; out = sys.argv[2]
for f in sorted(os.listdir(d)):
    if not f.endswith('.bin'): continue
    p = os.path.join(d, f); sz = os.path.getsize(p)
    with open(p, 'rb') as fh:
        n, = struct.unpack('<Q', fh.read(8))
        hdr = 16 if sz == 16 + 8*n else 8
        dec = struct.unpack('<q', fh.read(8))[0] if hdr == 16 else 0
        nb = (n + 999)//1000
        blocks = list(range(min(10, nb)))
        for k in range(10):
            b = (nb * (2*k+1)) // 20
            if b not in blocks: blocks.append(b)
        with open(os.path.join(out, f[:-4] + '.blocks'), 'wb') as o:
            o.write(struct.pack('<qq', dec, len(blocks)))
            for b in blocks:
                cnt = min(1000, n - b*1000)
                fh.seek(hdr + 8*b*1000); data = fh.read(8*cnt)
                o.write(struct.pack('<qq', b, cnt)); o.write(data)
    print(f, n, dec, len(blocks))
