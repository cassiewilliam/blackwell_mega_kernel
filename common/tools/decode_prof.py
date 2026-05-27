#!/usr/bin/env python3
"""Decode a mega profiler prof.bin into per-(role/block) durations.

Unlike a tag-only decode, this recovers the per-iteration cursor from the buffer
SLOT POSITION (the tag only carries blockIdx, not the loop iteration):
    slot i (i>=1):  cursor = (i-1)//(nb*ng); rest=(i-1)%(nb*ng); block=rest//ng; group=rest%ng
begin = even cursor, end = odd cursor; pair by (block, group, cursor//2).
"""
import collections
import statistics as st
import struct
import sys

NAMES = {0: "Dispatch", 1: "TMA-A", 2: "TMA-B", 3: "MMA", 4: "Epilogue", 5: "L1", 6: "L2"}
W = 2 ** 32


def main(path):
    raw = open(path, "rb").read()
    w = struct.unpack_from(f"<{len(raw)//8}Q", raw)
    nb, ng = w[0] & 0xFFFFFFFF, (w[0] >> 32) & 0xFFFFFFFF
    stride = nb * ng
    print(f"num_blocks={nb} num_groups={ng}")

    slices = collections.defaultdict(dict)  # (block,group,iter) -> {0:tb,1:te,ev,sm}
    for i in range(1, len(w)):
        x = w[i]
        if x == 0:
            continue
        tag = x & 0xFFFFFFFF
        ts = (x >> 32) & 0xFFFFFFFF
        typ, eidx, sm = tag & 3, (tag >> 2) & 0x3FF, (tag >> 24) & 0xFF
        p = i - 1
        cursor, rest = p // stride, p % stride
        block, group = rest // ng, rest % ng
        s = slices[(block, group, cursor // 2)]
        s[typ] = ts
        s["ev"], s["sm"] = eidx, sm

    byname = collections.defaultdict(list)
    persm = collections.defaultdict(collections.Counter)
    for v in slices.values():
        if 0 in v and 1 in v:
            byname[v["ev"]].append((v[1] - v[0]) % W)
            persm[v["ev"]][v["sm"]] += 1
    for e in sorted(byname):
        ds = sorted(byname[e])
        bpsm = sorted(persm[e].values())
        print(f"  {NAMES.get(e, e):9s} n={len(ds):5d}  dur_med={int(st.median(ds)):8d} ns  "
              f"[{ds[0]:8d}..{ds[-1]:8d}]  blocks/SM med={int(st.median(bpsm)) if (bpsm:=bpsm) else 0}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "prof.bin")
