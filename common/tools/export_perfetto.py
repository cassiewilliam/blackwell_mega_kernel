#!/usr/bin/env python3
"""export_perfetto.py —— 把 device 侧 profiler buffer 解码成 Perfetto trace。

对应 detail/profiler.cuh 的编码格式，以及 FlashInfer profiler/__init__.py 的
导出思路（每个 SM 一条 track，下挂 blk{block}_g{group} 子 track）。

buffer 布局（int64/uint64 数组）：
    [0]            : header，低 32 位 = num_blocks，高 32 位 = num_groups
    [1 + b*G + g + k*(B*G)] : 第 (block b, group g) 的第 k 条事件
        低 32 位 tag :  [1:0]=type  [11:2]=event_idx  [23:12]=block  [31:24]=sm
        高 32 位     :  globaltimer_lo 时间戳（ns）

用法::
    python tools/export_perfetto.py prof.bin -o trace.json
    # 然后把 trace.json 拖进 https://ui.perfetto.dev
"""
from __future__ import annotations

import argparse
import json
import struct

# Default event names = mega_moe warp roles (event_idx order; see the kernel probes).
# Override with --events, comma-separated, in event_idx order.
DEFAULT_EVENT_NAMES = ["Dispatch", "TMA-A", "TMA-B", "MMA", "Epilogue", "L1", "L2"]
TYPE_BEGIN, TYPE_END, TYPE_INSTANT = 0, 1, 2


def decode(raw: bytes):
    n = len(raw) // 8
    words = struct.unpack(f"<{n}Q", raw[: n * 8])
    header = words[0]
    num_blocks = header & 0xFFFFFFFF
    num_groups = (header >> 32) & 0xFFFFFFFF
    events = []
    for w in words[1:]:
        if w == 0:
            continue
        tag = w & 0xFFFFFFFF
        ts = (w >> 32) & 0xFFFFFFFF
        etype = tag & 0x3
        eidx = (tag >> 2) & 0x3FF
        block = (tag >> 12) & 0xFFF
        sm = (tag >> 24) & 0xFF
        events.append(dict(ts=ts, type=etype, eidx=eidx, block=block, sm=sm))
    return num_blocks, num_groups, events


PID = 0          # single process
ROLES_PER_SM = 8  # tid stride: one track per (SM, role); roles use event_idx 0..4


# Map event_idx -> (lane_id, lane_name). Events on the same lane share a track and
# render as alternating colored slices — matching PR #316's "(c) Ours" diagram rows:
#   Dispatch | Computation (L1,L2) | Activation&Combine (Act,Combine).
LANE = {
    0: (0, "Dispatch"),
    1: (1, "TMA-A"), 2: (2, "TMA-B"), 3: (3, "MMA"), 4: (5, "Act&Combine"),
    5: (3, "Computation"), 6: (3, "Computation"),   # L1, L2 -> one Computation row
    7: (5, "Act&Combine"), 8: (5, "Act&Combine"),   # Act, Combine -> one row
}
LANES_PER_SM = 8


def to_chrome_trace(events, event_names):
    """Chrome/Perfetto trace JSON. ONE process; per SM a few LANES (PR-diagram rows).

    Events that belong to the same lane (e.g. L1+L2 -> Computation) share a tid and
    render as a sequence of colored slices on one row (slice name = the event name,
    so Perfetto colors L1 vs L2 differently), just like the PR "(c) Ours" figure.
    """
    out = []
    tids = {}  # tid -> (sm, lane_name)
    for e in events:
        eidx = e["eidx"]
        ename = event_names[eidx] if eidx < len(event_names) else f"ev{eidx}"
        lane_id, lane_name = LANE.get(eidx, (eidx, ename))
        tid = e["sm"] * LANES_PER_SM + lane_id
        ph = {TYPE_BEGIN: "B", TYPE_END: "E", TYPE_INSTANT: "i"}[e["type"]]
        rec = dict(name=ename, ph=ph, ts=e["ts"] / 1000.0,
                   pid=PID, tid=tid, args={"cta": e["block"], "sm": e["sm"]})
        if ph == "i":
            rec["s"] = "t"
        out.append(rec)
        tids[tid] = (e["sm"], lane_name)
    out.append(dict(name="process_name", ph="M", pid=PID,
                    args={"name": "MegaMoE (per-SM, PR-style lanes)"}))
    for tid, (sm, lane_name) in sorted(tids.items()):
        out.append(dict(name="thread_name", ph="M", pid=PID, tid=tid,
                        args={"name": f"SM{sm:03d} {lane_name}"}))
        out.append(dict(name="thread_sort_index", ph="M", pid=PID, tid=tid,
                        args={"sort_index": tid}))
    return {"traceEvents": out, "displayTimeUnit": "ns"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("buffer", help="profiler buffer dump (raw little-endian uint64)")
    ap.add_argument("-o", "--out", default="trace.json")
    ap.add_argument("--events", default=None,
                    help="comma-separated event names (by event_idx); default = mega_moe table")
    ap.add_argument("--max-sms", type=int, default=0,
                    help="keep only the first N SM ids (smaller file; SMs run in lockstep)")
    ap.add_argument("--roles", default=None,
                    help="comma-separated event_idx to keep, e.g. '5,6' for L1/L2 only "
                         "(drops coarse role spans so the view scales to the compute tiles)")
    args = ap.parse_args()

    event_names = args.events.split(",") if args.events else DEFAULT_EVENT_NAMES
    with open(args.buffer, "rb") as f:
        raw = f.read()
    nb, ng, events = decode(raw)
    if args.roles:
        keep_e = {int(x) for x in args.roles.split(",")}
        events = [e for e in events if e["eidx"] in keep_e]
    if args.max_sms:
        keep = sorted({e["sm"] for e in events})[:args.max_sms]
        events = [e for e in events if e["sm"] in keep]
    trace = to_chrome_trace(events, event_names)
    with open(args.out, "w") as f:
        json.dump(trace, f)
    print(f"blocks={nb} groups={ng} events={len(events)} -> {args.out}")
    print("打开 https://ui.perfetto.dev 并加载该 json")


if __name__ == "__main__":
    main()
