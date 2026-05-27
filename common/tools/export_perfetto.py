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

# 默认事件名表 = mega_moe（顺序须与 kernels/mega_moe/include/mega_moe/events.h 一致）。
# 其它 kernel 用 --events 覆盖，逗号分隔，按 event_idx 顺序排列。
DEFAULT_EVENT_NAMES = [
    "Dispatch", "Linear1", "SwiGLU", "Linear2", "Combine",
    "TMA-A", "TMA-B", "MMA-issue", "Barrier",
]
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


def to_chrome_trace(events, event_names):
    """Perfetto 可直接读 Chrome trace JSON（ph=B/E/i）。track = SM，tid = block。"""
    out = []
    for e in events:
        name = event_names[e["eidx"]] if e["eidx"] < len(event_names) else f"ev{e['eidx']}"
        ph = {TYPE_BEGIN: "B", TYPE_END: "E", TYPE_INSTANT: "i"}[e["type"]]
        rec = dict(name=name, ph=ph, ts=e["ts"] / 1000.0,  # ns → us
                   pid=e["sm"], tid=e["block"])
        if ph == "i":
            rec["s"] = "t"
        out.append(rec)
    # 给每个 SM/ block 命名
    return {"traceEvents": out, "displayTimeUnit": "ns"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("buffer", help="profiler buffer dump (raw little-endian uint64)")
    ap.add_argument("-o", "--out", default="trace.json")
    ap.add_argument("--events", default=None,
                    help="逗号分隔的事件名表（按 event_idx 顺序）；缺省用 mega_moe 默认表")
    args = ap.parse_args()

    event_names = args.events.split(",") if args.events else DEFAULT_EVENT_NAMES
    with open(args.buffer, "rb") as f:
        raw = f.read()
    nb, ng, events = decode(raw)
    trace = to_chrome_trace(events, event_names)
    with open(args.out, "w") as f:
        json.dump(trace, f)
    print(f"blocks={nb} groups={ng} events={len(events)} -> {args.out}")
    print("打开 https://ui.perfetto.dev 并加载该 json")


if __name__ == "__main__":
    main()
