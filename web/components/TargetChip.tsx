"use client";

import { useMemo } from "react";

type Box2d = [number, number, number, number]; // [y1, x1, y2, x2] in 0..1000

type Props = {
  box: Box2d;
  label?: string;
};

const clamp = (v: number, min: number, max: number) => Math.max(min, Math.min(max, v));

export default function TargetChip({ box, label = "Target" }: Props) {
  const meta = useMemo(() => {
    const [y1, x1, y2, x2] = box;
    const left = clamp((x1 / 1000) * 100, 0, 100);
    const top = clamp((y1 / 1000) * 100, 0, 100);
    const maxW = Math.max(0.8, 100 - left);
    const maxH = Math.max(0.8, 100 - top);
    const width = clamp(((x2 - x1) / 1000) * 100, 0.8, maxW);
    const height = clamp(((y2 - y1) / 1000) * 100, 0.8, maxH);
    const cx = Math.round(clamp((x1 + x2) / 2, 0, 1000));
    const cy = Math.round(clamp((y1 + y2) / 2, 0, 1000));
    const x1i = Math.round(clamp(x1, 0, 1000));
    const y1i = Math.round(clamp(y1, 0, 1000));
    const x2i = Math.round(clamp(x2, 0, 1000));
    const y2i = Math.round(clamp(y2, 0, 1000));
    return { left, top, width, height, cx, cy, x1i, y1i, x2i, y2i };
  }, [box]);

  const title = `box_2d: [${box.join(", ")}]`;

  return (
    <div
      title={title}
      className="inline-flex items-center gap-2 rounded-full border border-emerald-500/20 bg-emerald-500/[0.12] px-3 py-1.5 text-emerald-100 shadow-[0_0_0_1px_rgba(255,255,255,0.05)]"
    >
      <span className="text-[11px] font-extrabold tracking-[0.16em] uppercase">{label}</span>
      <span className="font-mono text-[11px] tabular-nums text-emerald-100">
        x{meta.cx} y{meta.cy}
      </span>
    </div>
  );
}

