/**
 * GraphLegend — the single visual-encoding key for the L1 graph, floating over
 * the canvas. Collapsible so it never fights the graph for space.
 */
import { useState } from 'react';
import { COLOR } from '../lib/render';
import { PhiAsym } from './Phi';

export function GraphLegend() {
  const [open, setOpen] = useState(true);
  return (
    <div className="float pointer-events-auto absolute right-3 top-3 z-10 w-[268px] overflow-hidden text-[12px] text-ink-200">
      <button
        type="button"
        onClick={() => setOpen(o => !o)}
        aria-expanded={open}
        className="flex w-full items-center justify-between px-3 py-2 text-left transition-colors hover:bg-white/[0.04]"
      >
        <span className="eyebrow">圖例 legend</span>
        <span className={`text-ink-400 transition-transform duration-300 ${open ? 'rotate-180' : ''}`} aria-hidden>⌄</span>
      </button>
      {open && (
        <div className="hairline space-y-1.5 border-t px-3 py-2.5 leading-snug">
          <Row swatch={<Square color={COLOR.tx} />} label="交易 transaction（方形）" />
          <Row swatch={<Dot color={COLOR.wallet} />} label="錢包 wallet（圓形）" />
          <Row
            swatch={
              <span className="flex items-center gap-0.5">
                <Dot color={COLOR.fraud} />
                <Square color={COLOR.fraud} />
              </span>
            }
            label="非法 illicit（紅色，形狀依型別）"
          />
          <Row swatch={<Dot color={COLOR.dim} />} label="一階鄰居（反灰）" />
          <hr className="hairline my-1.5 border-t" />
          <Row swatch={<Line color={COLOR.cePos} />} label="CE 為正（越強邊越寬）" />
          <Row swatch={<Line color={COLOR.ceNeg} />} label="CE 為負" />
          <hr className="hairline my-1.5 border-t" />
          <Row swatch={<Ring color={COLOR.root} />} label="根因 root（光暈）" />
          <Row
            swatch={<span className="w-3.5 text-center" style={{ color: COLOR.pivot }}>★</span>}
            label={<>元兇 pivot（球越大 ＝ <PhiAsym /> 越高）</>}
          />
        </div>
      )}
    </div>
  );
}

function Row({ swatch, label }: { swatch: React.ReactNode; label: React.ReactNode }) {
  return (
    <div className="flex items-center gap-2.5">
      <span className="flex w-4 shrink-0 items-center justify-center">{swatch}</span>
      <span>{label}</span>
    </div>
  );
}

const Dot = ({ color }: { color: string }) => <span className="h-2.5 w-2.5 rounded-full" style={{ background: color }} />;
const Square = ({ color }: { color: string }) => <span className="h-2.5 w-2.5 rounded-[2px]" style={{ background: color }} />;
const Line = ({ color }: { color: string }) => <span className="h-0.5 w-3.5 rounded" style={{ background: color }} />;
const Ring = ({ color }: { color: string }) => (
  <span className="h-2.5 w-2.5 rounded-full border-2" style={{ borderColor: color }} />
);
