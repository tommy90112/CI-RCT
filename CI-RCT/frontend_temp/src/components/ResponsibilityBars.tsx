/**
 * L2 · 因果責任(CE vs φ_asym 雙欄長條,spec §7).
 *
 * Two diverging-bar columns per chain node, ordered upstream→downstream
 * (root → target) to match the L1 x-axis:
 *   - CE       : signed edge causal effect — 追路徑(分散在多條邊)
 *   - φ_asym   : signed asymmetric Shapley — 釘元兇(塌縮到單一 pivot)
 *
 * DESCRIPTIVE, not cumulative — no efficiency/Σ total is claimed. Each column is
 * scaled by its own max |value| so both stay readable despite different ranges.
 * The pivot row (peak |φ_asym|) is starred; root/target are tagged.
 */
import type { ResponsibilityRow } from '../types';
import { COLOR, shortId, typeGlyph } from '../lib/render';
import { PhiAsym } from './Phi';

interface ResponsibilityBarsProps {
  rows: ResponsibilityRow[];
  selectedGlobal: number | null;
  onSelect: (global: number) => void;
}

const fmt = (v: number | null): string =>
  v == null ? '—' : `${v >= 0 ? '+' : ''}${v.toFixed(3)}`;

/** One diverging bar in a centre-zero track; positive = right/warm, negative = left/cool. */
function DivergingBar({
  value,
  max,
  posColor,
  negColor,
}: {
  value: number | null;
  max: number;
  posColor: string;
  negColor: string;
}) {
  if (value == null) {
    return (
      <div className="relative h-3.5 flex-1">
        <div className="absolute inset-y-0 left-1/2 w-px bg-slate-600/70" />
        <div className="flex h-full items-center justify-center text-[9px] text-slate-600">n/a</div>
      </div>
    );
  }
  const positive = value >= 0;
  const widthPct = (max > 0 ? Math.min(1, Math.abs(value) / max) : 0) * 50;
  const color = positive ? posColor : negColor;
  return (
    <div className="relative h-3.5 flex-1">
      <div className="absolute inset-y-0 left-1/2 w-px bg-slate-600/70" />
      <div
        className="absolute top-0 bottom-0 rounded-sm"
        style={
          positive
            ? { left: '50%', width: `${widthPct}%`, backgroundColor: color }
            : { right: '50%', width: `${widthPct}%`, backgroundColor: color }
        }
      />
    </div>
  );
}

export function ResponsibilityBars({ rows, selectedGlobal, onSelect }: ResponsibilityBarsProps) {
  if (rows.length === 0) {
    return <div className="py-4 text-center font-mono text-[11px] text-slate-500">無責任鏈資料</div>;
  }

  const ceMax = rows.reduce((m, r) => Math.max(m, r.ce != null ? Math.abs(r.ce) : 0), 0);
  const phiMax = rows.reduce((m, r) => Math.max(m, r.phiAsym != null ? Math.abs(r.phiAsym) : 0), 0);
  const hasAnyPhi = rows.some(r => r.phiAsym != null);

  // Pivot summary so the punchline reads even when a long chain scrolls.
  const pivot = rows.find(r => r.is_pivot) ?? null;
  const totalAbs = rows.reduce((s, r) => s + (r.phiAsym != null ? Math.abs(r.phiAsym) : 0), 0);
  const pivotShare = pivot && pivot.phiAsym != null && totalAbs > 0
    ? Math.round((Math.abs(pivot.phiAsym) / totalAbs) * 100)
    : 0;

  return (
    <div className="w-full text-[11px]">
      {/* pivot 摘要 — 一眼看到元兇 */}
      {pivot && (
        <div className="mb-1.5 flex items-center gap-2 rounded-md border border-amber-500/30 bg-amber-500/10 px-2 py-1 text-[11px]">
          <span style={{ color: COLOR.pivot }}>★ 元兇 pivot</span>
          <span className="font-mono text-slate-200">{typeGlyph(pivot.type)} {shortId(pivot.real_id)}</span>
          <span className="font-mono" style={{ color: COLOR.pivot }}><PhiAsym /> = {fmt(pivot.phiAsym)}</span>
          <span className="ml-auto text-slate-400">佔 <span className="font-mono text-amber-300">{pivotShare}%</span> 責任</span>
        </div>
      )}

      {/* column header */}
      <div className="mb-1 flex items-center gap-2 px-1 text-[10px] font-medium text-slate-400">
        <div className="w-32 shrink-0" />
        <div className="flex-1 text-center">
          CE<span className="ml-1 text-slate-500">追路徑</span>
        </div>
        <div className="w-14 shrink-0" />
        <div className="flex-1 text-center">
          <PhiAsym /><span className="ml-1 text-slate-500">釘元兇</span>
        </div>
        <div className="w-14 shrink-0" />
      </div>

      {!hasAnyPhi && (
        <div className="mb-1.5 rounded-md border border-amber-500/30 bg-amber-500/10 px-2 py-1 text-[10px] text-amber-300">
          此鏈為 wallet-target,<PhiAsym /> 不適用(僅顯示 CE)。
        </div>
      )}

      <div className="flex flex-col gap-1">
        {rows.map(row => {
          const selected = row.global === selectedGlobal;
          const tags = [
            row.is_target ? '⚑' : '',
            row.is_root ? '◎' : '',
            row.is_pivot ? '★' : '',
          ].join('');
          return (
            <div
              key={row.global}
              onClick={() => onSelect(row.global)}
              title={`${row.real_id}\nCE=${fmt(row.ce)} · φ_asym=${fmt(row.phiAsym)}（點擊聚焦上方圖）`}
              className={`flex cursor-pointer items-center gap-2 rounded-md px-1 py-0.5 transition-colors ${
                selected ? 'bg-sky-400/15 ring-1 ring-sky-400/60' : row.is_pivot ? 'bg-amber-400/5' : 'hover:bg-slate-800/60'
              }`}
            >
              {/* label */}
              <div className="flex w-32 shrink-0 items-center gap-1 truncate font-mono">
                {row.is_pivot && <span className="text-amber-400" style={{ color: COLOR.pivot }}>★</span>}
                {row.is_root && !row.is_pivot && <span style={{ color: COLOR.root }}>◎</span>}
                <span className="text-slate-300">{typeGlyph(row.type)}</span>
                <span className="truncate text-slate-400">{shortId(row.real_id)}</span>
                {row.is_target && <span style={{ color: COLOR.fraud }}>⚑</span>}
              </div>

              {/* CE bar */}
              <DivergingBar value={row.ce} max={ceMax} posColor={COLOR.cePos} negColor={COLOR.ceNeg} />
              <div className="w-14 shrink-0 text-right font-mono text-[10px] text-slate-400">{fmt(row.ce)}</div>

              {/* φ_asym bar */}
              <DivergingBar
                value={row.phiAsym}
                max={phiMax}
                posColor={COLOR.pivot}
                negColor="#7c5cff"
              />
              <div
                className="w-14 shrink-0 text-right font-mono text-[10px]"
                style={{ color: row.is_pivot ? COLOR.pivot : '#94a3b8' }}
              >
                {fmt(row.phiAsym)}
              </div>
              {/* tags column intentionally folded into label/value for compactness */}
              <span className="sr-only">{tags}</span>
            </div>
          );
        })}
      </div>
    </div>
  );
}
