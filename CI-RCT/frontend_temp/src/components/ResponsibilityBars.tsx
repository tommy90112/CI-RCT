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

const PHI_NEG_COLOR = '#7c5cff';

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
      <div className="relative h-3 flex-1 rounded-sm bg-white/[0.03]">
        <div className="absolute inset-y-0 left-1/2 w-px bg-ink-500/70" />
        <div className="flex h-full items-center justify-center text-[9px] text-ink-500">n/a</div>
      </div>
    );
  }
  const positive = value >= 0;
  const widthPct = (max > 0 ? Math.min(1, Math.abs(value) / max) : 0) * 50;
  const color = positive ? posColor : negColor;
  return (
    <div className="relative h-3 flex-1 rounded-sm bg-white/[0.03]">
      <div className="absolute inset-y-0 left-1/2 w-px bg-ink-500/70" />
      <div
        className="absolute bottom-0 top-0 transition-[width] duration-300"
        style={
          positive
            ? { left: '50%', width: `${widthPct}%`, backgroundColor: color, borderRadius: '0 3px 3px 0' }
            : { right: '50%', width: `${widthPct}%`, backgroundColor: color, borderRadius: '3px 0 0 3px' }
        }
      />
    </div>
  );
}

export function ResponsibilityBars({ rows, selectedGlobal, onSelect }: ResponsibilityBarsProps) {
  if (rows.length === 0) {
    return <div className="py-4 text-center font-mono text-[11px] text-ink-400">無責任鏈資料</div>;
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
        <div
          className="mb-2 flex items-center gap-2 rounded-lg px-2.5 py-1.5 text-[11px] ring-1 ring-amber-400/25"
          style={{ background: 'linear-gradient(90deg, rgba(251,191,36,0.10), rgba(251,191,36,0.03))' }}
        >
          <span className="font-semibold" style={{ color: COLOR.pivot }}>★ 元兇 pivot</span>
          <span className="font-mono text-ink-100">{typeGlyph(pivot.type)} {shortId(pivot.real_id)}</span>
          <span className="font-mono" style={{ color: COLOR.pivot }}><PhiAsym /> = {fmt(pivot.phiAsym)}</span>
          <span className="ml-auto text-ink-300">
            佔 <span className="font-mono font-semibold text-amber-200">{pivotShare}%</span> 責任
          </span>
        </div>
      )}

      {!hasAnyPhi && (
        <div className="notice-warn mb-2">
          此鏈為 wallet-target，<PhiAsym /> 不適用（僅顯示 CE）。
        </div>
      )}

      {/* column header */}
      <div className="mb-1 flex items-center gap-2 px-1.5 text-[10px] font-medium text-ink-400">
        <div className="w-36 shrink-0">節點（上游 → 下游）</div>
        <div className="flex-1 text-center">
          <span className="text-ink-200">CE</span><span className="ml-1">追路徑</span>
        </div>
        <div className="w-14 shrink-0" />
        <div className="flex-1 text-center">
          <span className="text-ink-200"><PhiAsym /></span><span className="ml-1">釘元兇</span>
        </div>
        <div className="w-14 shrink-0" />
      </div>

      <div className="flex flex-col gap-0.5">
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
              role="button"
              tabIndex={0}
              onClick={() => onSelect(row.global)}
              onKeyDown={e => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); onSelect(row.global); } }}
              title={`${row.real_id}\nCE=${fmt(row.ce)} · φ_asym=${fmt(row.phiAsym)}（點擊聚焦上方圖）`}
              className={`flex cursor-pointer items-center gap-2 rounded-md px-1.5 py-1 ring-1 transition-colors duration-150 ${
                selected
                  ? 'bg-brand/10 ring-brand/50'
                  : row.is_pivot
                    ? 'bg-amber-400/[0.04] ring-transparent hover:bg-white/[0.04]'
                    : 'ring-transparent hover:bg-white/[0.04]'
              }`}
            >
              {/* label */}
              <div className="flex w-36 shrink-0 items-center gap-1.5 truncate font-mono">
                {row.is_pivot && <span style={{ color: COLOR.pivot }}>★</span>}
                {row.is_root && !row.is_pivot && <span style={{ color: COLOR.root }}>◎</span>}
                <span style={{ color: row.type === 'transaction' ? COLOR.tx : COLOR.wallet }}>{typeGlyph(row.type)}</span>
                <span className={`truncate ${selected ? 'text-ink-100' : 'text-ink-300'}`}>{shortId(row.real_id)}</span>
                {row.is_target && <span style={{ color: COLOR.fraud }}>⚑</span>}
              </div>

              {/* CE bar */}
              <DivergingBar value={row.ce} max={ceMax} posColor={COLOR.cePos} negColor={COLOR.ceNeg} />
              <div className="w-14 shrink-0 text-right font-mono text-[10.5px] text-ink-300">{fmt(row.ce)}</div>

              {/* φ_asym bar */}
              <DivergingBar value={row.phiAsym} max={phiMax} posColor={COLOR.pivot} negColor={PHI_NEG_COLOR} />
              <div
                className="w-14 shrink-0 text-right font-mono text-[10.5px]"
                style={{ color: row.is_pivot ? COLOR.pivot : '#98a1b3' }}
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
