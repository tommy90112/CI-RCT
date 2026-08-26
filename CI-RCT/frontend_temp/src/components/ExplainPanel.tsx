/**
 * ExplainPanel — bottom-drawer content implementing the spec's explainability
 * narrative:
 *   L1 結構   = the node-link graph above (GraphCanvas): 金流鏈 + φ_asym 聚光燈
 *   L2 因果責任 = the CE vs φ_asym dual-column bars (追路徑 vs 釘元兇)
 *   L3 根因特徵 = per-node feature attribution (deferred export)
 * Pure presentational: all selection state is lifted to App via onSelectGlobal.
 */
import type { CrimeChain, CrimeChainNode, ResponsibilityRow } from '../types';
import { COLOR, phiColor, phiNorm, shortId, typeGlyph } from '../lib/render';
import { ResponsibilityBars } from './ResponsibilityBars';
import { FeatureAttribution } from './FeatureAttribution';
import { PhiAsym } from './Phi';

interface ExplainPanelProps {
  chain: CrimeChain;
  rows: ResponsibilityRow[];
  selectedNode: CrimeChainNode | null;
  selectedGlobal: number | null;
  onSelectGlobal: (global: number) => void;
}

export function ExplainPanel({
  chain,
  rows,
  selectedNode,
  selectedGlobal,
  onSelectGlobal,
}: ExplainPanelProps) {
  const isTp = chain.is_true_positive;
  const verdict = isTp === undefined ? null : isTp ? 'true-positive' : 'false-positive';
  const verdictColor = isTp === false ? COLOR.ceNeg : COLOR.cePos;

  const pivot = rows.find(r => r.is_pivot) ?? null;
  const phiMax = rows.reduce((m, r) => Math.max(m, r.phiAsym != null ? Math.abs(r.phiAsym) : 0), 0);

  return (
    <div className="flex h-full flex-col gap-3 overflow-hidden p-3 text-slate-200">
      {/* Header — narrative + chain summary */}
      <header className="flex flex-wrap items-baseline gap-x-4 gap-y-1">
        <h2 className="text-sm font-semibold text-sky-400">可解釋性面板</h2>
        <span className="text-[11px] text-slate-400">
          L1 金流鏈(上方圖)→ L2 因果責任(CE 追路徑 / <PhiAsym /> 釘元兇)→ L3 根因特徵
        </span>
        <div className="ml-auto flex flex-wrap items-center gap-x-3 gap-y-1 font-mono text-[11px] text-slate-400">
          <span>
            tx <span className="text-slate-200">{shortId(chain.target_txid)}</span>
          </span>
          <span>
            深度 <span className="text-slate-200">{chain.depth}</span>
          </span>
          {verdict && <span style={{ color: verdictColor }}>{verdict}</span>}
          {pivot && (
            <span>
              pivot{' '}
              <span style={{ color: COLOR.pivot }}>
                ★ {typeGlyph(pivot.type)} {shortId(pivot.real_id)} <PhiAsym /> = {pivot.phiAsym?.toFixed(3)}
              </span>
            </span>
          )}
        </div>
      </header>

      {/* Two-column: L2 dual-bars (wide) + L3 feature attribution */}
      <div className="grid min-h-0 flex-1 grid-cols-1 gap-3 overflow-hidden lg:grid-cols-[3fr_2fr]">
        <section className="flex min-h-0 flex-col overflow-hidden rounded-xl border border-slate-700/60 bg-slate-900/85 p-2 shadow-xl backdrop-blur-sm">
          <div className="mb-1 flex items-baseline gap-2">
            <span className="text-[11px] font-semibold text-indigo-300">
              L2 · 因果責任(CE vs <PhiAsym />,逐節點)
            </span>
            <span className="font-mono text-[10px] text-slate-500">描述性,非守恆</span>
          </div>
          <div className="min-h-0 flex-1 overflow-auto">
            <ResponsibilityBars rows={rows} selectedGlobal={selectedGlobal} onSelect={onSelectGlobal} />
          </div>

          {/* Compact chain strip: root → … → target ⚑ */}
          <ChainStrip rows={rows} selectedGlobal={selectedGlobal} onSelect={onSelectGlobal} phiMax={phiMax} />
        </section>

        <section className="flex min-h-0 flex-col overflow-hidden rounded-xl border border-slate-700/60 bg-slate-900/85 p-2 shadow-xl backdrop-blur-sm">
          <div className="mb-1 text-[11px] font-semibold text-indigo-300">L3 · 根因特徵</div>
          <div className="min-h-0 flex-1 overflow-auto">
            <FeatureAttribution node={selectedNode} />
          </div>
        </section>
      </div>
    </div>
  );
}

interface ChainStripProps {
  rows: ResponsibilityRow[];
  selectedGlobal: number | null;
  onSelect: (global: number) => void;
  phiMax: number;
}

/**
 * Horizontal chain of clickable chips upstream→downstream (root → … → target).
 * Chip left-border colour encodes |φ_asym|; the pivot is starred.
 */
function ChainStrip({ rows, selectedGlobal, onSelect, phiMax }: ChainStripProps) {
  return (
    <div className="mt-2 flex items-center gap-1 overflow-x-auto border-t border-slate-700/50 pt-2">
      {rows.map((row, i) => {
        const selected = row.global === selectedGlobal;
        const mag = row.phiAsym != null ? Math.abs(row.phiAsym) : 0;
        const chipColor = phiColor(phiNorm(mag, phiMax));
        const tail = row.is_target ? ' ⚑' : '';
        const label = `${typeGlyph(row.type)} ${shortId(row.real_id)}${tail}`;
        return (
          <div key={row.global} className="flex shrink-0 items-center gap-1">
            {i > 0 && <Connector />}
            <button
              type="button"
              onClick={() => onSelect(row.global)}
              title={`${row.real_id} · CE=${row.ce?.toFixed(4) ?? '—'} · φ_asym=${row.phiAsym?.toFixed(4) ?? '—'}`}
              className={`rounded-md border px-2 py-1 font-mono text-[10px] transition-colors ${
                selected
                  ? 'border-sky-400 bg-sky-400/15 text-sky-200'
                  : 'border-slate-700/60 bg-slate-800/60 text-slate-300 hover:border-slate-500'
              }`}
              style={selected ? undefined : { borderLeft: `3px solid ${chipColor}` }}
            >
              {row.is_pivot && <span className="mr-1" style={{ color: COLOR.pivot }}>★</span>}
              {row.is_root && !row.is_pivot && <span className="mr-1 text-[9px] text-violet-300">根</span>}
              {label}
            </button>
          </div>
        );
      })}
    </div>
  );
}

/** Directional connector (upstream→downstream) between two chain chips. */
function Connector() {
  return <span className="px-0.5 font-mono text-[9px] text-slate-600">→</span>;
}
