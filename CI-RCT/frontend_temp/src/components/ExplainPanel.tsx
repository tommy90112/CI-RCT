/**
 * ExplainPanel — bottom-drawer content implementing the spec's explainability
 * narrative:
 *   L1 結構   = the node-link graph above (GraphCanvas): 金流鏈 + φ_asym 聚光燈
 *   L2 因果責任 = the CE vs φ_asym dual-column bars (追路徑 vs 釘元兇)
 *   L3 根因特徵 = feature attribution of the chain's attributed node (its pivot);
 *               the column is omitted entirely for chains without one
 * Pure presentational: all selection state is lifted to App via onSelectGlobal.
 */
import type { CrimeChain, CrimeChainNode, ResponsibilityRow } from '../types';
import { COLOR, phiColor, phiNorm, shortId, typeGlyph } from '../lib/render';
import { ResponsibilityBars } from './ResponsibilityBars';
import { FeatureAttribution, hasFeatureAttribution } from './FeatureAttribution';
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

  const pivot = rows.find(r => r.is_pivot) ?? null;
  const phiMax = rows.reduce((m, r) => Math.max(m, r.phiAsym != null ? Math.abs(r.phiAsym) : 0), 0);

  // L3 explains the node that actually carries an attribution: the selected
  // node when it has one (dumps made with --feat_attr_all_nodes), otherwise the
  // chain's attributed node (its φ_asym pivot). A chain with no attribution at
  // all simply has no L3 column — L2 takes the full width.
  const l3Node: CrimeChainNode | null = hasFeatureAttribution(selectedNode)
    ? selectedNode
    : chain.nodes.find(hasFeatureAttribution) ?? null;
  const l3IsPivot = l3Node != null && pivot != null && l3Node.global === pivot.global;
  const showL3 = l3Node != null;

  return (
    <div className="flex h-full flex-col overflow-hidden text-ink-200">
      {/* Header — narrative + chain summary */}
      <header className="hairline flex flex-wrap items-center gap-x-5 gap-y-2 border-b px-4 py-2">
        <h2 className="text-[15px] font-semibold tracking-tight text-ink-100">可解釋性面板</h2>

        <ol className="flex items-center gap-1 text-[11.5px]" aria-label="解釋層級">
          <Step n="L1" label="金流鏈" hint="上方圖" />
          <Arrow />
          <Step n="L2" label="因果責任" hint={<>CE 追路徑 / <PhiAsym /> 釘元兇</>} />
          {showL3 && (
            <>
              <Arrow />
              <Step n="L3" label="根因特徵" />
            </>
          )}
        </ol>

        <dl className="ml-auto flex flex-wrap items-center gap-x-4 gap-y-1 font-mono text-[12px]">
          <Stat label="tx" value={shortId(chain.target_txid)} title={chain.target_txid} />
          <Stat label="深度" value={String(chain.depth)} />
          {verdict && (
            <span className={`chip ${isTp ? 'chip-ok' : 'chip-alert'} font-mono`}>{verdict}</span>
          )}
          {pivot && (
            <Stat
              label="pivot"
              value={
                <span style={{ color: COLOR.pivot }}>
                  ★ {typeGlyph(pivot.type)} {shortId(pivot.real_id)} · <PhiAsym /> = {pivot.phiAsym?.toFixed(3)}
                </span>
              }
              title={pivot.real_id}
            />
          )}
        </dl>
      </header>

      {/* Two-column: L2 dual-bars (wide) + L3 feature attribution */}
      <div
        className={`grid min-h-0 flex-1 grid-cols-1 gap-px overflow-hidden bg-white/[0.06] ${
          showL3 ? 'lg:grid-cols-[3fr_2fr]' : ''
        }`}
      >
        <section className="flex min-h-0 flex-col overflow-hidden bg-ink-900/90 px-4 py-2.5">
          <div className="mb-2 flex items-baseline gap-2">
            <span className="eyebrow text-brand">L2 · 因果責任</span>
            <span className="text-[12px] text-ink-300">CE vs <PhiAsym />，逐節點</span>
            <span className="ml-auto font-mono text-[11px] text-ink-500">描述性，非守恆</span>
          </div>

          {/* Path breadcrumb: root → … → target ⚑ */}
          <ChainStrip rows={rows} selectedGlobal={selectedGlobal} onSelect={onSelectGlobal} phiMax={phiMax} />

          <div className="scroll-thin mt-2 min-h-0 flex-1 overflow-auto">
            <ResponsibilityBars rows={rows} selectedGlobal={selectedGlobal} onSelect={onSelectGlobal} />
          </div>
        </section>

        {showL3 && (
          <section className="scroll-thin flex min-h-0 flex-col overflow-auto bg-ink-900/90 px-4 py-2.5">
            <FeatureAttribution node={l3Node} isPivot={l3IsPivot} />
          </section>
        )}
      </div>
    </div>
  );
}

function Step({ n, label, hint }: { n: string; label: string; hint?: React.ReactNode }) {
  return (
    <li className="flex items-center gap-1.5 rounded-md bg-ink-800/70 px-2 py-1 ring-1 ring-white/[0.05]">
      <span className="font-mono font-semibold text-brand">{n}</span>
      <span className="text-ink-200">{label}</span>
      {hint && <span className="text-ink-400">· {hint}</span>}
    </li>
  );
}

function Arrow() {
  return <li aria-hidden className="px-0.5 text-ink-500">→</li>;
}

function Stat({ label, value, title }: { label: string; value: React.ReactNode; title?: string }) {
  return (
    <div className="flex items-baseline gap-1.5" title={title}>
      <dt className="text-ink-400">{label}</dt>
      <dd className="text-ink-100">{value}</dd>
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
    <div className="scroll-thin flex items-center gap-1 overflow-x-auto pb-1">
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
              className={`rounded-md px-2 py-1 font-mono text-[11.5px] ring-1 transition-all duration-200 active:scale-[0.98] ${
                selected
                  ? 'bg-brand/15 text-brand-soft ring-brand/60'
                  : 'bg-ink-800/70 text-ink-200 ring-white/[0.06] hover:bg-ink-700 hover:ring-white/10'
              }`}
              style={selected ? undefined : { boxShadow: `inset 3px 0 0 ${chipColor}` }}
            >
              {row.is_pivot && <span className="mr-1" style={{ color: COLOR.pivot }}>★</span>}
              {row.is_root && !row.is_pivot && <span className="mr-1 text-[10px]" style={{ color: COLOR.root }}>根</span>}
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
  return <span className="px-0.5 font-mono text-[10px] text-ink-500">→</span>;
}
