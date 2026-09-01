import type { CrimeChainNode, FeatureContribution } from '../types';
import { COLOR, shortId, typeGlyph } from '../lib/render';

// L3 — root-cause feature attribution (spec §5)

interface FeatureAttributionProps {
  node: CrimeChainNode | null;
  /** True when `node` is the chain's φ_asym pivot (the 元兇 the L3 drill-down explains). */
  isPivot?: boolean;
}

const POS_COLOR = '#ff7849'; // warm — positive contribution
const NEG_COLOR = '#3b82f6'; // cool — negative contribution
const MAX_ROWS = 10;

const SUP: Record<string, string> = {
  '-': '⁻', '0': '⁰', '1': '¹', '2': '²', '3': '³', '4': '⁴',
  '5': '⁵', '6': '⁶', '7': '⁷', '8': '⁸', '9': '⁹',
};
const toSuperscript = (n: number): string =>
  String(n).split('').map((c) => SUP[c] ?? c).join('');

interface Scale {
  factor: number; // multiply values by this before display
  unit: string; // e.g. "×10⁻⁴" (empty when values are already readable)
  decimals: number;
}

/**
 * Pick a shared display scale from the largest |contribution|. CFE feature
 * effects are often tiny (~1e-4), so toFixed(3) collapses distinct values to
 * "0.000/±0.001" — mismatching the (correct) bar lengths. We rescale to a common
 * power of ten and show it as a unit so the numbers read clearly and track the bars.
 */
function chooseScale(maxAbs: number): Scale {
  if (!(maxAbs > 0)) return { factor: 1, unit: '', decimals: 3 };
  const exp = Math.floor(Math.log10(maxAbs));
  if (exp >= -2 && exp <= 1) {
    return { factor: 1, unit: '', decimals: Math.min(4, Math.max(2, 2 - exp)) };
  }
  return { factor: 10 ** -exp, unit: `×10${toSuperscript(exp)}`, decimals: 2 };
}

function sortByAbsDesc(items: FeatureContribution[]): FeatureContribution[] {
  // immutable: copy before sort
  return [...items].sort((a, b) => Math.abs(b.value) - Math.abs(a.value));
}

const isAnonName = (name: string): boolean =>
  name.startsWith('Local_feature_') || name.startsWith('Aggregate_feature_');

function Header({ node, isPivot }: { node: CrimeChainNode; isPivot: boolean }) {
  const causal = node.feature_attribution_method !== 'saliency_fallback';
  return (
    <div className="mb-2 flex flex-wrap items-center gap-2">
      <span className="eyebrow text-brand">L3 · 根因特徵歸因</span>
      <span className="font-mono text-[11px] text-ink-100">
        {isPivot && <span className="mr-1" style={{ color: COLOR.pivot }}>★</span>}
        <span style={{ color: node.type === 'transaction' ? COLOR.tx : COLOR.wallet }}>{typeGlyph(node.type)}</span>{' '}
        {shortId(node.real_id)}
      </span>
      <span className="text-[10.5px] text-ink-400">{isPivot ? `${node.type} · 元兇 pivot` : node.type}</span>
      <span
        className={`chip ml-auto ${causal ? 'chip-ok' : 'chip-warn'}`}
        title={causal ? '逐特徵 do-intervention 因果效應 (CFE)' : '因果效應過小，改用 saliency 相關性排序'}
      >
        {causal ? '因果 do' : 'saliency 備案'}
      </span>
    </div>
  );
}

function BarRow({ item, maxAbs, scale }: { item: FeatureContribution; maxAbs: number; scale: Scale }) {
  const positive = item.value >= 0;
  const color = positive ? POS_COLOR : NEG_COLOR;
  const frac = maxAbs > 0 ? Math.abs(item.value) / maxAbs : 0;
  // each side gets up to 50% of the track width
  const widthPct = frac * 50;
  const shown = (item.value * scale.factor).toFixed(scale.decimals);
  // Tooltip keeps the exact unscaled value for verification.
  const exact = item.value.toExponential(3);
  const title =
    item.raw !== undefined
      ? `${item.name} = ${item.raw}（貢獻 ${exact}）`
      : `${item.name}（貢獻 ${exact}）`;

  return (
    <div className="flex items-center gap-2 rounded-md px-1 py-0.5 transition-colors hover:bg-white/[0.04]" title={title}>
      <div className="w-32 shrink-0 truncate text-right font-mono text-[10.5px] text-ink-300" title={item.name}>
        {item.name}
      </div>
      <div className="relative h-3 flex-1 rounded-sm bg-white/[0.03]">
        {/* zero axis */}
        <div className="absolute inset-y-0 left-1/2 w-px bg-ink-500/70" />
        {/* bar */}
        <div
          className="absolute bottom-0 top-0 transition-[width] duration-300"
          style={
            positive
              ? { left: '50%', width: `${widthPct}%`, backgroundColor: color, borderRadius: '0 3px 3px 0' }
              : { right: '50%', width: `${widthPct}%`, backgroundColor: color, borderRadius: '3px 0 0 3px' }
          }
        />
      </div>
      <div className="w-16 shrink-0 text-right font-mono text-[10.5px]" style={{ color }}>
        {positive ? '+' : ''}
        {shown}
      </div>
    </div>
  );
}

/** True iff the node carries an L3 attribution worth rendering. */
export const hasFeatureAttribution = (node: CrimeChainNode | null | undefined): node is CrimeChainNode =>
  !!node && Array.isArray(node.feature_attribution) && node.feature_attribution.length > 0;

/**
 * Renders the per-feature causal bars for `node`. Returns null when the node
 * carries no attribution — the caller decides what (if anything) to show
 * instead; there is no in-panel placeholder.
 */
export function FeatureAttribution({ node, isPivot = false }: FeatureAttributionProps) {
  if (!hasFeatureAttribution(node)) return null;
  const attribution = node.feature_attribution as FeatureContribution[];

  const sorted = sortByAbsDesc(attribution).slice(0, MAX_ROWS);
  const maxAbs = sorted.reduce((acc, it) => Math.max(acc, Math.abs(it.value)), 0);
  const scale = chooseScale(maxAbs);
  const anyAnon = sorted.some((it) => isAnonName(it.name));

  return (
    <div className="flex flex-col">
      <Header node={node} isPivot={isPivot} />
      {anyAnon && (
        <p className="notice-warn mb-2 text-[10.5px]">
          ⚠ 此節點部分特徵為匿名（Elliptic++ 交易 <span className="font-mono">Local_feature_*</span>），無語意；錢包特徵則具名可讀。
        </p>
      )}
      <div className="mb-1.5 flex items-center justify-between text-[10px] text-ink-400">
        <span>對詐欺機率的因果貢獻（do-intervention）</span>
        {scale.unit && <span className="font-mono">數值單位 {scale.unit}</span>}
      </div>
      <div className="flex flex-col gap-0.5">
        {sorted.map((item) => (
          <BarRow key={item.name} item={item} maxAbs={maxAbs} scale={scale} />
        ))}
      </div>
      <div className="mt-2.5 flex items-center gap-4 text-[10px] text-ink-400">
        <span className="flex items-center gap-1.5">
          <span className="inline-block h-2 w-2 rounded-sm" style={{ backgroundColor: POS_COLOR }} />
          正貢獻（提升詐欺責任）
        </span>
        <span className="flex items-center gap-1.5">
          <span className="inline-block h-2 w-2 rounded-sm" style={{ backgroundColor: NEG_COLOR }} />
          負貢獻（降低）
        </span>
      </div>
    </div>
  );
}
