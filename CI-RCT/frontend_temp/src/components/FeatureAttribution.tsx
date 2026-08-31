import type { CrimeChainNode, FeatureContribution } from '../types';
import { shortId, typeGlyph } from '../lib/render';

// L3 — root-cause feature attribution (spec §5)

interface FeatureAttributionProps {
  node: CrimeChainNode | null;
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

function Header({ node }: { node: CrimeChainNode }) {
  const causal = node.feature_attribution_method !== 'saliency_fallback';
  return (
    <div className="flex items-center gap-2 mb-3">
      <span className="text-sky-400 font-semibold text-xs">L3 · 根因特徵歸因</span>
      <span className="text-slate-500 text-[11px]">|</span>
      <span className="font-mono text-[11px] text-slate-300">
        {typeGlyph(node.type)} {shortId(node.real_id)}
      </span>
      <span className="text-[11px] text-slate-500">{node.type}</span>
      <span
        className={`ml-auto rounded-md border px-1.5 py-0.5 text-[10px] ${
          causal
            ? 'border-emerald-500/50 bg-emerald-500/10 text-emerald-300'
            : 'border-amber-500/50 bg-amber-500/10 text-amber-300'
        }`}
        title={causal ? '逐特徵 do-intervention 因果效應 (CFE)' : '因果效應過小,改用 saliency 相關性排序'}
      >
        {causal ? '因果 do' : 'saliency 備案'}
      </span>
    </div>
  );
}

function Placeholder({ node }: { node: CrimeChainNode | null }) {
  return (
    <div className="rounded-xl border border-slate-700/60 bg-slate-900/85 p-4 shadow-xl backdrop-blur-sm">
      {node ? (
        <Header node={node} />
      ) : (
        <div className="mb-3 text-xs font-semibold text-sky-400">L3 · 根因特徵歸因</div>
      )}
      <p className="text-xs leading-relaxed text-slate-400">
        此節點非 φ 樞紐(pivot)。L3 因果特徵歸因僅對每條鏈的 pivot 節點計算
        (省算力),請點選標有 <span style={{ color: '#fbbf24' }}>★</span> 的 pivot 節點查看其特徵貢獻。
      </p>
      <p className="mt-2 text-[11px] leading-relaxed text-slate-500">
        註:L3 解釋「節點為何危險」(特徵層),與 φ(L1/L2 的節點責任分解)互補但不同。
      </p>
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
    <div className="flex items-center gap-2" title={title}>
      <div className="w-28 shrink-0 truncate text-right text-[11px] text-slate-300" title={item.name}>
        {item.name}
      </div>
      <div className="relative h-3.5 flex-1">
        {/* zero axis */}
        <div className="absolute inset-y-0 left-1/2 w-px bg-slate-600/70" />
        {/* bar */}
        <div
          className="absolute top-0 bottom-0 rounded-sm"
          style={
            positive
              ? { left: '50%', width: `${widthPct}%`, backgroundColor: color }
              : { right: '50%', width: `${widthPct}%`, backgroundColor: color }
          }
        />
      </div>
      <div className="w-20 shrink-0 text-right font-mono text-[11px]" style={{ color }}>
        {positive ? '+' : ''}
        {shown}
      </div>
    </div>
  );
}

export function FeatureAttribution({ node }: FeatureAttributionProps) {
  const attribution = node?.feature_attribution;

  if (!node || !attribution || attribution.length === 0) {
    return <Placeholder node={node} />;
  }

  const sorted = sortByAbsDesc(attribution).slice(0, MAX_ROWS);
  const maxAbs = sorted.reduce((acc, it) => Math.max(acc, Math.abs(it.value)), 0);
  const scale = chooseScale(maxAbs);
  const anyAnon = sorted.some((it) => isAnonName(it.name));

  return (
    <div className="rounded-xl border border-slate-700/60 bg-slate-900/85 p-4 shadow-xl backdrop-blur-sm">
      <Header node={node} />
      {anyAnon && (
        <p className="mb-2 rounded-md border border-amber-500/30 bg-amber-500/10 px-2 py-1 text-[10px] leading-relaxed text-amber-300">
          ⚠ 此節點部分特徵為匿名(Elliptic++ 交易 <span className="font-mono">Local_feature_*</span>),無語意;錢包特徵則具名可讀。
        </p>
      )}
      <div className="mb-1.5 flex items-center justify-between text-[10px] text-slate-500">
        <span>對詐欺機率的因果貢獻(do-intervention)</span>
        {scale.unit && <span className="font-mono">數值單位 {scale.unit}</span>}
      </div>
      <div className="flex flex-col gap-1.5">
        {sorted.map((item) => (
          <BarRow key={item.name} item={item} maxAbs={maxAbs} scale={scale} />
        ))}
      </div>
      <div className="mt-3 flex items-center gap-4 text-[10px] text-slate-500">
        <span className="flex items-center gap-1">
          <span className="inline-block h-2 w-2 rounded-sm" style={{ backgroundColor: POS_COLOR }} />
          正貢獻(提升詐欺責任)
        </span>
        <span className="flex items-center gap-1">
          <span className="inline-block h-2 w-2 rounded-sm" style={{ backgroundColor: NEG_COLOR }} />
          負貢獻(降低)
        </span>
      </div>
    </div>
  );
}
