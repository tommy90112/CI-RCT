/**
 * TopBar — the app bar that stays visible whatever the drawers are doing:
 *   brand · dataset variant · current-chain stepper · status chips · panel toggles
 * The chain stepper walks the same filtered list as the control panel picker
 * (shared via useChainFilter) so the two never disagree.
 */
import type { CrimeChain, DataVariant } from '../types';
import type { ChainFilter } from '../hooks/useChainFilter';
import { COLOR, shortId, typeGlyph } from '../lib/render';
import { Segmented } from './ui/Segmented';
import { IDEA_HREF } from '../lib/route';
import { PhiAsym } from './Phi';

export const TOP_BAR_HEIGHT = 56;

const VARIANTS: { value: DataVariant; label: string; title: string }[] = [
  { value: 'joint', label: 'joint', title: '主結果：交易 + 錢包聯合偵測頭' },
  { value: 'transaction', label: 'transaction', title: '單任務：交易偵測頭' },
  { value: 'wallet', label: 'wallet', title: '單任務：錢包偵測頭' },
];

interface TopBarProps {
  variant: DataVariant;
  onVariantChange: (v: DataVariant) => void;
  chain: CrimeChain;
  selectedIdx: number;
  filter: ChainFilter;
  totalChains: number;
  convergeCount: number;
  onExitConverge: () => void;
  datasetHasPhi: boolean;
  chainHasPhi: boolean;
  leftOpen: boolean;
  bottomOpen: boolean;
  onToggleLeft: () => void;
  onToggleBottom: () => void;
}

export function TopBar(props: TopBarProps) {
  const {
    variant, onVariantChange, chain, selectedIdx, filter, totalChains,
    convergeCount, onExitConverge, datasetHasPhi, chainHasPhi,
    leftOpen, bottomOpen, onToggleLeft, onToggleBottom,
  } = props;

  const pos = filter.posInMatches;
  const positionLabel = pos >= 0 ? `${pos + 1} / ${filter.matches.length}` : `— / ${filter.matches.length}`;

  return (
    <header
      className="panel hairline absolute inset-x-0 top-0 z-30 flex items-center gap-3 border-b px-4"
      style={{ height: TOP_BAR_HEIGHT }}
    >
      {/* Brand — links back to the idea page */}
      <a
        href={IDEA_HREF}
        title="詐欺資金鏈可解釋視覺化 · 回到想法頁"
        className="group flex shrink-0 items-center gap-2.5 rounded-lg pr-1 transition-colors hover:bg-white/[0.04]"
      >
        <Mark />
        <div className="leading-none">
          <div className="text-[15px] font-bold tracking-tight text-ink-100">CI-RCT</div>
          <div className="mt-1 text-[11.5px] font-medium tracking-[0.06em] text-ink-400 group-hover:text-brand">← 想法頁</div>
        </div>
      </a>

      <span className="hairline h-6 border-l" aria-hidden />

      {/* Dataset variant */}
      <div className="flex shrink-0 items-center gap-2">
        <span className="eyebrow hidden xl:inline">變體</span>
        <Segmented
          ariaLabel="資料變體"
          value={variant}
          onChange={onVariantChange}
          options={VARIANTS.map(v => ({
            value: v.value,
            title: v.title,
            label: v.value === 'joint' ? <span>joint<span className="ml-1 text-brand">·主</span></span> : v.label,
          }))}
        />
      </div>

      {/* Current chain stepper (centre) */}
      <div className="flex min-w-0 flex-1 items-center justify-center">
        <div className="flex min-w-0 items-center gap-1 rounded-xl bg-ink-800/70 p-1 ring-1 ring-white/[0.05]">
          <button
            type="button"
            onClick={() => filter.step(-1)}
            disabled={filter.matches.length === 0}
            aria-label="上一條詐欺鏈"
            className="btn-icon h-8 w-8 rounded-lg bg-transparent text-[16px] ring-0"
          >
            ‹
          </button>
          <ChainSummary chain={chain} idx={selectedIdx} />
          <button
            type="button"
            onClick={() => filter.step(1)}
            disabled={filter.matches.length === 0}
            aria-label="下一條詐欺鏈"
            className="btn-icon h-8 w-8 rounded-lg bg-transparent text-[16px] ring-0"
          >
            ›
          </button>
          <span
            className="hidden shrink-0 pl-1 pr-2 font-mono text-[12px] text-ink-400 lg:inline"
            title={`目前位置 / 符合篩選的鏈數（資料共 ${totalChains} 條）`}
          >
            {positionLabel}
          </span>
        </div>
      </div>

      {/* Status chips */}
      <div className="flex shrink-0 items-center gap-2">
        {convergeCount > 0 && (
          <button
            type="button"
            onClick={onExitConverge}
            className="chip chip-alert transition hover:bg-rose-400/20"
            title="退出收斂視圖"
          >
            收斂視圖 · {convergeCount} 條 → 源頭
            <span aria-hidden>✕</span>
          </button>
        )}
        {!datasetHasPhi ? (
          <span className="chip chip-warn">此資料無 <PhiAsym /></span>
        ) : !chainHasPhi ? (
          <span className="chip chip-warn">wallet-target · 無 <PhiAsym />，僅 CE</span>
        ) : (
          <span className="chip chip-ok">
            <span className="chip-dot bg-emerald-400" />
            <PhiAsym /> pivot 已標示
          </span>
        )}
      </div>

      <span className="hairline h-6 border-l" aria-hidden />

      {/* Panel toggles */}
      <div className="flex shrink-0 items-center gap-1">
        <PanelToggle label="控制台" active={leftOpen} onClick={onToggleLeft} kind="left" />
        <PanelToggle label="解釋面板" active={bottomOpen} onClick={onToggleBottom} kind="bottom" />
      </div>
    </header>
  );
}

/** Brand mark: wallet dot + transaction square joined by a causal edge. */
function Mark() {
  return (
    <svg width="30" height="30" viewBox="0 0 32 32" aria-hidden className="shrink-0">
      <rect width="32" height="32" rx="8" fill="#181c24" />
      <rect width="32" height="32" rx="8" fill="none" stroke="rgba(255,255,255,0.06)" />
      <circle cx="11" cy="21" r="4.5" fill={COLOR.wallet} />
      <rect x="17" y="7" width="9" height="9" rx="1.5" fill={COLOR.fraud} />
      <path d="M14.5 18 18 15.5" stroke="#98a1b3" strokeWidth="1.6" strokeLinecap="round" />
    </svg>
  );
}

function ChainSummary({ chain, idx }: { chain: CrimeChain; idx: number }) {
  const rootFraud = chain.root_is_fraud;
  const tp = chain.is_true_positive;
  return (
    <div
      className="flex min-w-0 items-center gap-2 px-2 font-mono text-[12.5px] text-ink-200"
      title={`target ${chain.target_txid}\nroot ${chain.root_real_id}`}
    >
      <span className="shrink-0 text-ink-400">#{idx}</span>
      <span className="shrink-0 whitespace-nowrap">
        <span className="text-ink-400">◆ </span>
        {shortId(chain.target_txid)}
      </span>
      <span className="text-ink-500">·</span>
      <span className="whitespace-nowrap">
        <span className="text-ink-400">深度 </span>
        {chain.depth}
      </span>
      <span className="hidden text-ink-500 xl:inline">·</span>
      <span className="hidden min-w-0 truncate whitespace-nowrap xl:inline">
        <span className="text-ink-400">root </span>
        {typeGlyph(chain.root_type)} {chain.root_type}
        {rootFraud && <span className="ml-1 text-rose-300">✓詐欺</span>}
      </span>
      {tp !== undefined && (
        <span
          className={`rounded px-1.5 py-px text-[11px] font-semibold tracking-wider ${
            tp ? 'bg-emerald-400/15 text-emerald-300' : 'bg-rose-400/15 text-rose-300'
          }`}
          title={tp ? 'true-positive：目標確為非法' : 'false-positive：目標實為合法'}
        >
          {tp ? 'TP' : 'FP'}
        </span>
      )}
    </div>
  );
}

function PanelToggle({
  label, active, onClick, kind,
}: { label: string; active: boolean; onClick: () => void; kind: 'left' | 'bottom' }) {
  return (
    <button
      type="button"
      onClick={onClick}
      aria-pressed={active}
      title={`${active ? '收合' : '展開'}${label}`}
      className={`btn-ghost ${active ? 'bg-ink-600 text-ink-100 ring-white/10' : ''}`}
    >
      <svg width="12" height="12" viewBox="0 0 12 12" aria-hidden>
        <rect x="0.75" y="0.75" width="10.5" height="10.5" rx="2" fill="none" stroke="currentColor" strokeWidth="1.3" />
        {kind === 'left' ? (
          <rect x="0.75" y="0.75" width="4" height="10.5" rx="1.5" fill={active ? '#4fd1c5' : 'currentColor'} opacity={active ? 1 : 0.5} />
        ) : (
          <rect x="0.75" y="7.25" width="10.5" height="4" rx="1.5" fill={active ? '#4fd1c5' : 'currentColor'} opacity={active ? 1 : 0.5} />
        )}
      </svg>
      {label}
    </button>
  );
}
