import { useRef, type ChangeEvent } from 'react'
import type { CrimeChainData, DisplayOptions, NeighborSource, NodeType } from '../types'
import type { RootGroup } from '../lib/graph'
import type { ChainFilter } from '../hooks/useChainFilter'
import { MAX_OPTIONS } from '../hooks/useChainFilter'
import { COLOR, shortId, typeGlyph } from '../lib/render'
import { PhiAsym } from './Phi'
import { Switch } from './ui/Switch'
import { Section, Divider } from './ui/Section'
import { Segmented } from './ui/Segmented'

// Left-drawer content: chain picker + node-display controls (控制台).
interface ControlPanelProps {
  data: CrimeChainData
  display: DisplayOptions
  onDisplayChange: (next: DisplayOptions) => void
  selectedIdx: number
  onSelectedIdxChange: (i: number) => void
  filter: ChainFilter
  rootGroups: RootGroup[]
  convergeRootId: string | null
  onConvergeChange: (rootRealId: string | null) => void
  dataError: string | null
  visible: { nodes: number; links: number }
  neighborSource: NeighborSource
  source: string
  onLoadFile: (file: File) => void
}

const TYPE_FILTERS: { value: NodeType; label: string; glyph: string; color: string }[] = [
  { value: 'transaction', label: '交易 transaction', glyph: '◆', color: COLOR.tx },
  { value: 'wallet', label: '錢包 wallet', glyph: '●', color: COLOR.wallet },
]

const NEIGHBOR_SOURCE_LABEL: Record<NeighborSource, string> = {
  full: '完整 Elliptic++ 圖',
  union: '鏈聯集（未載入 chain_neighbors.json）',
  none: '—',
}

export function ControlPanel(props: ControlPanelProps) {
  const {
    data, display, onDisplayChange, selectedIdx, onSelectedIdxChange, filter,
    rootGroups, convergeRootId, onConvergeChange, dataError, visible,
    neighborSource, source, onLoadFile,
  } = props

  const fileRef = useRef<HTMLInputElement | null>(null)
  const { query, setQuery, onlyPhi, toggleOnlyPhi, matches, shown, nWithPhi, phiPct } = filter

  const setLayout = (temporal: boolean) =>
    onDisplayChange({ ...display, layout: temporal ? 'temporal' : 'structure' })

  const setDimensions = (d: 2 | 3) => onDisplayChange({ ...display, dimensions: d })

  const setField = (field: 'showAll' | 'neighbors' | 'particles' | 'sizeByPhi', value: boolean) =>
    onDisplayChange({ ...display, [field]: value })

  const toggleType = (t: NodeType) => {
    const next = new Set(display.typeFilter)
    if (next.has(t)) next.delete(t)
    else next.add(t)
    onDisplayChange({ ...display, typeFilter: next })
  }

  const onFile = (e: ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (file) onLoadFile(file)
    e.target.value = ''
  }

  return (
    <div className="flex flex-col gap-5 text-[12px] text-ink-200">
      {/* Title */}
      <div>
        <h2 className="text-[13px] font-semibold tracking-tight text-ink-100">控制台</h2>
        <p className="mt-0.5 text-[11px] text-ink-400">選鏈、切換視圖、調整圖形呈現</p>
      </div>

      {dataError && <p className="notice-warn">{dataError}</p>}

      {/* 鏈選擇 */}
      <Section
        title="詐欺鏈 fraud chain"
        meta={`共 ${data.chains.length} 條${query.trim() ? ` · 符合 ${matches.length}` : ''}`}
      >
        <div className="relative">
          <SearchIcon />
          <input
            type="text"
            value={query}
            onChange={e => setQuery(e.target.value)}
            placeholder="搜尋 target txid / root 位址…"
            className="field pl-7 font-mono"
          />
        </div>

        <div className="flex items-stretch gap-1.5">
          <button
            type="button"
            onClick={() => filter.step(-1)}
            disabled={matches.length === 0}
            aria-label="上一條"
            className="btn-icon h-auto"
          >
            ‹
          </button>
          <div className="relative min-w-0 flex-1">
          <select
            value={selectedIdx}
            onChange={e => onSelectedIdxChange(Number(e.target.value))}
            disabled={matches.length === 0}
            className="field cursor-pointer appearance-none pr-7 font-mono"
          >
            {shown.map(({ c, i }) => {
              const tp = c.is_true_positive ? ' [TP]' : ''
              const fr = c.root_is_fraud ? ' ✓' : ''
              return (
                <option key={i} value={i}>
                  {`#${i} · tx ${shortId(c.target_txid)} · d${c.depth} · root ${c.root_type}${fr}${tp}`}
                </option>
              )
            })}
          </select>
          <CaretIcon />
          </div>
          <button
            type="button"
            onClick={() => filter.step(1)}
            disabled={matches.length === 0}
            aria-label="下一條"
            className="btn-icon h-auto"
          >
            ›
          </button>
        </div>

        {matches.length > MAX_OPTIONS && (
          <p className="text-[10.5px] leading-relaxed text-ink-400">
            僅列出前 {MAX_OPTIONS} 筆，請用搜尋縮小範圍（上／下一條仍可走訪全部 {matches.length} 條）。
          </p>
        )}
        {matches.length === 0 && (
          <p className="text-[10.5px] text-amber-300/90">查無符合的詐欺鏈。</p>
        )}

        {/* φ_asym 覆蓋率 + 只看有 φ 的鏈 */}
        <div className="rounded-lg bg-ink-800/60 p-1 ring-1 ring-white/[0.05]">
          <Switch
            checked={onlyPhi}
            onChange={toggleOnlyPhi}
            label={
              <span className="flex items-center gap-1.5">
                <span style={{ color: COLOR.pivot }}>★</span>
                只看有 <PhiAsym /> 的鏈
              </span>
            }
            hint={
              <>
                <PhiAsym /> 覆蓋{' '}
                <span className="font-mono text-ink-200">{nWithPhi}/{data.chains.length}</span>（{phiPct}%）
                = transaction-target 鏈；其餘為 wallet-target（任務外，僅有 CE 追蹤）。
              </>
            }
          />
        </div>
      </Section>

      <Divider />

      {/* 收斂視圖 (fan-in) */}
      <Section
        title="收斂源頭 fan-in"
        description="多條可疑交易回溯到同一源頭。點選查看收斂扇入。"
        meta={
          convergeRootId ? (
            <button type="button" onClick={() => onConvergeChange(null)} className="text-rose-300 hover:underline">
              退出 ✕
            </button>
          ) : undefined
        }
      >
        {rootGroups.length === 0 ? (
          <p className="text-[11px] text-ink-500">無多鏈共用的源頭。</p>
        ) : (
          <div className="scroll-thin flex max-h-48 flex-col gap-1 overflow-y-auto pr-1">
            {rootGroups.map(g => {
              const active = g.rootRealId === convergeRootId
              return (
                <button
                  key={g.rootRealId}
                  type="button"
                  onClick={() => onConvergeChange(active ? null : g.rootRealId)}
                  className={`flex items-center justify-between gap-2 rounded-lg px-2 py-1.5 text-left ring-1 transition-all duration-200 active:scale-[0.99] ${
                    active
                      ? 'bg-rose-400/10 text-rose-100 ring-rose-400/40'
                      : 'bg-ink-800/60 text-ink-200 ring-white/[0.05] hover:bg-ink-700 hover:ring-white/10'
                  }`}
                >
                  <span className="flex min-w-0 items-center gap-1.5 font-mono text-[11px]">
                    <span style={{ color: g.rootType === 'transaction' ? COLOR.tx : COLOR.wallet }}>{typeGlyph(g.rootType)}</span>
                    <span className="truncate">{shortId(g.rootRealId)}</span>
                    {g.isFraud && <span className="shrink-0 text-rose-300">✓詐欺</span>}
                  </span>
                  <span className={`shrink-0 rounded-md px-1.5 py-0.5 font-mono text-[10.5px] ${active ? 'bg-rose-400/20 text-rose-100' : 'bg-ink-700 text-amber-200'}`}>
                    {g.count} 條
                  </span>
                </button>
              )
            })}
          </div>
        )}
      </Section>

      <Divider />

      {/* 檢視 */}
      <Section title="檢視 view">
        <Segmented
          ariaLabel="呈現方式"
          value={display.dimensions}
          onChange={setDimensions}
          options={[
            { value: 2, label: '2D 平面' },
            { value: 3, label: '3D 立體' },
          ]}
        />
        <Switch
          checked={display.layout === 'temporal'}
          onChange={setLayout}
          label="時序佈局"
          hint="X＝時間、上下依型別分層"
        />
      </Section>

      <Divider />

      {/* 顯示選項 */}
      <Section title="顯示 display">
        <div className="flex flex-col">
          <Switch checked={display.showAll} onChange={v => setField('showAll', v)} label="顯示全部鏈" />
          <Switch
            checked={display.neighbors}
            onChange={v => setField('neighbors', v)}
            label="顯示一階鄰居（反灰）"
            hint={display.neighbors && !display.showAll ? `鄰居來源：${NEIGHBOR_SOURCE_LABEL[neighborSource]}` : undefined}
          />
          <Switch checked={display.particles} onChange={v => setField('particles', v)} label="金流方向粒子" />
          <Switch
            checked={display.sizeByPhi}
            onChange={v => setField('sizeByPhi', v)}
            label={<span>依 <PhiAsym /> 大小著色</span>}
          />
        </div>
      </Section>

      <Divider />

      {/* 型別篩選 */}
      <Section title="型別篩選 type" description="未勾選任何項＝全部顯示">
        <div className="flex gap-1.5">
          {TYPE_FILTERS.map(f => {
            const on = display.typeFilter.has(f.value)
            return (
              <button
                key={f.value}
                type="button"
                role="checkbox"
                aria-checked={on}
                onClick={() => toggleType(f.value)}
                className={`flex flex-1 items-center gap-2 rounded-lg px-2.5 py-1.5 text-[11px] ring-1 transition-all duration-200 active:scale-[0.98] ${
                  on ? 'bg-ink-600 text-ink-100 ring-white/10' : 'bg-ink-800/60 text-ink-300 ring-white/[0.05] hover:bg-ink-700'
                }`}
              >
                <span style={{ color: f.color }}>{f.glyph}</span>
                {f.label}
                <span className={`ml-auto h-1.5 w-1.5 rounded-full ${on ? 'bg-brand' : 'bg-ink-500'}`} aria-hidden />
              </button>
            )
          })}
        </div>
      </Section>

      <Divider />

      {/* 資料載入 */}
      <Section title="資料 data">
        <button type="button" onClick={() => fileRef.current?.click()} className="btn-ghost w-full justify-center">
          <UploadIcon />
          載入 crime_chains.json / .csv
        </button>
        <input ref={fileRef} type="file" accept=".csv,application/json,.json" onChange={onFile} className="hidden" />
        <dl className="grid grid-cols-[auto_1fr] gap-x-3 gap-y-1 font-mono text-[10.5px] text-ink-400">
          <dt>來源</dt>
          <dd className="truncate text-ink-200" title={source}>{source}</dd>
          <dt>節點</dt>
          <dd className="text-ink-200">{visible.nodes}</dd>
          <dt>連線</dt>
          <dd className="text-ink-200">{visible.links}</dd>
        </dl>
      </Section>
    </div>
  )
}

function SearchIcon() {
  return (
    <svg width="12" height="12" viewBox="0 0 12 12" aria-hidden className="pointer-events-none absolute left-2.5 top-1/2 -translate-y-1/2 text-ink-500">
      <circle cx="5" cy="5" r="3.5" fill="none" stroke="currentColor" strokeWidth="1.4" />
      <path d="M7.8 7.8 11 11" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" />
    </svg>
  )
}

function CaretIcon() {
  return (
    <svg width="10" height="10" viewBox="0 0 10 10" aria-hidden className="pointer-events-none absolute right-2.5 top-1/2 -translate-y-1/2 text-ink-400">
      <path d="M2 3.5 5 6.5 8 3.5" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  )
}

function UploadIcon() {
  return (
    <svg width="12" height="12" viewBox="0 0 12 12" aria-hidden>
      <path d="M6 8V1.5M3.5 4 6 1.5 8.5 4" fill="none" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" strokeLinejoin="round" />
      <path d="M1.5 8.5v1.5a.5.5 0 0 0 .5.5h8a.5.5 0 0 0 .5-.5V8.5" fill="none" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" />
    </svg>
  )
}
