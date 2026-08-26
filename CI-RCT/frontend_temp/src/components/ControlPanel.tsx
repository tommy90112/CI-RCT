import { useMemo, useRef, useState, type ChangeEvent } from 'react'
import type { CrimeChain, CrimeChainData, DataVariant, DisplayOptions, NeighborSource, NodeType } from '../types'
import type { RootGroup } from '../lib/graph'
import { COLOR, shortId, typeGlyph } from '../lib/render'
import { PhiAsym } from './Phi'

// Cap rendered <option> count so a 2000-chain CSV stays snappy; users narrow
// the list with the search box.
const MAX_OPTIONS = 300

/** A chain has a φ_asym responsibility spotlight iff it is transaction-target. */
const chainHasPhi = (c: CrimeChain): boolean => c.nodes.some(n => n.phi_asym != null)

// Left-drawer content: node-display controls only (控制面板).
interface ControlPanelProps {
  data: CrimeChainData
  display: DisplayOptions
  onDisplayChange: (next: DisplayOptions) => void
  selectedIdx: number
  onSelectedIdxChange: (i: number) => void
  rootGroups: RootGroup[]
  convergeRootId: string | null
  onConvergeChange: (rootRealId: string | null) => void
  variant: DataVariant
  onVariantChange: (variant: DataVariant) => void
  dataError: string | null
  visible: { nodes: number; links: number }
  neighborSource: NeighborSource
  source: string
  onLoadFile: (file: File) => void
}

const VARIANTS: { value: DataVariant; label: string }[] = [
  { value: 'joint', label: 'joint（主）' },
  { value: 'transaction', label: 'transaction' },
  { value: 'wallet', label: 'wallet' },
]

const TYPE_FILTERS: { value: NodeType; label: string }[] = [
  { value: 'transaction', label: '交易 transaction' },
  { value: 'wallet', label: '錢包 wallet' },
]

export function ControlPanel(props: ControlPanelProps) {
  const {
    data,
    display,
    onDisplayChange,
    selectedIdx,
    onSelectedIdxChange,
    rootGroups,
    convergeRootId,
    onConvergeChange,
    variant,
    onVariantChange,
    dataError,
    visible,
    neighborSource,
    source,
    onLoadFile,
  } = props

  const fileRef = useRef<HTMLInputElement | null>(null)
  const [query, setQuery] = useState('')
  const [onlyPhi, setOnlyPhi] = useState(false)

  // φ_asym coverage across the loaded data (spec §3): transaction-target chains.
  const nWithPhi = useMemo(() => data.chains.filter(chainHasPhi).length, [data.chains])
  const phiPct = data.chains.length ? Math.round((nWithPhi / data.chains.length) * 100) : 0

  // Chains matching the search box (by target txid / root address / root type),
  // optionally restricted to those carrying φ_asym (transaction-target).
  const matches = useMemo(() => {
    const q = query.trim().toLowerCase()
    const all = data.chains.map((c, i) => ({ c, i }))
    return all.filter(({ c }) => {
      if (onlyPhi && !chainHasPhi(c)) return false
      if (!q) return true
      return (
        c.target_txid.toLowerCase().includes(q) ||
        c.root_real_id.toLowerCase().includes(q) ||
        c.root_type.toLowerCase().includes(q)
      )
    })
  }, [data.chains, query, onlyPhi])

  // Turning the φ-only filter on jumps to the first φ chain if the current one lacks φ.
  const toggleOnlyPhi = () => {
    const next = !onlyPhi
    setOnlyPhi(next)
    if (next && !chainHasPhi(data.chains[selectedIdx])) {
      const firstPhi = data.chains.findIndex(chainHasPhi)
      if (firstPhi >= 0) onSelectedIdxChange(firstPhi)
    }
  }

  // Options to render: first MAX_OPTIONS matches, always including the selected
  // chain so the <select> value stays in sync even when filtered out.
  const shown = useMemo(() => {
    const slice = matches.slice(0, MAX_OPTIONS)
    if (slice.some((m) => m.i === selectedIdx)) return slice
    const sel = data.chains[selectedIdx]
    return sel ? [{ c: sel, i: selectedIdx }, ...slice] : slice
  }, [matches, selectedIdx, data.chains])

  const posInMatches = matches.findIndex((m) => m.i === selectedIdx)

  const step = (delta: number) => {
    if (matches.length === 0) return
    const base = posInMatches < 0 ? 0 : posInMatches
    const next = (base + delta + matches.length) % matches.length
    onSelectedIdxChange(matches[next].i)
  }

  const setLayout = (temporal: boolean) =>
    onDisplayChange({ ...display, layout: temporal ? 'temporal' : 'structure' })

  const setDimensions = (d: 2 | 3) => onDisplayChange({ ...display, dimensions: d })

  const toggleField = (field: 'showAll' | 'neighbors' | 'particles' | 'sizeByPhi') =>
    onDisplayChange({ ...display, [field]: !display[field] })

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
    <div className="flex flex-col gap-4 text-xs text-slate-300">
      {/* Title */}
      <div>
        <h2 className="text-sm font-semibold text-sky-400">節點顯示控制</h2>
        <p className="mt-0.5 text-[11px] text-slate-500">
          控制面板 · 調整圖形呈現方式
        </p>
      </div>

      {/* 資料變體 */}
      <section className="border-t border-slate-700/60 pt-3">
        <label className="mb-1.5 block text-[11px] font-medium text-slate-400">
          資料變體 (variant)
        </label>
        <div className="flex overflow-hidden rounded-md border border-slate-700/60">
          {VARIANTS.map(v => (
            <button
              key={v.value}
              type="button"
              onClick={() => onVariantChange(v.value)}
              className={`flex-1 px-1.5 py-1.5 text-[10px] font-medium transition ${
                variant === v.value
                  ? 'bg-sky-500/20 text-sky-300'
                  : 'bg-slate-800/80 text-slate-400 hover:bg-slate-700/60'
              }`}
            >
              {v.label}
            </button>
          ))}
        </div>
        {dataError && (
          <p className="mt-1.5 rounded-md border border-amber-500/30 bg-amber-500/10 px-2 py-1 text-[10px] leading-relaxed text-amber-300">
            {dataError}
          </p>
        )}
      </section>

      {/* 鏈選擇 */}
      <section className="border-t border-slate-700/60 pt-3">
        <div className="mb-1.5 flex items-baseline justify-between">
          <label className="text-[11px] font-medium text-slate-400">
            選擇詐欺鏈 (fraud chain)
          </label>
          <span className="font-mono text-[10px] text-slate-500">
            共 {data.chains.length} 條{query.trim() ? ` · 符合 ${matches.length}` : ''}
          </span>
        </div>

        {/* 搜尋 */}
        <input
          type="text"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="搜尋 target txid / root 位址…"
          className="mb-1.5 w-full rounded-md border border-slate-700/60 bg-slate-800/80 px-2 py-1.5 font-mono text-[11px] text-slate-200 outline-none placeholder:text-slate-600 focus:border-sky-500"
        />

        {/* φ_asym 覆蓋率 + 只看有 φ 的鏈 */}
        <div className="mb-1.5 rounded-md border border-slate-700/60 bg-slate-800/40 p-2">
          <button
            type="button"
            onClick={toggleOnlyPhi}
            className={`flex w-full items-center justify-between rounded px-1.5 py-1 text-[11px] font-medium transition ${
              onlyPhi ? 'bg-amber-500/20 text-amber-300' : 'text-slate-300 hover:bg-slate-700/40'
            }`}
          >
            <span className="flex items-center gap-1.5">
              <span style={{ color: COLOR.pivot }}>★</span>
              只看有 <PhiAsym /> 的鏈
            </span>
            <span className={`rounded px-1.5 py-0.5 text-[10px] ${onlyPhi ? 'bg-amber-400/30' : 'bg-slate-700/60'}`}>
              {onlyPhi ? '開' : '關'}
            </span>
          </button>
          <p className="mt-1 px-1.5 text-[10px] leading-relaxed text-slate-500">
            <PhiAsym /> 覆蓋 <span className="font-mono text-slate-300">{nWithPhi}/{data.chains.length}</span>（{phiPct}%）
            = transaction-target 鏈;其餘為 wallet-target（任務外,僅有 CE 追蹤)。
          </p>
        </div>

        {/* 下拉選單 + 上/下一條 */}
        <div className="flex items-stretch gap-1.5">
          <button
            type="button"
            onClick={() => step(-1)}
            disabled={matches.length === 0}
            aria-label="上一條"
            className="rounded-md border border-slate-700/60 bg-slate-800/80 px-2 text-slate-300 hover:bg-slate-700 disabled:opacity-40"
          >
            ‹
          </button>
          <select
            value={selectedIdx}
            onChange={(e) => onSelectedIdxChange(Number(e.target.value))}
            disabled={matches.length === 0}
            className="min-w-0 flex-1 rounded-md border border-slate-700/60 bg-slate-800/80 px-2 py-1.5 font-mono text-[11px] text-slate-200 outline-none focus:border-sky-500 disabled:opacity-40"
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
          <button
            type="button"
            onClick={() => step(1)}
            disabled={matches.length === 0}
            aria-label="下一條"
            className="rounded-md border border-slate-700/60 bg-slate-800/80 px-2 text-slate-300 hover:bg-slate-700 disabled:opacity-40"
          >
            ›
          </button>
        </div>

        {matches.length > MAX_OPTIONS && (
          <p className="mt-1 text-[10px] text-slate-500">
            僅列出前 {MAX_OPTIONS} 筆,請用搜尋縮小範圍(上/下一條仍可走訪全部 {matches.length} 條)。
          </p>
        )}
        {matches.length === 0 && (
          <p className="mt-1 text-[10px] text-amber-400/80">查無符合的詐欺鏈。</p>
        )}
      </section>

      {/* 收斂視圖 (fan-in) */}
      <section className="border-t border-slate-700/60 pt-3">
        <div className="mb-1.5 flex items-baseline justify-between">
          <label className="text-[11px] font-medium text-slate-400">
            收斂源頭 (fan-in)
          </label>
          {convergeRootId && (
            <button
              type="button"
              onClick={() => onConvergeChange(null)}
              className="font-mono text-[10px] text-rose-300 hover:underline"
            >
              退出 ✕
            </button>
          )}
        </div>
        <p className="mb-1.5 text-[10px] leading-relaxed text-slate-500">
          多條可疑交易回溯到同一源頭。點選查看收斂扇入。
        </p>
        {rootGroups.length === 0 ? (
          <p className="text-[10px] text-slate-600">無多鏈共用的源頭。</p>
        ) : (
          <div className="flex max-h-44 flex-col gap-1 overflow-y-auto pr-1">
            {rootGroups.map(g => {
              const active = g.rootRealId === convergeRootId
              return (
                <button
                  key={g.rootRealId}
                  type="button"
                  onClick={() => onConvergeChange(active ? null : g.rootRealId)}
                  className={`flex items-center justify-between gap-2 rounded-md border px-2 py-1 text-left transition ${
                    active
                      ? 'border-rose-500/60 bg-rose-500/15 text-rose-200'
                      : 'border-slate-700/60 bg-slate-800/60 text-slate-300 hover:border-slate-500'
                  }`}
                >
                  <span className="flex min-w-0 items-center gap-1 font-mono text-[10px]">
                    {typeGlyph(g.rootType)} {shortId(g.rootRealId)}
                    {g.isFraud && <span className="text-rose-400">✓詐欺</span>}
                  </span>
                  <span className="shrink-0 rounded bg-slate-700/60 px-1.5 py-0.5 font-mono text-[10px] text-amber-300">
                    {g.count} 條
                  </span>
                </button>
              )
            })}
          </div>
        )}
      </section>

      {/* 呈現方式 2D / 3D */}
      <section className="border-t border-slate-700/60 pt-3">
        <label className="mb-1.5 block text-[11px] font-medium text-slate-400">
          呈現方式
        </label>
        <div className="flex overflow-hidden rounded-md border border-slate-700/60">
          {([2, 3] as const).map((d) => (
            <button
              key={d}
              type="button"
              onClick={() => setDimensions(d)}
              className={`flex-1 px-2 py-1.5 text-[11px] font-medium transition ${
                display.dimensions === d
                  ? 'bg-sky-500/20 text-sky-300'
                  : 'bg-slate-800/80 text-slate-400 hover:bg-slate-700/60'
              }`}
            >
              {d === 2 ? '2D 平面' : '3D 立體'}
            </button>
          ))}
        </div>
      </section>

      {/* 佈局 */}
      <section className="border-t border-slate-700/60 pt-3">
        <label className="mb-1.5 block text-[11px] font-medium text-slate-400">
          佈局
        </label>
        <label className="flex cursor-pointer items-center gap-2">
          <input
            type="checkbox"
            checked={display.layout === 'temporal'}
            onChange={(e) => setLayout(e.target.checked)}
            className="accent-sky-500"
          />
          <span>時序佈局（X＝時間、上下分層）</span>
        </label>
      </section>

      {/* 顯示選項 */}
      <section className="border-t border-slate-700/60 pt-3">
        <label className="mb-1.5 block text-[11px] font-medium text-slate-400">
          顯示選項
        </label>
        <div className="flex flex-col gap-1.5">
          <label className="flex cursor-pointer items-center gap-2">
            <input
              type="checkbox"
              checked={display.showAll}
              onChange={() => toggleField('showAll')}
              className="accent-sky-500"
            />
            <span>顯示全部鏈</span>
          </label>
          <label className="flex cursor-pointer items-center gap-2">
            <input
              type="checkbox"
              checked={display.neighbors}
              onChange={() => toggleField('neighbors')}
              className="accent-sky-500"
            />
            <span>顯示一階鄰居（反灰）</span>
          </label>
          {display.neighbors && !display.showAll && (
            <p className="ml-6 text-[10px] text-slate-500">
              鄰居來源：
              {neighborSource === 'full'
                ? '完整 Elliptic++ 圖'
                : neighborSource === 'union'
                  ? '鏈聯集（未載入 chain_neighbors.json）'
                  : '—'}
            </p>
          )}
          <label className="flex cursor-pointer items-center gap-2">
            <input
              type="checkbox"
              checked={display.particles}
              onChange={() => toggleField('particles')}
              className="accent-sky-500"
            />
            <span>金流方向粒子</span>
          </label>
          <label className="flex cursor-pointer items-center gap-2">
            <input
              type="checkbox"
              checked={display.sizeByPhi}
              onChange={() => toggleField('sizeByPhi')}
              className="accent-sky-500"
            />
            <span>依 <PhiAsym /> 大小著色</span>
          </label>
        </div>
      </section>

      {/* 型別篩選 */}
      <section className="border-t border-slate-700/60 pt-3">
        <label className="mb-1 block text-[11px] font-medium text-slate-400">
          型別篩選
        </label>
        <p className="mb-1.5 text-[11px] text-slate-500">未勾選任何項＝全部顯示</p>
        <div className="flex flex-col gap-1.5">
          {TYPE_FILTERS.map((f) => (
            <label key={f.value} className="flex cursor-pointer items-center gap-2">
              <input
                type="checkbox"
                checked={display.typeFilter.has(f.value)}
                onChange={() => toggleType(f.value)}
                className="accent-sky-500"
              />
              <span>{f.label}</span>
            </label>
          ))}
        </div>
      </section>

      {/* 圖例 LEGEND */}
      <section className="border-t border-slate-700/60 pt-3">
        <label className="mb-1.5 block text-[11px] font-medium text-slate-400">
          圖例
        </label>
        <div className="flex flex-col gap-1.5 text-[11px]">
          <LegendDot color={COLOR.tx} label="交易 transaction" />
          <LegendDot color={COLOR.wallet} label="錢包 wallet" />
          <LegendDot color={COLOR.fraud} label="非法 illicit" />
          <LegendDot color={COLOR.root} label="根因 root（光暈）" />
          <div className="flex items-center gap-2">
            <span style={{ color: COLOR.pivot }}>★</span>
            <span><PhiAsym /> 元兇 pivot</span>
          </div>
          <LegendDot color={COLOR.dim} label="一階鄰居（反灰）" />
          <div className="mt-1 flex items-center gap-2">
            <span className="inline-block h-0.5 w-5 rounded" style={{ backgroundColor: COLOR.cePos }} />
            <span>CE＋（正向因果效應）</span>
          </div>
          <div className="flex items-center gap-2">
            <span className="inline-block h-0.5 w-5 rounded" style={{ backgroundColor: COLOR.ceNeg }} />
            <span>CE－（負向因果效應）</span>
          </div>
          <p className="mt-1 text-slate-500">球越大＝<PhiAsym /> 越高;★＝pivot 元兇</p>
        </div>
      </section>

      {/* 資料載入 */}
      <section className="border-t border-slate-700/60 pt-3">
        <label className="mb-1.5 block text-[11px] font-medium text-slate-400">
          資料載入
        </label>
        <button
          type="button"
          onClick={() => fileRef.current?.click()}
          className="w-full rounded-md border border-sky-500/60 bg-sky-500/10 px-2 py-1.5 text-[11px] font-medium text-sky-300 transition hover:bg-sky-500/20"
        >
          載入 crime_chains.json / .csv
        </button>
        <input
          ref={fileRef}
          type="file"
          accept=".csv,application/json,.json"
          onChange={onFile}
          className="hidden"
        />
        <div className="mt-2 space-y-0.5 font-mono text-[11px] text-slate-500">
          <div className="truncate" title={source}>
            來源：{source}
          </div>
          <div>
            節點 {visible.nodes} · 連線 {visible.links}
          </div>
        </div>
      </section>
    </div>
  )
}

interface LegendDotProps {
  color: string
  label: string
}

function LegendDot(props: LegendDotProps) {
  return (
    <div className="flex items-center gap-2">
      <span
        className="inline-block h-2.5 w-2.5 rounded-full"
        style={{ backgroundColor: props.color }}
      />
      <span>{props.label}</span>
    </div>
  )
}
