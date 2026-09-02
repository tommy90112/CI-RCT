import { Section } from '../Section';
import { PhiAsym } from '../../components/Phi';
import type { ChainStats } from '../../hooks/useChainStats';

const pct = (v: number | undefined): string => (v == null ? '—' : `${Math.round(v * 100)}%`);

export function TwoExplanations({ stats }: { stats: ChainStats | null }) {
  return (
    <Section
      id="explanations"
      eyebrow="兩種「解釋」，要分清楚"
      title="我們問的是「為什麼這筆交易可疑」，答案常常是「某個上游錢包該負責」。"
      lede="被解釋的對象與負責任的對象不是同一件事。前者是提問的主詞，只能是交易（詐欺讀出頭是交易分類器）；後者是答案裡被點名的人，可以是任何型別——而資料告訴我們，它多半是錢包。這正是跨型別 follow-the-money 根因追溯的意義。"
    >
      <div className="grid gap-4 lg:grid-cols-[minmax(0,1fr)_minmax(0,1fr)]">
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-1">
          <div className="rounded-xl bg-ink-800/40 p-6 ring-1 ring-white/[0.05]">
            <div className="eyebrow mb-2">A · 被解釋的對象</div>
            <h3 className="text-[19px] font-semibold text-ink-100">「為什麼這個節點被判可疑？」</h3>
            <p className="mt-3 text-[14.5px] leading-relaxed text-ink-300">提問主詞。必須是 <span className="font-mono text-ink-100">transaction</span>：只有交易有詐欺讀出頭，才有「可疑機率」可以被解釋。</p>
          </div>
          <div className="rounded-xl bg-ink-800/40 p-6 ring-1 ring-white/[0.05]">
            <div className="eyebrow mb-2">B · 責任歸屬的對象</div>
            <h3 className="text-[19px] font-semibold text-ink-100">「答案裡的元兇是誰？」</h3>
            <p className="mt-3 text-[14.5px] leading-relaxed text-ink-300">被點名者，也就是拿到 <PhiAsym /> 的節點。可以是錢包或交易；而在實際追溯結果裡，它絕大多數是<span className="text-ink-100">上游錢包</span>。</p>
          </div>
        </div>

        {/* One headline number, four supporting ones. */}
        <div className="grid gap-px overflow-hidden rounded-xl bg-white/[0.06] ring-1 ring-white/[0.06] sm:grid-cols-[1.3fr_1fr_1fr] sm:grid-rows-2">
          <div className="flex flex-col bg-ink-900/90 p-6 sm:row-span-2">
            <div className="text-[12px] tracking-[0.08em] text-ink-400">在有 <PhiAsym /> 的追溯鏈中</div>
            <div className="mt-3 font-mono text-[60px] font-semibold leading-none tracking-tight text-brand">{pct(stats?.pivotWalletShare)}</div>
            <div className="mt-2 text-[16px] font-medium text-ink-100">元兇是錢包</div>
            <div className="mt-auto pt-4 text-[13px] leading-relaxed text-ink-400">問的是交易，答的是上游錢包——跨型別追溯的意義所在。</div>
          </div>
          <Tile value={pct(stats?.pivotBeyondParentShare)} label="元兇不是目標的直接上游" note="責任落在兩跳以上" />
          <Tile value={pct(stats?.rootWalletShare)} label="根因是錢包" note="全部追溯鏈" />
          <Tile value={stats ? `${stats.maxFanIn} 條` : '—'} label="匯入同一源頭的最多鏈數" note="收斂扇入" />
          <Tile value={stats ? stats.meanDepth.toFixed(2) : '—'} label="平均鏈深度" note="目標到根因的跳數" />
        </div>
      </div>
      <p className="mt-3 text-[12.5px] text-ink-500">
        {stats
          ? `計算自 ${stats.nChains} 條追溯鏈（其中 ${stats.nScored} 條為交易目標、帶有 φ）。皆為追溯輸出的描述性事實，非偵測指標。`
          : '正在從追溯輸出計算描述性統計…'}
      </p>
    </Section>
  );
}

function Tile({ value, label, note }: { value: string; label: string; note: string }) {
  return (
    <div className="bg-ink-900/90 p-5">
      <div className="font-mono text-[28px] font-semibold leading-none tracking-tight text-ink-100">{value}</div>
      <div className="mt-2 text-[13.5px] text-ink-200">{label}</div>
      <div className="mt-0.5 text-[12px] text-ink-500">{note}</div>
    </div>
  );
}
