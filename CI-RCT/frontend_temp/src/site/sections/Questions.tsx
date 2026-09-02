import { Section } from '../Section';
import { explorerHref } from '../../lib/route';
import type { ExplorerPreset } from '../../lib/route';
import { PhiAsym } from '../../components/Phi';
import type { ChainStats } from '../../hooks/useChainStats';

// The three questions are progressive (macro → one chain → one node), so they
// are drawn as a connected sequence rather than three parallel cards.
const QUESTIONS: { preset: ExplorerPreset; n: string; q: string; view: React.ReactNode; tag: string; scope: string }[] = [
  { preset: 'converge', n: '01', q: '這些可疑交易的共同源頭是誰？', view: '收斂視圖：多條鏈匯入同一個源頭，一眼看到扇入。', tag: 'macro', scope: '整批可疑交易' },
  { preset: 'chain', n: '02', q: '單一條詐欺鏈長怎樣、錢從哪來？', view: '金流鏈：錢包與交易交替、時序正確的資金路徑，可切 2D／3D 與時序佈局。', tag: 'L1 結構', scope: '一條鏈' },
  { preset: 'responsibility', n: '03', q: '鏈上誰負最大因果責任？', view: <>因果責任面板：CE 追路徑、<PhiAsym /> 釘元兇，並下鑽到元兇的特徵歸因。</>, tag: 'L2 / L3', scope: '鏈上的一個節點' },
];

export function Questions({ stats, tone }: { stats: ChainStats | null; tone?: 'base' | 'alt' }) {
  return (
    <Section
      tone={tone}
      id="explore"
      eyebrow="開始探索"
      title="三個遞進的問題，交給探索器回答。"
      lede={stats ? `探索器載入的是同一份追溯輸出：${stats.nChains} 條鏈、真實的 Elliptic++ 交易與錢包。由整批到一條鏈、再到一個節點，逐步縮小範圍。` : '由整批到一條鏈、再到一個節點，逐步縮小範圍。'}
    >
      <ol className="relative">
        {/* connector */}
        <span aria-hidden className="absolute bottom-6 left-[27px] top-6 w-px bg-gradient-to-b from-brand/60 via-brand/30 to-transparent" />
        {QUESTIONS.map(item => (
          <li key={item.preset} className="relative grid gap-4 py-5 md:grid-cols-[56px_minmax(0,1fr)] md:items-center">
            <span className="relative z-10 flex h-14 w-14 items-center justify-center rounded-full bg-ink-800 font-mono text-[14px] text-brand ring-1 ring-brand/40">
              {item.n}
            </span>
            <a
              href={explorerHref(item.preset)}
              className="group rounded-xl px-5 py-4 transition-colors duration-200 hover:bg-white/[0.04]"
            >
              <div className="flex flex-wrap items-center gap-2 text-[12.5px] text-ink-400">
                <span className="chip chip-neutral text-[11px]">{item.tag}</span>
                <span>範圍：{item.scope}</span>
              </div>
              <h3 className="mt-2 text-[22px] font-semibold leading-snug tracking-tight text-ink-100">{item.q}</h3>
              <p className="mt-2 max-w-prose text-[14.5px] leading-relaxed text-ink-300">{item.view}</p>
              <span className="mt-3 inline-block text-[14px] font-medium text-brand transition-transform duration-200 group-hover:translate-x-0.5">
                打開探索器 →
              </span>
            </a>
          </li>
        ))}
      </ol>

      <div className="hairline mt-10 grid gap-3 border-t pt-6 text-[13.5px] leading-relaxed text-ink-400 md:grid-cols-[auto_minmax(0,1fr)]">
        <strong className="text-ink-200">適用範圍</strong>
        <p>
          <PhiAsym /> 只對交易目標的鏈定義（錢包沒有詐欺讀出頭），錢包目標的鏈以 CE 追溯呈現；元兇的特徵歸因（L3）在骨幹的受域內以 do-intervention 計算，受域外的節點不做歸因。這些邊界在探索器中都會如實標示。
        </p>
      </div>
    </Section>
  );
}
