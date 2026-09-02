/**
 * GuideCard — entry overlay in the explorer: the three questions the viewer
 * answers, each opening the matching view, plus a "just explore" escape hatch.
 * It is shown every time the explorer is opened without a preset (e.g. via
 * 「開始探索」); deep links from the idea page's question cards skip it.
 */
import { explorerHref } from '../lib/route';
import type { ExplorerPreset } from '../lib/route';
import { PhiAsym } from '../components/Phi';

const QUESTIONS: { preset: ExplorerPreset; n: string; q: string; a: React.ReactNode }[] = [
  { preset: 'converge', n: '01', q: '這些可疑交易的共同源頭是誰？', a: '收斂視圖：多條鏈匯入同一個源頭。' },
  { preset: 'chain', n: '02', q: '單一條詐欺鏈長怎樣、錢從哪來？', a: '金流鏈：錢包與交易交替、時序正確的資金路徑。' },
  { preset: 'responsibility', n: '03', q: '鏈上誰負最大因果責任？', a: <>因果責任面板：CE 追路徑、<PhiAsym /> 釘元兇。</> },
];

export function GuideCard({ onDismiss }: { onDismiss: () => void }) {
  return (
    <div className="absolute inset-0 z-40 flex items-center justify-center bg-ink-950/70 p-6 backdrop-blur-sm animate-fade-in">
      <div className="float w-full max-w-2xl p-6 animate-slide-up" role="dialog" aria-labelledby="guide-title">
        <div className="eyebrow mb-2">追溯探索器</div>
        <h2 id="guide-title" className="text-lg font-semibold tracking-tight text-ink-100">
          從三個問題開始
        </h2>
        <p className="mt-1 text-[12px] text-ink-300">選一個問題，探索器會直接切到對應的視圖；之後可隨時用頂欄與左側控制台切換。</p>
        <div className="mt-5 grid gap-2 sm:grid-cols-3">
          {QUESTIONS.map(item => (
            <a
              key={item.preset}
              href={explorerHref(item.preset)}
              onClick={onDismiss}
              className="group flex flex-col gap-2 rounded-xl bg-ink-800/70 p-4 ring-1 ring-white/[0.06] transition-all duration-200 hover:bg-ink-700 hover:ring-brand/50 active:scale-[0.99]"
            >
              <span className="font-mono text-[12px] text-brand">{item.n}</span>
              <span className="text-[14px] font-semibold leading-snug text-ink-100">{item.q}</span>
              <span className="text-[12px] leading-relaxed text-ink-400 group-hover:text-ink-300">{item.a}</span>
            </a>
          ))}
        </div>
        <div className="mt-5 flex items-center justify-between">
          <a href="#/" className="text-[12px] text-ink-400 hover:text-brand">← 回想法頁</a>
          <button type="button" onClick={onDismiss} className="btn-ghost">直接探索</button>
        </div>
      </div>
    </div>
  );
}
