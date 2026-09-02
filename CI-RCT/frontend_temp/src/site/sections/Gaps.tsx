import { Section, Fact } from '../Section';

// A ledger, not three equal cards: number · the gap · the framework's answer.
// Each row pairs a gap with the research objective and module that answer it.
const GAPS = [
  {
    n: '01',
    title: '異質性被抹平',
    ask: '錢包與交易的角色差異，在模型裡去了哪裡？',
    body: '真實網絡並非由單一型別的節點構成，但傳統 GNN 把所有節點視為等價，抹除了型別語意。錢包與交易若被同質化處理，跨型別的因果機制便無從區分。',
    goal: '目的一',
    answer: '型別感知的異質圖學習與因果建模管道：節點與邊的型別語意在訊息傳遞與因果建模兩個階段都被完整保留。',
    module: 'Module 1 · HGT 骨幹 + TypedCausalGraph',
  },
  {
    n: '02',
    title: '解釋停留在「相關」',
    ask: '哪個子圖與預測最相關——但是誰造成的？',
    body: '既有的 GNN 解釋方法（GNNExplainer、PGExplainer 等）以統計相關性為基礎，只能回答「哪個子圖與結果統計上最相關」，答不出異常從哪裡開始、沿何路徑擴散、最終是誰造成的。',
    goal: '目的二',
    answer: '以 Pearl SCM 與 do-calculus 為基礎的 HeteroNCM 與非對稱因果 Shapley，把解釋從相關性提升為具反事實意義的因果效應。',
    module: 'Module 2 · Causal Intervention Engine',
  },
  {
    n: '03',
    title: '缺乏由結果回溯源頭的能力',
    ask: '從被舉報的帳戶，能不能一路追回去？',
    body: '多數 GNN 應用止於端到端的偵測與分類，鮮少從異常結果回溯其源頭；面對跨型別、跨資訊層級的真實資料，能逐層往上追蹤異常路徑的研究屈指可數。',
    goal: '目的三',
    answer: 'RootCauseTracer 從異常節點沿因果關係反向追溯，輸出橫跨多個邏輯層、可解釋、可稽核的完整因果鏈。',
    module: 'Module 3 · RootCauseTracer',
  },
];

export function Gaps() {
  return (
    <Section
      id="gaps"
      eyebrow="三個缺口 → 三個目的"
      title="把 GNN 帶進真實場景，三個問題就浮現了。"
      lede="每個缺口都對應一個現有方法答不出來的問題，也對應本框架的一個研究目的與一個模組。"
      aside={<Fact label="兩項支撐機制" value="CausalAdversarialGAN · 因果 ground-truth" hint="模擬偽裝攻擊以維持穩定；為根因追溯建立客觀驗證基準。" />}
    >
      <div className="hairline hidden grid-cols-[64px_minmax(0,1.15fr)_minmax(0,1fr)] gap-8 border-b pb-3 text-[11.5px] tracking-[0.12em] text-ink-500 md:grid">
        <span />
        <span>缺口 · 現有方法答不出的問題</span>
        <span>本框架的回應</span>
      </div>
      <ol className="hairline divide-y divide-white/[0.06]">
        {GAPS.map(g => (
          <li key={g.n} className="grid gap-5 py-9 md:grid-cols-[64px_minmax(0,1.15fr)_minmax(0,1fr)] md:gap-8">
            <span className="font-mono text-[15px] text-brand md:pt-1">{g.n}</span>
            <div>
              <h3 className="text-[22px] font-semibold tracking-tight text-ink-100">{g.title}</h3>
              <p className="mt-3 border-l-2 border-brand/60 pl-3 text-[15px] italic leading-relaxed text-ink-200">{g.ask}</p>
              <p className="mt-4 max-w-prose text-[14.5px] leading-[1.8] text-ink-400">{g.body}</p>
            </div>
            <div className="self-start rounded-lg bg-ink-800/50 p-5 md:mt-1">
              <div className="flex flex-wrap items-center gap-2">
                <span className="chip chip-ok">{g.goal}</span>
                <span className="text-[12.5px] text-ink-400">{g.module}</span>
              </div>
              <p className="mt-3 text-[14.5px] leading-[1.75] text-ink-200">{g.answer}</p>
            </div>
          </li>
        ))}
      </ol>
    </Section>
  );
}
