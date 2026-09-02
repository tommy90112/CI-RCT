import { useState } from 'react';
import { Section, Fact } from '../Section';
import { PhiAsym } from '../../components/Phi';

type ModuleId = 'm1' | 'm2' | 'm3' | 'm4';

const MODULES: { id: ModuleId; n: string; name: string; short: string; training?: boolean; what: React.ReactNode; out: React.ReactNode }[] = [
  {
    id: 'm1', n: 'Module 1', name: 'HeteroGNN Backbone', short: '型別感知的表徵',
    what: '以異質圖轉換器（HGT）為骨幹，為每一組「來源型別—邊型別—目標型別」學習獨立的注意力參數，讓錢包對交易的付款關係與交易對交易的資金流向各自以獨立權重聚合；同一組嵌入同時供偵測頭與後續因果計算共用。',
    out: '型別感知的節點嵌入，以及交易節點的詐欺／正常判定。',
  },
  {
    id: 'm2', n: 'Module 2', name: 'Causal Intervention Engine', short: '從相關到因果',
    what: <>在尊重時序的型別化因果圖（DAG）上，HeteroNCM 為每種邊型別配置獨立的 MLP，以 do-calculus 估計逐邊的因果效應 CE(u→v)：比較「保留上游節點」與「把它介入至基線」時下游詐欺機率的差異。並行地，非對稱因果 Shapley 以切斷父邊、重跑骨幹得到聯盟值，依拓樸序的前綴聯盟以線性複雜度逐段分配責任，得到 <PhiAsym />。</>,
    out: <>每條邊的 CE（追路徑的排序訊號），與每個上游節點的 <PhiAsym />（逐段局部因果責任）。</>,
  },
  {
    id: 'm3', n: 'Module 3', name: 'RootCauseTracer', short: '反向逐跳回溯',
    what: '從一筆可疑交易出發，沿因果效應最強的上游邊逐跳上溯；四個停止條件（抵達源頭、因果訊號過弱、偵測到環、達深度上限）確保追溯必然終止。安排在整條管線的末端，待嵌入、CE 與 φ 都穩定後才反向溯源。',
    out: '根因節點，以及一條由錢包與交易交替構成、可稽核的資金因果鏈。',
  },
  {
    id: 'm4', n: 'Module 4', name: 'CausalAdversarialGAN', short: '只在訓練階段', training: true,
    what: '以 WGAN-GP 生成器合成「困難詐欺樣本」模擬偽裝行為，判別器沿用骨幹的特徵投影與分類頭；Causal Consistency Loss 約束因果結構在對抗訓練中保持穩定，讓框架在類別不平衡與偽裝之下仍維持偵測與解釋的穩定。',
    out: '更穩健的決策邊界；推論階段完全不介入。',
  },
];

const FLOW = [
  { k: '輸入', v: '型別感知的有向異質圖：交易 ◆ 與錢包 ●，四種型別化有向邊，依時間戳定向為 DAG' },
  { k: 'Module 1', v: '節點嵌入 h · 詐欺判定' },
  { k: 'Module 2', v: '逐邊 CE · 逐段 φ' },
  { k: 'Module 3', v: '根因 + 資金因果鏈' },
  { k: '輸出', v: '可稽核的 crime chain（JSON），鏈上各節點標註因果責任' },
];

export function Framework({ tone }: { tone?: 'base' | 'alt' }) {
  const [active, setActive] = useState<ModuleId>('m2');
  const mod = MODULES.find(m => m.id === active)!;
  return (
    <Section
      tone={tone}
      id="framework"
      eyebrow="主張與框架"
      title="把端到端的偵測，延伸為可解釋、可稽核的根因追溯。"
      lede="CI-RCT 以 Pearl 結構因果模型為理論基礎，整合型別感知的因果干預、不對稱因果 Shapley，以及具因果約束的對抗式訓練。四個模組串成「表徵與偵測 → 因果效應估計 → 根因追溯」的管線，偵測與解釋以單一聯合損失共同最佳化，而非相互犧牲。"
      aside={
        <Fact
          label="設計邏輯"
          value="偵測與解釋一起學，追溯放最後"
          hint="同一組節點嵌入既要判別詐欺、又要承載可信的因果結構；根因追溯安排在嵌入、CE 與 φ 都穩定之後才進行。"
        />
      }
    >
      {/* Data flow ribbon */}
      <ol className="mb-4 grid gap-px overflow-hidden rounded-xl bg-white/[0.06] ring-1 ring-white/[0.06] md:grid-cols-5">
        {FLOW.map((f, i) => (
          <li key={f.k} className="relative bg-ink-900/90 px-4 py-3.5">
            <div className="flex items-center gap-2">
              <span className={`font-mono text-[12px] ${i === 0 || i === FLOW.length - 1 ? 'text-ink-400' : 'text-brand'}`}>{f.k}</span>
              {i < FLOW.length - 1 && <span className="ml-auto text-ink-500" aria-hidden>→</span>}
            </div>
            <p className="mt-1 text-[13.5px] leading-snug text-ink-200">{f.v}</p>
          </li>
        ))}
      </ol>

      <div className="float overflow-hidden">
        {/* Module selector */}
        <div className="hairline grid gap-px border-b bg-white/[0.06] md:grid-cols-4">
          {MODULES.map(m => {
            const on = m.id === active;
            return (
              <button
                key={m.id}
                type="button"
                onClick={() => setActive(m.id)}
                aria-pressed={on}
                className={`flex flex-col items-start gap-1 p-5 text-left transition-colors duration-200 ${
                  on ? 'bg-ink-700/80' : 'bg-ink-900/90 hover:bg-ink-800'
                }`}
              >
                <span className={`font-mono text-[12px] ${on ? 'text-brand' : 'text-ink-500'}`}>{m.n}</span>
                <span className="text-[16px] font-semibold tracking-tight text-ink-100">{m.name}</span>
                <span className="text-[13.5px] text-ink-400">{m.short}</span>
                {m.training && <span className="chip chip-neutral mt-1.5 text-[11px]">training only</span>}
              </button>
            );
          })}
        </div>
        {/* Detail */}
        <div className="grid gap-6 p-6 md:grid-cols-[2fr_1fr]">
          <div>
            <div className="eyebrow mb-2">{mod.n} · 做什麼</div>
            <p className="text-[15px] leading-[1.8] text-ink-200">{mod.what}</p>
          </div>
          <div className="hairline border-l pl-6">
            <div className="eyebrow mb-2">產出</div>
            <p className="text-[14.5px] leading-relaxed text-ink-300">{mod.out}</p>
          </div>
        </div>
      </div>
    </Section>
  );
}
