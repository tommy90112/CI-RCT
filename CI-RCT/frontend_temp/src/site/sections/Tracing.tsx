import { useEffect, useState } from 'react';
import { Section, ReadingGuide } from '../Section';
import { DEMO_CHAIN } from '../demoChain';
import { COLOR, shortId } from '../../lib/render';
import { PhiAsym } from '../../components/Phi';

// Step-through of the backward trace on the real demo chain: each hop picks the
// strongest-CE upstream edge; other upstream candidates are shown greyed (the
// dump carries no CE for edges the tracer rejected, so they carry no number).
const N = DEMO_CHAIN.length; // 4 → hops 1..3, final = stop
const LAST = N; // step index of the stop state

const STOP_CONDITIONS = ['抵達源頭（無進一步上游）', '因果訊號過弱', '偵測到環', '達深度上限'];

export function Tracing({ tone }: { tone?: 'base' | 'alt' }) {
  const [step, setStep] = useState(0);
  const [playing, setPlaying] = useState(false);
  useEffect(() => {
    if (!playing) return;
    const t = window.setInterval(() => setStep(s => (s >= LAST ? (setPlaying(false), s) : s + 1)), 1400);
    return () => window.clearInterval(t);
  }, [playing]);

  // Rendering goes upstream → downstream (left to right) so the target sits at the right.
  const W = 640, H = 250, cy = 120;
  const xs = DEMO_CHAIN.map((_, i) => 80 + i * 160);
  const reachedIdx = (i: number) => N - 1 - i; // chain index reached at step i (0 = target)
  const reached = new Set<number>();
  for (let s = 0; s <= Math.min(step, N - 1); s++) reached.add(reachedIdx(s));
  const frontier = step >= 1 && step <= N - 1 ? reachedIdx(step) : null;

  return (
    <Section
      tone={tone}
      id="tracing"
      eyebrow="回溯"
      title="從目標往上游，每一跳選因果效應最強的邊。"
      lede="RootCauseTracer 是一個反向的逐跳搜尋：從被判可疑的交易出發，在它的上游候選裡挑 CE 最強的一條邊走過去，再對新節點重複同樣的事。四個停止條件保證它一定會停下來，停下的地方就是根因；走過的路就是可稽核的資金因果鏈。"
    >
      <div className="grid gap-4 lg:grid-cols-[minmax(0,2fr)_minmax(0,1fr)]">
        <div className="float flex flex-col p-6">
          <svg viewBox={`0 0 ${W} ${H}`} className="w-full" role="img" aria-label="反向逐跳回溯的逐步示意">
            {DEMO_CHAIN.slice(0, -1).map((u, i) => {
              const on = reached.has(i) && reached.has(i + 1);
              return (
                <g key={u.id}>
                  <line x1={xs[i] + 22} y1={cy} x2={xs[i + 1] - 22} y2={cy} stroke={on ? COLOR.cePos : '#2c323d'} strokeWidth={on ? 3 : 1.5} />
                  {on && (
                    <text x={(xs[i] + xs[i + 1]) / 2} y={cy - 12} textAnchor="middle" fontSize="12" fontFamily="Geist Mono, monospace" fill={COLOR.cePos}>
                      CE {u.ce?.toFixed(3)}
                    </text>
                  )}
                </g>
              );
            })}
            {frontier != null && frontier > 0 && [-70, 70].map(dy => (
              <g key={dy} opacity="0.6">
                <line x1={xs[frontier] - 22} y1={cy + dy * 0.55} x2={xs[frontier] - 120} y2={cy + dy} stroke="#3d4452" strokeWidth="1.5" strokeDasharray="4 4" />
                <circle cx={xs[frontier] - 120} cy={cy + dy} r="10" fill="#2c323d" stroke="#3d4452" />
                <text x={xs[frontier] - 120} y={cy + dy + 24} textAnchor="middle" fontSize="10.5" fill="#6b7382">其他上游候選</text>
              </g>
            ))}
            {DEMO_CHAIN.map((n, i) => {
              const on = reached.has(i);
              const isTx = n.type === 'transaction';
              const fill = on ? COLOR.fraud : '#2c323d';
              const isFrontier = frontier === i || (step === 0 && i === N - 1);
              return (
                <g key={n.id}>
                  {isFrontier && <circle cx={xs[i]} cy={cy} r="30" fill="none" stroke={COLOR.pivot} strokeWidth="1.5" strokeDasharray="3 3" />}
                  {step === LAST && n.role === 'root' && <circle cx={xs[i]} cy={cy} r="28" fill="none" stroke={COLOR.root} strokeWidth="2.5" />}
                  {isTx ? <rect x={xs[i] - 17} y={cy - 17} width="34" height="34" rx="4" fill={fill} /> : <circle cx={xs[i]} cy={cy} r="18" fill={fill} />}
                  <text x={xs[i]} y={cy + 40} textAnchor="middle" fontSize="12" fontFamily="Geist Mono, monospace" fill={on ? '#e7eaf0' : '#6b7382'}>{shortId(n.id)}</text>
                  <text x={xs[i]} y={cy + 56} textAnchor="middle" fontSize="11" fill="#98a1b3">
                    {n.role === 'target' ? '⚑ 目標' : n.role === 'root' && step === LAST ? '根因 root' : n.type}
                  </text>
                  {step === LAST && n.phi != null && (
                    <text x={xs[i]} y={cy - 34} textAnchor="middle" fontSize="12" fontFamily="Geist Mono, monospace" fill={n.role === 'pivot' ? COLOR.pivot : '#98a1b3'}>
                      {n.role === 'pivot' ? '★ ' : ''}φ {n.phi.toFixed(3)}
                    </text>
                  )}
                </g>
              );
            })}
            <text x={W - 10} y={H - 8} textAnchor="end" fontSize="11" fill="#6b7382">← 回溯方向　　金流方向 →</text>
          </svg>

          {/* Controls + narration */}
          <div className="hairline mt-2 flex flex-col gap-4 border-t pt-4">
            <div className="flex flex-wrap items-center gap-2">
              <button type="button" className="btn-icon" aria-label="上一步" onClick={() => { setPlaying(false); setStep(s => Math.max(0, s - 1)); }} disabled={step === 0}>‹</button>
              <button type="button" className="btn-ghost" onClick={() => { if (step >= LAST) setStep(0); setPlaying(p => !p); }}>
                {playing ? '暫停' : step >= LAST ? '重播' : '自動播放'}
              </button>
              <button type="button" className="btn-icon" aria-label="下一步" onClick={() => { setPlaying(false); setStep(s => Math.min(LAST, s + 1)); }} disabled={step === LAST}>›</button>
              <span className="ml-1 font-mono text-[12.5px] text-ink-400">步驟 {step} / {LAST}</span>
              <ol className="ml-auto flex items-center gap-1" aria-label="步驟進度">
                {Array.from({ length: LAST + 1 }, (_, i) => (
                  <li key={i}>
                    <button
                      type="button"
                      aria-label={`跳到步驟 ${i}`}
                      onClick={() => { setPlaying(false); setStep(i); }}
                      className={`h-2 w-6 rounded-full transition-colors ${i <= step ? 'bg-brand' : 'bg-ink-600 hover:bg-ink-500'}`}
                    />
                  </li>
                ))}
              </ol>
            </div>
            <Narration step={step} />
          </div>
        </div>

        <div className="flex flex-col gap-4">
          <ReadingGuide
            title="停止條件"
            items={STOP_CONDITIONS.map((c, i) => ({
              k: <span className={step === LAST && i === 0 ? 'text-brand' : undefined}>{i + 1}. {c}{step === LAST && i === 0 ? ' ← 本例' : ''}</span>,
              v: ['上游候選集為空，這個節點就是源頭。', '最強候選的 CE 低於門檻，再往上追已無因果意義。', '走回已在鏈上的節點，避免無限循環。', '鏈長達到上限，避免在大圖上無止境搜尋。'][i],
            }))}
          />
          <ReadingGuide
            title="追溯之後"
            items={[
              { k: <>CE 追路徑，<PhiAsym /> 釘元兇</>, v: '回溯用 CE 決定走哪條邊；鏈確定後，再逐段計算非對稱因果 Shapley，把責任分配給鏈上的節點。' },
              { k: '元兇 ≠ 根因', v: '根因是鏈的起點；元兇是責任最重的節點。本例責任塌縮到中段的交易 ★，而不是源頭錢包。' },
            ]}
          />
        </div>
      </div>
    </Section>
  );
}

function Narration({ step }: { step: number }) {
  if (step === 0) return <p className="text-[14.5px] leading-relaxed text-ink-200">起點：被判可疑的交易 <span className="font-mono">{shortId(DEMO_CHAIN[N - 1].id)}</span>。往上游看它的候選父節點。</p>;
  if (step < LAST) {
    const node = DEMO_CHAIN[N - 1 - step];
    return (
      <p className="text-[14.5px] leading-relaxed text-ink-200">
        第 {step} 跳：候選裡 CE 最強的是 <span className="font-mono">{shortId(node.id)}</span>（CE {node.ce?.toFixed(3)}），走過去；其他候選被淘汰。
      </p>
    );
  }
  return (
    <p className="text-[14.5px] leading-relaxed text-ink-200">
      停止：<span className="font-mono">{shortId(DEMO_CHAIN[0].id)}</span> 沒有進一步上游，判為根因。回溯完成後逐段計算 <PhiAsym />：責任塌縮到 ★ <span className="font-mono">{shortId(DEMO_CHAIN[1].id)}</span>（{DEMO_CHAIN[1].phi?.toFixed(3)}）——元兇不一定是根因，也不一定是目標的直接上游。
    </p>
  );
}
