import { Fragment, useState } from 'react';
import { Section, ReadingGuide } from '../Section';
import { DEMO_CHAIN } from '../demoChain';
import { COLOR, shortId, typeGlyph } from '../../lib/render';

// Interactive do-intervention: cut one edge of the real demo chain and read
// off the causal effect CE(u→v) that the engine measured for that edge.
export function Intervention() {
  const [cut, setCut] = useState<number | null>(1); // edge index: node i → node i+1
  const edges = DEMO_CHAIN.slice(0, -1).map((u, i) => ({ i, u, v: DEMO_CHAIN[i + 1], ce: u.ce ?? 0 }));
  const sel = cut != null ? edges[cut] : null;

  return (
    <Section
      id="intervention"
      eyebrow="干預，而不是相關"
      title="切斷一條邊，看下游怎麼變——差值就是因果效應。"
      lede="do-calculus 的做法很直接：把某個上游節點的狀態「介入」到一個參考基線（該型別的平均嵌入），重新讀出下游的詐欺機率。與保留原狀相比的差，就是這條邊的因果效應 CE。它回答的是「拿掉它，結果會不會變」，而不是「它跟結果像不像」。"
    >
      <div>
        <div className="float p-6 md:p-8">
          {/* Chain with cuttable edges — a 7-column grid (node·edge·node·…) that
              always fills the card width; only very narrow screens scroll.
              `overflow-x-auto` also clips vertically, so keep 4px of padding
              above/below for the 3px root/pivot rings (box-shadow). */}
          <div className="overflow-x-auto py-1">
            <div className="grid min-w-[640px] grid-cols-[1fr_1.3fr_1fr_1.3fr_1fr_1.3fr_1fr] items-start">
              {DEMO_CHAIN.map((n, i) => {
                const isTx = n.type === 'transaction';
                const downstreamOfCut = sel != null && i > sel.i;
                return (
                  <Fragment key={n.id}>
                    <div className={`flex flex-col items-center gap-2 px-1 transition-opacity duration-300 ${downstreamOfCut ? 'opacity-60' : ''}`}>
                      <div
                        className={`flex h-16 w-16 items-center justify-center text-[22px] font-bold text-ink-950 ${isTx ? 'rounded-lg' : 'rounded-full'}`}
                        style={{ background: COLOR.fraud, boxShadow: n.role === 'pivot' ? `0 0 0 3px ${COLOR.pivot}` : n.role === 'root' ? `0 0 0 3px ${COLOR.root}` : undefined }}
                      >
                        {typeGlyph(n.type)}
                      </div>
                      <div className="whitespace-nowrap font-mono text-[13.5px] text-ink-100">{shortId(n.id)}</div>
                      <div className="whitespace-nowrap text-[12.5px] text-ink-400">
                        {n.role === 'root' ? '根因 root' : n.role === 'pivot' ? '★ 元兇 pivot' : n.role === 'target' ? '⚑ 目標 target' : n.type}
                      </div>
                    </div>
                    {i < edges.length && (
                      <EdgeControl cut={cut === i} ce={edges[i].ce} onToggle={() => setCut(cut === i ? null : i)} />
                    )}
                  </Fragment>
                );
              })}
            </div>
          </div>

          {/* Readout */}
          <div className="hairline mt-6 grid gap-4 border-t pt-5 md:grid-cols-[1fr_auto]">
            {sel ? (
              <p className="text-[15px] leading-[1.75] text-ink-200">
                對 <span className="font-mono text-ink-100">{shortId(sel.u.id)}</span> 做 do(·＝基線)，
                下游 <span className="font-mono text-ink-100">{shortId(sel.v.id)}</span> 的詐欺機率下降
                <span className="mx-1 font-mono font-semibold" style={{ color: COLOR.cePos }}>{sel.ce.toFixed(3)}</span>
                ——這條邊上的上游節點確實在「推動」下游的詐欺判定。
              </p>
            ) : (
              <p className="text-[15px] leading-relaxed text-ink-400">點一條邊上的「切斷」，讀出引擎量到的因果效應。</p>
            )}
            <div className="rounded-lg bg-ink-800/70 px-4 py-3 font-mono text-[13px] leading-relaxed text-ink-200 ring-1 ring-white/[0.05]">
              CE(u→v) = P(v 詐欺 | u) − P(v 詐欺 | do(u＝基線))
              {sel && <div className="mt-1" style={{ color: COLOR.cePos }}>= +{sel.ce.toFixed(3)}</div>}
            </div>
          </div>
          <p className="mt-4 text-[12.5px] text-ink-500">數值為 Elliptic++ 鏈 #33 的實際 CE；基線為該節點型別的平均嵌入（marginal）。</p>
        <ReadingGuide
          horizontal
          title="怎麼讀"
          items={[
            { k: '基線是什麼', v: '不是把節點刪掉，而是把它的嵌入換成同型別的平均值——「一個沒有特別之處的錢包／交易」。這樣干預後的圖仍是一張合法的圖。' },
            { k: '正負號的意義', v: 'CE 為正表示上游在推動下游的詐欺判定；為負表示它在抑制。追溯時以帶號 CE 排序，正向最強的邊優先。' },
            { k: '與相關性的差別', v: '相關性方法問「哪些鄰居長得像詐欺」；干預問「拿掉這個鄰居，判定會不會變」。後者才有反事實意義，也才能拿來究責。' },
          ]}
        />
        </div>

      </div>
    </Section>
  );
}

function EdgeControl({ cut, ce, onToggle }: { cut: boolean; ce: number; onToggle: () => void }) {
  const color = cut ? COLOR.ceNeg : COLOR.cePos;
  return (
    <div className="flex flex-col items-center gap-2.5 px-3 pt-6">
      {/* Arrow row: 16px tall, offset 24px → its centre (32px) lines up with
          the centre of the 64px node tiles. Line and head sit in one
          vertically-centred flex row, so nothing depends on glyph metrics. */}
      <div className="relative flex h-4 w-full items-center">
        <div
          className="h-0.5 flex-1 rounded-full transition-all duration-300"
          style={cut
            ? { backgroundImage: `repeating-linear-gradient(90deg, ${COLOR.ceNeg} 0 6px, transparent 6px 12px)` }
            : { background: COLOR.cePos }}
        />
        <svg width="10" height="12" viewBox="0 0 10 12" aria-hidden="true" className="-ml-px shrink-0" style={{ color }}>
          <path d="M0 0 L10 6 L0 12 Z" fill="currentColor" />
        </svg>
        {cut && (
          <span className="absolute left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2 text-[18px] leading-none" style={{ color: COLOR.ceNeg }}>✂</span>
        )}
      </div>
      <button
        type="button"
        onClick={onToggle}
        aria-pressed={cut}
        className={`whitespace-nowrap rounded-md px-2.5 py-1 font-mono text-[12.5px] ring-1 transition-all duration-200 active:scale-95 ${
          cut ? 'bg-rose-400/15 text-rose-200 ring-rose-400/50' : 'bg-ink-800/80 text-ink-300 ring-white/[0.06] hover:bg-ink-700 hover:text-ink-100'
        }`}
      >
        {cut ? `CE ${ce.toFixed(3)}` : '切斷 do(·)'}
      </button>
    </div>
  );
}
