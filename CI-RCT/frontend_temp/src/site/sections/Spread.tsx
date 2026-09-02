import { Section } from '../Section';
import { COLOR } from '../../lib/render';

// A small heterogeneous graph in which an anomaly propagates outward from one
// source, hop by hop — the "anomalies are paths, not points" beat.
//
// Timing uses SVG's own timeline (SMIL) so every element shares ONE period:
// each hop only differs in when it lights up; everything switches off together.
const NODES: { id: number; x: number; y: number; type: 'w' | 't'; hop: number }[] = [
  { id: 0, x: 70, y: 150, type: 'w', hop: 0 },
  { id: 1, x: 170, y: 90, type: 't', hop: 1 },
  { id: 2, x: 170, y: 210, type: 't', hop: 1 },
  { id: 3, x: 280, y: 60, type: 'w', hop: 2 },
  { id: 4, x: 280, y: 150, type: 'w', hop: 2 },
  { id: 5, x: 280, y: 240, type: 'w', hop: 2 },
  { id: 6, x: 390, y: 100, type: 't', hop: 3 },
  { id: 7, x: 390, y: 200, type: 't', hop: 3 },
  { id: 8, x: 500, y: 150, type: 'w', hop: 4 },
  { id: 9, x: 120, y: 30, type: 'w', hop: 9 },
  { id: 10, x: 410, y: 30, type: 'w', hop: 9 },
  { id: 11, x: 470, y: 260, type: 't', hop: 9 },
];
const EDGES: [number, number][] = [
  [0, 1], [0, 2], [1, 3], [1, 4], [2, 4], [2, 5], [4, 6], [5, 7], [6, 8], [7, 8],
  [9, 1], [3, 10], [11, 8],
];

const PERIOD = 8; // seconds per cycle
const START = 0.6; // s before the source lights
const HOP = 0.8; // s between hops
const DRAW = 0.45; // s an edge takes to draw
const OFF_AT = 0.82; // fraction of the period when everything switches off together
const OFF_LEN = 0.05;

/** keyTimes for "off → (wait) → on → hold → off together". */
function timeline(hop: number, lead = 0): { keyTimes: string; on: string; off: string } {
  const s = Math.min(OFF_AT - 0.02, (START + hop * HOP + lead) / PERIOD);
  const e = Math.min(OFF_AT - 0.01, s + DRAW / PERIOD);
  return {
    keyTimes: `0;${s.toFixed(4)};${e.toFixed(4)};${OFF_AT};${(OFF_AT + OFF_LEN).toFixed(3)};1`,
    on: 'on',
    off: 'off',
  };
}

const fmt = (vals: (string | number)[]) => vals.join(';');

export function Spread({ tone }: { tone?: 'base' | 'alt' }) {
  const byId = new Map(NODES.map(n => [n.id, n]));
  return (
    <Section
      tone={tone}
      id="problem"
      eyebrow="問題不是單點事件"
      title="異常沿著關係結構擴散，跨越多個實體與層級。"
      lede="金融交易、網路通訊、能源調度，本質上都是由多種型別的節點與連結構成的關係網絡。從金融詐欺到基礎設施滲透，異常早已不是單點發生的事件，而是沿著關係結構擴散、跨越多個系統層級的協同模式。"
    >
      <div className="grid items-center gap-8 lg:grid-cols-[1.1fr_1fr]">
        <figure className="float p-4">
          <svg viewBox="0 0 560 290" className="mx-auto w-full max-w-xl" role="img" aria-label="異常從一個源頭沿著關係網絡逐跳擴散的示意動畫">
            {EDGES.map(([a, b]) => {
              const A = byId.get(a)!, B = byId.get(b)!;
              const hop = Math.min(A.hop, B.hop);
              const live = hop < 9;
              if (!live) return <line key={`${a}-${b}`} x1={A.x} y1={A.y} x2={B.x} y2={B.y} stroke="#2c323d" strokeWidth="1.2" />;
              const len = Math.hypot(B.x - A.x, B.y - A.y);
              // Edges draw right after their upstream node lights.
              const { keyTimes } = timeline(hop, 0.25);
              return (
                <line
                  key={`${a}-${b}`}
                  x1={A.x} y1={A.y} x2={B.x} y2={B.y}
                  stroke={COLOR.cePos} strokeWidth="2" strokeDasharray={len} strokeDashoffset={len} opacity="0.3"
                >
                  <animate attributeName="stroke-dashoffset" dur={`${PERIOD}s`} repeatCount="indefinite" keyTimes={keyTimes} values={fmt([len, len, 0, 0, len, len])} />
                  <animate attributeName="opacity" dur={`${PERIOD}s`} repeatCount="indefinite" keyTimes={keyTimes} values={fmt([0.3, 0.3, 1, 1, 0.3, 0.3])} />
                </line>
              );
            })}
            {NODES.map(n => {
              const live = n.hop < 9;
              const base = live ? (n.type === 't' ? COLOR.tx : COLOR.wallet) : '#3d4452';
              const anim = live ? (() => {
                const { keyTimes } = timeline(n.hop);
                return <animate attributeName="fill" dur={`${PERIOD}s`} repeatCount="indefinite" keyTimes={keyTimes} values={fmt([base, base, COLOR.fraud, COLOR.fraud, base, base])} />;
              })() : null;
              return n.type === 't' ? (
                <rect key={n.id} x={n.x - 11} y={n.y - 11} width="22" height="22" rx="3" fill={base}>{anim}</rect>
              ) : (
                <circle key={n.id} cx={n.x} cy={n.y} r="12" fill={base}>{anim}</circle>
              );
            })}
            <text x="70" y="185" textAnchor="middle" fontSize="11.5" fill="#c3c9d4">源頭</text>
            <text x="500" y="185" textAnchor="middle" fontSize="11.5" fill="#c3c9d4">被舉報的帳戶</text>
          </svg>
          <figcaption className="mt-2 text-[12.5px] text-ink-400">
            <span className="mr-2 inline-block h-2 w-2 rounded-full" style={{ background: COLOR.fraud }} />
            異常沿著關係逐跳擴散：錢包 ● 與交易 ◆ 交替，跨越多筆中介交易才抵達被舉報的帳戶。
          </figcaption>
        </figure>

        <div className="space-y-5 text-[15px] leading-[1.8] text-ink-300">
          <p>
            <strong className="text-ink-100">關係網絡，而非孤立樣本。</strong>
            要刻畫並追溯這類攻擊，我們面對的是一張由實體與互動交織而成的關係圖；圖神經網路（GNN）正是為此而生，透過鄰域聚合在網絡上做端到端的預測。
          </p>
          <p>
            <strong className="text-ink-100">真實資料是異質、多層的。</strong>
            比特幣交易圖同時有交易與錢包兩種實體，透過「發起、付款、流向」多種有向邊串連；洗錢資金鏈可能跨越數十筆中介交易，才抵達被舉報的可疑帳戶。
          </p>
          <p>
            <strong className="text-ink-100">偵測到了，然後呢？</strong>
            端到端的分類器能標出「這筆交易可疑」，卻答不出異常從哪裡開始、沿何路徑擴散、最終是誰造成的——而這才是稽核與究責真正需要的答案。
          </p>
        </div>
      </div>
    </Section>
  );
}
