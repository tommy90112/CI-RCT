import { SITE_META } from '../siteMeta';
import { DEMO_CHAIN } from '../demoChain';
import { explorerHref } from '../../lib/route';
import { COLOR } from '../../lib/render';
import { SECTIONS } from '../IdeaPage';

// One line per story beat — the roadmap strip under the hero doubles as a
// table of contents so the page never opens on empty space.
const ROADMAP: Record<string, string> = {
  problem: '異常是路徑，不是點',
  gaps: '現有方法答不出的三件事',
  framework: '四模組，一條管線',
  intervention: '切斷一條邊，量因果效應',
  tracing: '逐跳往上游，停在根因',
  explanations: '被解釋的 vs 該負責的',
  explore: '三個問題，交給探索器',
};

export function Hero() {
  const meta = [
    ['作者', SITE_META.author],
    ['指導教授', SITE_META.advisor],
    ['系所', SITE_META.department],
    ['年份', SITE_META.year],
  ].filter(([, v]) => v);

  return (
    <section className="relative overflow-hidden pb-12 pt-28 md:pt-32">
      <Backdrop />
      <div className="relative mx-auto max-w-6xl px-6">
        <div className="grid items-center gap-10 lg:grid-cols-[1.1fr_1fr]">
          <div>
            <div className="eyebrow mb-5 text-ink-400">{SITE_META.framework}</div>
            <h1 className="text-[clamp(36px,5.2vw,58px)] font-bold leading-[1.1] tracking-tight text-ink-100 [text-wrap:balance]">
              {SITE_META.title}
            </h1>
            <p className="mt-5 max-w-xl text-[17px] leading-[1.75] text-ink-300 [text-wrap:pretty]">
              {SITE_META.subtitle}
            </p>
            <div className="mt-7 flex flex-wrap items-center gap-3">
              <a href={explorerHref()} className="btn-primary px-5 py-2.5 text-[14px]">開始探索 →</a>
              <a
                href="#/"
                onClick={e => { e.preventDefault(); document.getElementById('problem')?.scrollIntoView({ behavior: 'smooth' }); }}
                className="btn-ghost px-5 py-2.5 text-[14px]"
              >
                先看想法 ↓
              </a>
            </div>
            {meta.length > 0 && (
              <dl className="hairline mt-8 flex flex-wrap gap-x-8 gap-y-3 border-t pt-5">
                {meta.map(([k, v]) => (
                  <div key={k}>
                    <dt className="text-[11.5px] tracking-[0.12em] text-ink-500">{k}</dt>
                    <dd className="mt-0.5 text-[14px] font-medium text-ink-100">{v}</dd>
                  </div>
                ))}
              </dl>
            )}
          </div>
          <ChainGlyph />
        </div>

        {/* Roadmap strip — what this page will walk through, in order. */}
        <ol className="mt-12 grid gap-px overflow-hidden rounded-xl bg-white/[0.06] ring-1 ring-white/[0.06] sm:grid-cols-2 lg:grid-cols-7">
          {SECTIONS.map((s, i) => (
            <li key={s.id}>
              <a
                href="#/"
                onClick={e => { e.preventDefault(); document.getElementById(s.id)?.scrollIntoView({ behavior: 'smooth' }); }}
                className="flex h-full flex-col gap-1.5 bg-ink-900/90 px-4 py-3.5 transition-colors hover:bg-ink-800"
              >
                <span className="font-mono text-[11.5px] text-brand">0{i + 1}</span>
                <span className="text-[14px] font-semibold text-ink-100">{s.label}</span>
                <span className="text-[12.5px] leading-snug text-ink-400">{ROADMAP[s.id]}</span>
              </a>
            </li>
          ))}
        </ol>
      </div>
    </section>
  );
}

/** Soft radial glow behind the hero — one hue, no gradient banding. */
function Backdrop() {
  return (
    <div aria-hidden className="pointer-events-none absolute inset-0">
      <div className="dot-grid absolute inset-0" />
      <div
        className="absolute inset-0"
        style={{ background: 'radial-gradient(60% 50% at 75% 30%, rgba(79,209,197,0.10) 0%, rgba(79,209,197,0) 70%)' }}
      />
    </div>
  );
}

/** The real demo chain, upstream → downstream, with a money-flow particle. */
function ChainGlyph() {
  const W = 520, H = 300;
  const xs = [60, 200, 340, 470];
  const ys = [200, 90, 190, 110];
  const path = xs.map((x, i) => `${i === 0 ? 'M' : 'L'} ${x} ${ys[i]}`).join(' ');
  return (
    <figure className="float relative p-4">
      <svg viewBox={`0 0 ${W} ${H}`} className="w-full" role="img" aria-label="一條真實追溯鏈：根因錢包 → 交易 → 錢包 → 目標交易">
        <path d={path} fill="none" stroke={COLOR.cePos} strokeWidth="2.5" strokeLinejoin="round" opacity="0.9" />
        <circle r="4" fill="#e7eaf0">
          <animateMotion dur="4s" repeatCount="indefinite" path={path} />
        </circle>
        {DEMO_CHAIN.map((n, i) => {
          const x = xs[i], y = ys[i];
          const isTx = n.type === 'transaction';
          return (
            <g key={n.id}>
              {n.role === 'root' && <circle cx={x} cy={y} r="24" fill="none" stroke={COLOR.root} strokeWidth="2.5" />}
              {n.role === 'pivot' && <Star cx={x} cy={y} r={30} color={COLOR.pivot} />}
              {isTx
                ? <rect x={x - 15} y={y - 15} width="30" height="30" rx="4" fill={COLOR.fraud} />
                : <circle cx={x} cy={y} r="16" fill={COLOR.fraud} />}
              <text x={x} y={y + 44} textAnchor="middle" fontSize="12" fill="#c3c9d4" fontFamily="Geist Mono, monospace">
                {n.id.length > 12 ? `${n.id.slice(0, 6)}…${n.id.slice(-3)}` : n.id}
              </text>
              <text x={x} y={y + 60} textAnchor="middle" fontSize="11" fill="#98a1b3">
                {n.role === 'root' ? '根因 root' : n.role === 'pivot' ? '★ 元兇 pivot' : n.role === 'target' ? '⚑ 目標 target' : n.type}
              </text>
            </g>
          );
        })}
      </svg>
      <figcaption className="mt-1 flex items-center justify-between text-[12px] text-ink-400">
        <span>金流方向 →</span>
        <span>Elliptic++ 實際追溯結果 · 鏈 #33 · CE 追路徑、φ 釘元兇</span>
      </figcaption>
    </figure>
  );
}

function Star({ cx, cy, r, color }: { cx: number; cy: number; r: number; color: string }) {
  const pts: string[] = [];
  for (let i = 0; i < 10; i++) {
    const rad = i % 2 === 0 ? r : r * 0.45;
    const a = (Math.PI / 5) * i - Math.PI / 2;
    pts.push(`${cx + Math.cos(a) * rad},${cy + Math.sin(a) * rad}`);
  }
  return <polygon points={pts.join(' ')} fill="none" stroke={color} strokeWidth="2.2" strokeLinejoin="round" />;
}
