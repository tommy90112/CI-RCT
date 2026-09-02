import type { ReactNode } from 'react';
import { useReveal } from './useReveal';

// Shared scaffolding for one story beat: eyebrow, headline, lede, then the body.
// Alternating tones and optically heavier bottom padding keep the page from
// reading as a stack of identical blocks.
interface SectionProps {
  id: string;
  eyebrow: string;
  title: ReactNode;
  lede?: ReactNode;
  children: ReactNode;
  /** Optional right-aligned aside next to the headline (a fact, a note). */
  aside?: ReactNode;
  tone?: 'base' | 'alt';
}

export function Section({ id, eyebrow, title, lede, children, aside, tone = 'base' }: SectionProps) {
  const bodyRef = useReveal<HTMLDivElement>();
  return (
    <section
      id={id}
      className={`hairline scroll-mt-20 border-t pb-20 pt-16 md:pb-24 md:pt-20 ${tone === 'alt' ? 'bg-ink-900/40' : ''}`}
    >
      <div className="mx-auto max-w-6xl px-6">
        <div className="grid gap-8 lg:grid-cols-[minmax(0,2fr)_minmax(0,1fr)] lg:items-end">
          <div>
            <div className="eyebrow mb-3 text-brand">{eyebrow}</div>
            <h2 className="text-[clamp(28px,3.6vw,42px)] font-bold leading-[1.15] tracking-tight text-ink-100 [text-wrap:balance]">
              {title}
            </h2>
            {lede && <p className="mt-4 text-[16px] leading-[1.75] text-ink-300 [text-wrap:pretty]">{lede}</p>}
          </div>
          {aside && <div className="lg:pb-1">{aside}</div>}
        </div>
        <div ref={bodyRef} data-reveal className="mt-10">{children}</div>
      </div>
    </section>
  );
}

/** Small labelled fact used in section asides and side panels. */
export function Fact({ label, value, hint }: { label: string; value: ReactNode; hint?: ReactNode }) {
  return (
    <div className="rounded-lg bg-ink-800/60 px-4 py-3 ring-1 ring-white/[0.05]">
      <div className="text-[11.5px] tracking-[0.08em] text-ink-400">{label}</div>
      <div className="mt-1 text-[15px] font-semibold text-ink-100">{value}</div>
      {hint && <div className="mt-1 text-[12.5px] leading-relaxed text-ink-400">{hint}</div>}
    </div>
  );
}

/** "How to read this" panel for the demos: a side card, or a horizontal strip below a full-width demo. */
export function ReadingGuide({ title, items, horizontal }: { title: string; items: { k: ReactNode; v: ReactNode }[]; horizontal?: boolean }) {
  if (horizontal) {
    return (
      <aside className="hairline mt-5 border-t pt-5">
        <div className="eyebrow mb-3">{title}</div>
        <dl className="grid gap-6 md:grid-cols-3">
          {items.map((it, i) => (
            <div key={i} className="border-l-2 border-brand/40 pl-4">
              <dt className="text-[14.5px] font-semibold text-ink-100">{it.k}</dt>
              <dd className="mt-1 text-[14px] leading-relaxed text-ink-300">{it.v}</dd>
            </div>
          ))}
        </dl>
      </aside>
    );
  }
  return (
    <aside className="flex flex-col gap-3 rounded-xl bg-ink-800/50 p-5 ring-1 ring-white/[0.05]">
      <div className="eyebrow">{title}</div>
      <dl className="flex flex-col gap-3">
        {items.map((it, i) => (
          <div key={i}>
            <dt className="text-[13.5px] font-semibold text-ink-100">{it.k}</dt>
            <dd className="mt-0.5 text-[13.5px] leading-relaxed text-ink-300">{it.v}</dd>
          </div>
        ))}
      </dl>
    </aside>
  );
}
