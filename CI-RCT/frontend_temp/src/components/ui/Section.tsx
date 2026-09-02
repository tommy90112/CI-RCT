import type { ReactNode } from 'react';

// Control-panel section: tracked eyebrow label, optional right-side meta, body.
interface SectionProps {
  title: ReactNode;
  meta?: ReactNode;
  description?: ReactNode;
  children: ReactNode;
}

export function Section({ title, meta, description, children }: SectionProps) {
  return (
    <section className="flex flex-col gap-2">
      <div className="flex items-baseline justify-between gap-2">
        <h3 className="eyebrow">{title}</h3>
        {meta && <span className="font-mono text-[11.5px] text-ink-400">{meta}</span>}
      </div>
      {description && <p className="-mt-1 text-[10.5px] leading-relaxed text-ink-400">{description}</p>}
      {children}
    </section>
  );
}

/** Thin divider between sections. */
export function Divider() {
  return <hr className="hairline border-t" />;
}
