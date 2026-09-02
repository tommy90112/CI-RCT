import { useEffect, useState } from 'react';
import { explorerHref } from '../lib/route';
import { SECTIONS } from './IdeaPage';

/** Fixed top nav: brand · section anchors (with the active one marked) · primary CTA. */
export function SiteNav() {
  const [scrolled, setScrolled] = useState(false);
  const [active, setActive] = useState<string>('');
  useEffect(() => {
    const on = () => setScrolled(window.scrollY > 24);
    on();
    window.addEventListener('scroll', on, { passive: true });
    return () => window.removeEventListener('scroll', on);
  }, []);
  // Track which section is in view so the nav doubles as a progress indicator.
  useEffect(() => {
    const els = SECTIONS.map(s => document.getElementById(s.id)).filter((e): e is HTMLElement => !!e);
    const io = new IntersectionObserver(
      entries => {
        const hit = entries.filter(e => e.isIntersecting).sort((a, b) => b.intersectionRatio - a.intersectionRatio)[0];
        if (hit) setActive(hit.target.id);
      },
      { rootMargin: '-30% 0px -55% 0px', threshold: [0, 0.2, 0.5] },
    );
    els.forEach(el => io.observe(el));
    return () => io.disconnect();
  }, []);
  const go = (id: string) => (e: React.MouseEvent) => {
    e.preventDefault();
    document.getElementById(id)?.scrollIntoView({ behavior: 'smooth', block: 'start' });
  };
  return (
    <header
      className={`fixed inset-x-0 top-0 z-30 transition-all duration-300 ${
        scrolled ? 'panel hairline border-b' : 'bg-transparent'
      }`}
    >
      <div className="mx-auto flex h-16 max-w-6xl items-center gap-6 px-6">
        <a href="#/" className="flex items-center gap-2.5">
          <Mark />
          <span className="text-[15px] font-bold tracking-tight text-ink-100">CI-RCT</span>
        </a>
        <nav className="hidden items-center gap-0.5 md:flex" aria-label="章節">
          {SECTIONS.map(s => (
            <a
              key={s.id}
              href="#/"
              onClick={go(s.id)}
              aria-current={active === s.id ? 'location' : undefined}
              className={`rounded-md px-3 py-1.5 text-[13.5px] transition-colors hover:bg-white/[0.05] hover:text-ink-100 ${
                active === s.id ? 'text-brand' : 'text-ink-300'
              }`}
            >
              {s.label}
            </a>
          ))}
        </nav>
        <a href={explorerHref()} className="btn-primary ml-auto">
          開始探索 →
        </a>
      </div>
    </header>
  );
}

function Mark() {
  return (
    <svg width="28" height="28" viewBox="0 0 32 32" aria-hidden>
      <rect width="32" height="32" rx="8" fill="#181c24" />
      <rect width="32" height="32" rx="8" fill="none" stroke="rgba(255,255,255,0.06)" />
      <circle cx="11" cy="21" r="4.5" fill="#f2b53a" />
      <rect x="17" y="7" width="9" height="9" rx="1.5" fill="#ff5470" />
      <path d="M14.5 18 18 15.5" stroke="#98a1b3" strokeWidth="1.6" strokeLinecap="round" />
    </svg>
  );
}
