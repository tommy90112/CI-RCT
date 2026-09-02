/**
 * IdeaPage — the research framework told as one scrolling story (route #/).
 * No detection metrics: every number on this page is a descriptive fact about
 * what the tracer produced, computed live from crime_chains.json.
 */
import { SiteNav } from './SiteNav';
import { Hero } from './sections/Hero';
import { Spread } from './sections/Spread';
import { Gaps } from './sections/Gaps';
import { Framework } from './sections/Framework';
import { Intervention } from './sections/Intervention';
import { Tracing } from './sections/Tracing';
import { TwoExplanations } from './sections/TwoExplanations';
import { Questions } from './sections/Questions';
import { SiteFooter } from './SiteFooter';
import { useChainStats } from '../hooks/useChainStats';

export const SECTIONS = [
  { id: 'problem', label: '問題' },
  { id: 'gaps', label: '缺口' },
  { id: 'framework', label: '框架' },
  { id: 'intervention', label: '干預' },
  { id: 'tracing', label: '回溯' },
  { id: 'explanations', label: '解釋' },
  { id: 'explore', label: '開始探索' },
] as const;

export function IdeaPage() {
  const { stats } = useChainStats();
  return (
    <div className="site min-h-dvh bg-ink-950 text-ink-200">
      <a href="#main" className="skip-link">跳至主內容</a>
      <SiteNav />
      <main id="main">
        <Hero />
        <Spread tone="alt" />
        <Gaps />
        <Framework tone="alt" />
        <Intervention />
        <Tracing tone="alt" />
        <TwoExplanations stats={stats} />
        <Questions stats={stats} tone="alt" />
      </main>
      <SiteFooter />
    </div>
  );
}
