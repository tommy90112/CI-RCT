import { SITE_META } from './siteMeta';
import { explorerHref } from '../lib/route';

export function SiteFooter() {
  return (
    <footer className="hairline border-t">
      <div className="mx-auto flex max-w-6xl flex-col gap-3 px-6 py-10 text-[13px] text-ink-400 md:flex-row md:items-center md:justify-between">
        <div>
          <span className="font-semibold text-ink-200">CI-RCT</span> · {SITE_META.framework}
          <span className="mx-2 text-ink-600">/</span>
          {SITE_META.author} · {SITE_META.year}
        </div>
        <div className="flex flex-wrap items-center gap-4">
          <span>本站所有數值皆來自 {SITE_META.dataset} 的實際追溯輸出。</span>
          <a href={SITE_META.repoUrl} target="_blank" rel="noreferrer" className="hover:text-brand">原始碼 ↗</a>
          <a href={explorerHref()} className="hover:text-brand">探索器 →</a>
        </div>
      </div>
    </footer>
  );
}
