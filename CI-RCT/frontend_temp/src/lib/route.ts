import { useEffect, useState } from 'react';

// Minimal hash router: `#/` = idea page, `#/explorer?q=<preset>` = the viewer.
// Hash routing keeps GitHub Pages happy (no server-side rewrites needed).

export type Page = 'idea' | 'explorer';

/** Which view the explorer should open in (from the idea page's three questions). */
export type ExplorerPreset = 'converge' | 'chain' | 'responsibility';

export interface Route {
  page: Page;
  preset: ExplorerPreset | null;
}

const PRESETS: ReadonlySet<string> = new Set(['converge', 'chain', 'responsibility']);

export function parseHash(hash: string): Route {
  const raw = hash.replace(/^#\/?/, '');
  const [path, query = ''] = raw.split('?');
  const page: Page = path.startsWith('explorer') ? 'explorer' : 'idea';
  const q = new URLSearchParams(query).get('q');
  const preset = q && PRESETS.has(q) ? (q as ExplorerPreset) : null;
  return { page, preset };
}

export function useHashRoute(): Route {
  const [route, setRoute] = useState<Route>(() => parseHash(window.location.hash));
  useEffect(() => {
    const onChange = () => setRoute(parseHash(window.location.hash));
    window.addEventListener('hashchange', onChange);
    return () => window.removeEventListener('hashchange', onChange);
  }, []);
  return route;
}

export const explorerHref = (preset?: ExplorerPreset): string =>
  preset ? `#/explorer?q=${preset}` : '#/explorer';

export const IDEA_HREF = '#/';
