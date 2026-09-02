/**
 * Root: a two-page site behind a hash router.
 *   #/           idea page — the research framework, told as one scrolling story
 *   #/explorer   the interactive crime-chain explainability viewer
 */
import { useHashRoute } from './lib/route';
import { IdeaPage } from './site/IdeaPage';
import { ExplorerApp } from './explorer/ExplorerApp';

export function App() {
  const route = useHashRoute();
  if (route.page === 'explorer') return <ExplorerApp preset={route.preset} />;
  return <IdeaPage />;
}
