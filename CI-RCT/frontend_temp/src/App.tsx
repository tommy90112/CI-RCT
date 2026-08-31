/**
 * Root component for the CI-RCT crime-chain explainability viewer.
 *
 * Layout: the whole viewport is a 3D node-link graph (GraphCanvas). A LEFT
 * drawer (hover/click) holds node-display controls; a BOTTOM drawer (click)
 * holds the L1/L2/L3 explainability panel. A slim glass header floats on top
 * with the title + dataset/chain meta chips + a φ-approximation badge.
 */
import { useCallback, useMemo, useState } from 'react';
import type { ChangeEvent } from 'react';
import type {
  CrimeChain,
  CrimeChainData,
  CrimeChainNode,
  DataVariant,
  DisplayOptions,
} from './types';
import {
  buildMergedGraph,
  buildResponsibilityRows,
  chainSets,
  hasPhiAsym,
  phiAsymMag,
  pivotGlobalOf,
  topRoots,
  unionChainSets,
} from './lib/graph';
import { buildGraphView } from './lib/neighbors';
import { useCrimeChains } from './hooks/useCrimeChains';
import { useChainNeighbors } from './hooks/useChainNeighbors';
import { Drawer } from './components/Drawer';
import { PhiAsym } from './components/Phi';
import { GraphCanvas } from './components/GraphCanvas';
import { ControlPanel } from './components/ControlPanel';
import { ExplainPanel } from './components/ExplainPanel';

const LEFT_DRAWER_WIDTH = 340;
const BOTTOM_DRAWER_HEIGHT = 340;

const DEFAULT_DISPLAY: DisplayOptions = {
  dimensions: 2,
  layout: 'structure',
  showAll: false,
  neighbors: true,
  particles: true,
  sizeByPhi: true,
  typeFilter: new Set<string>(),
};

const clamp = (value: number, min: number, max: number): number =>
  Math.max(min, Math.min(max, value));

/** Resolve the chain's traced root cause (stored last in [target..root] order). */
const rootGlobalOf = (chain: CrimeChain): number =>
  chain.nodes[chain.nodes.length - 1]?.global ?? chain.nodes[0]?.global ?? -1;

/** Max |φ_asym| across a chain's nodes — safe scale for sizing. */
const phiMaxOfChain = (chain: CrimeChain): number =>
  chain.nodes.reduce((max, n) => Math.max(max, phiAsymMag(n)), 0);

export function App() {
  const { data, loading, error, source, variant, loadFromFile, loadVariant } = useCrimeChains();

  const onFileInput = useCallback(
    (e: ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0];
      if (file) loadFromFile(file);
    },
    [loadFromFile],
  );

  // Only block the whole screen on the FIRST load. Once data exists, keep the
  // viewer mounted during reloads (e.g. variant switches) so a failed switch
  // stays on the current view with its drawers open and surfaces the error.
  if (loading && !data) {
    return (
      <div className="w-screen h-screen flex items-center justify-center bg-slate-950 text-slate-300">
        <div className="font-mono text-sm animate-pulse">載入犯罪鏈資料中…</div>
      </div>
    );
  }

  if (!data) {
    return (
      <div className="w-screen h-screen flex items-center justify-center bg-slate-950 px-6">
        <div className="max-w-md w-full bg-slate-900/85 backdrop-blur-sm border border-slate-700/60 rounded-xl shadow-xl p-6 text-slate-300">
          <h1 className="text-sky-400 text-lg font-semibold mb-2">
            CI-RCT · 詐欺資金鏈可解釋視覺化
          </h1>
          <p className="text-xs text-slate-400 mb-4">
            找不到資料。請選取
            <span className="font-mono text-[11px] text-sky-300"> viz/crime_chains.json </span>
            (或相容的 crime_chains.csv)。
          </p>
          <label className="inline-flex items-center gap-2 cursor-pointer rounded-lg bg-sky-600/90 hover:bg-sky-500 px-3 py-2 text-xs font-medium text-white transition-colors">
            選取 crime_chains.json
            <input
              type="file"
              accept=".csv,application/json,.json"
              className="hidden"
              onChange={onFileInput}
            />
          </label>
          {error && (
            <p className="mt-4 text-[11px] font-mono text-rose-400 break-words">{error}</p>
          )}
        </div>
      </div>
    );
  }

  return (
    <Viewer
      key={variant}
      data={data}
      source={source}
      variant={variant}
      dataError={error}
      onLoadFile={loadFromFile}
      onVariantChange={loadVariant}
    />
  );
}

interface ViewerProps {
  data: CrimeChainData;
  source: string;
  variant: DataVariant;
  dataError: string | null;
  onLoadFile: (file: File) => void;
  onVariantChange: (variant: DataVariant) => void;
}

/**
 * Data-present view. Lives at module scope (NOT nested in App) so it keeps a
 * stable component identity across re-renders — otherwise React would remount
 * the whole 3D graph on every state change, resetting the camera.
 */
function Viewer({ data, source, variant, dataError, onLoadFile, onVariantChange }: ViewerProps) {
  const { index: neighborIndex } = useChainNeighbors();
  const [display, setDisplay] = useState<DisplayOptions>(DEFAULT_DISPLAY);
  // Default to a SHORT, true-positive transaction-target chain so the φ_asym
  // pivot is visible at a glance (φ_asym only exists for transaction targets;
  // a short depth keeps the whole L2 list — incl. the pivot — on screen).
  const firstTxIdx = useMemo(() => {
    const hasPivot = (c: CrimeChain): boolean => c.nodes.some(n => n.phi_asym != null);
    const ideal = data.chains.findIndex(
      c => c.nodes[0]?.type === 'transaction' && c.is_true_positive && c.depth >= 2 && c.depth <= 4 && hasPivot(c),
    );
    if (ideal >= 0) return ideal;
    const anyTx = data.chains.findIndex(c => c.nodes[0]?.type === 'transaction');
    return anyTx >= 0 ? anyTx : 0;
  }, [data]);
  const [selectedIdx, setSelectedIdx] = useState(firstTxIdx);
  const [selectedNodeId, setSelectedNodeId] = useState<number | null>(null);
  const [convergeRootId, setConvergeRootId] = useState<string | null>(null);
  const [leftOpen, setLeftOpen] = useState(false);
  const [bottomOpen, setBottomOpen] = useState(false);

  // Top fan-in roots (macro convergence, spec §5).
  const rootGroups = useMemo(() => topRoots(data), [data]);

  // Changing the active chain resets node selection and exits convergence view.
  const handleSelectedIdxChange = useCallback((i: number) => {
    setSelectedIdx(i);
    setSelectedNodeId(null);
    setConvergeRootId(null);
  }, []);

  // Enter/exit the macro convergence view; entering jumps to a representative chain.
  const handleConvergeChange = useCallback(
    (rootRealId: string | null) => {
      setConvergeRootId(rootRealId);
      setSelectedNodeId(null);
      if (rootRealId) {
        const i = data.chains.findIndex(c => c.root_real_id === rootRealId);
        if (i >= 0) setSelectedIdx(i);
      }
    },
    [data],
  );

  // Clicking a node in the graph drives L3 and reveals the explain drawer.
  const handleSelectNode = useCallback((id: number | null) => {
    setSelectedNodeId(id);
    if (id != null) setBottomOpen(true);
  }, []);

  // Picking a step in the L2 waterfall also drives L3.
  const handleSelectGlobal = useCallback((global: number) => {
    setSelectedNodeId(global);
  }, []);

  const merged = useMemo(() => buildMergedGraph(data), [data]);

  const maxIdx = data.chains.length - 1;
  const safeIdx = clamp(selectedIdx, 0, maxIdx);
  const selectedChain = data.chains[safeIdx];

  const rows = useMemo(() => buildResponsibilityRows(selectedChain), [selectedChain]);
  const pivotGlobal = useMemo(() => pivotGlobalOf(selectedChain), [selectedChain]);

  // φ_asym scale: prefer the merged graph max, fall back to the selected chain's
  // so node sizing/colour still reads when most chains are unscored.
  const phiMax = useMemo(() => {
    const fromGraph = merged.nodes.reduce((max, n) => Math.max(max, n.phiAsym), 0);
    return fromGraph > 0 ? fromGraph : phiMaxOfChain(selectedChain);
  }, [merged, selectedChain]);

  // φ_asym only exists for transaction-target chains; flag wallet-target chains.
  const datasetHasPhi = useMemo(() => hasPhiAsym(data), [data]);
  const chainHasPhi = pivotGlobal != null;

  // Macro convergence: highlight the union of all chains sharing the chosen root.
  const convergeHighlight = useMemo(
    () =>
      convergeRootId
        ? unionChainSets(data.chains.filter(c => c.root_real_id === convergeRootId))
        : null,
    [convergeRootId, data],
  );
  const convergeCount = convergeRootId
    ? rootGroups.find(g => g.rootRealId === convergeRootId)?.count ?? 0
    : 0;

  // Same view GraphCanvas renders — drives the control panel counts + source tag.
  const view = useMemo(() => {
    const sel = chainSets(selectedChain);
    return buildGraphView(merged, sel, neighborIndex, {
      showAll: display.showAll,
      neighbors: display.neighbors,
      typeFilter: display.typeFilter,
    });
  }, [merged, selectedChain, neighborIndex, display.showAll, display.neighbors, display.typeFilter]);
  const visible = { nodes: view.nodes.length, links: view.links.length };

  // L3 node = explicitly selected, else the φ_asym pivot (the node L3 explains),
  // falling back to the chain's root cause when nothing was scored.
  const rootGlobal = rootGlobalOf(selectedChain);
  const resolvedGlobal = selectedNodeId ?? pivotGlobal ?? rootGlobal;
  const selectedNode: CrimeChainNode | null =
    selectedChain.nodes.find(n => n.global === resolvedGlobal) ?? null;

  return (
    <div className="relative w-screen h-screen overflow-hidden bg-slate-950">
      {/* Top header bar */}
      <header className="absolute top-0 inset-x-0 z-30 flex items-center gap-3 px-4 h-12 bg-gradient-to-r from-slate-950/95 via-slate-900/90 to-slate-950/80 backdrop-blur-xl border-b border-slate-700/50 shadow-xl shadow-black/30">
        <div className="flex items-center gap-2 whitespace-nowrap">
          <span className="relative flex h-2.5 w-2.5">
            <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-sky-400/60" />
            <span className="relative inline-flex h-2.5 w-2.5 rounded-full bg-sky-400" />
          </span>
          <h1 className="bg-gradient-to-r from-sky-300 to-indigo-300 bg-clip-text text-sm font-bold tracking-tight text-transparent">
            CI-RCT
          </h1>
          <span className="text-sm font-medium text-slate-300">詐欺資金鏈可解釋視覺化</span>
        </div>
        {convergeRootId && (
          <button
            type="button"
            onClick={() => handleConvergeChange(null)}
            className="ml-2 inline-flex items-center gap-1.5 rounded-full border border-rose-500/40 bg-rose-500/10 px-2.5 py-0.5 text-[11px] font-medium text-rose-300 hover:bg-rose-500/20"
            title="退出收斂視圖"
          >
            收斂視圖 · {convergeCount} 條 → 源頭 ✕
          </button>
        )}
        <span className="ml-auto">
          {!datasetHasPhi ? (
            <span className="rounded-full border border-amber-500/40 bg-amber-500/10 px-2.5 py-0.5 text-[11px] font-medium text-amber-300">
              此資料無 <PhiAsym />
            </span>
          ) : !chainHasPhi ? (
            <span className="rounded-full border border-amber-500/40 bg-amber-500/10 px-2.5 py-0.5 text-[11px] font-medium text-amber-300">
              此鏈為 wallet-target(無 <PhiAsym />,僅 CE)
            </span>
          ) : (
            <span className="rounded-full border border-emerald-500/40 bg-emerald-500/10 px-2.5 py-0.5 text-[11px] font-medium text-emerald-300">
              <PhiAsym /> ✓ pivot 已標示
            </span>
          )}
        </span>
      </header>

      {/* Full-screen node-link graph */}
      <div className="absolute inset-0">
        <GraphCanvas
          merged={merged}
          display={display}
          selectedChain={selectedChain}
          selectedNodeId={selectedNodeId}
          phiMax={phiMax}
          pivotGlobal={pivotGlobal}
          highlight={convergeHighlight}
          neighborIndex={neighborIndex}
          onSelectNode={handleSelectNode}
        />
      </div>

      {/* Left control drawer (hover or click) */}
      <Drawer
        side="left"
        open={leftOpen}
        onOpenChange={setLeftOpen}
        hoverOpen
        tab="控制台"
        size={LEFT_DRAWER_WIDTH}
      >
        <ControlPanel
          data={data}
          display={display}
          onDisplayChange={setDisplay}
          selectedIdx={safeIdx}
          onSelectedIdxChange={handleSelectedIdxChange}
          rootGroups={rootGroups}
          convergeRootId={convergeRootId}
          onConvergeChange={handleConvergeChange}
          variant={variant}
          onVariantChange={onVariantChange}
          dataError={dataError}
          visible={visible}
          neighborSource={view.neighborSource}
          source={source}
          onLoadFile={onLoadFile}
        />
      </Drawer>

      {/* Bottom explainability drawer (hover to peek, click to pin) */}
      <Drawer
        side="bottom"
        open={bottomOpen}
        onOpenChange={setBottomOpen}
        hoverOpen
        tab="可解釋性面板"
        size={BOTTOM_DRAWER_HEIGHT}
      >
        <ExplainPanel
          chain={selectedChain}
          rows={rows}
          selectedNode={selectedNode}
          selectedGlobal={resolvedGlobal}
          onSelectGlobal={handleSelectGlobal}
        />
      </Drawer>
    </div>
  );
}
