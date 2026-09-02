/**
 * ExplorerApp — the CI-RCT crime-chain explainability viewer (route #/explorer).
 *
 * Layout: a fixed app bar (TopBar) on top; the rest of the viewport is a 3D
 * node-link graph (GraphCanvas). A LEFT drawer (hover/click) holds the chain
 * picker + node-display controls; a BOTTOM drawer (click) holds the L1/L2/L3
 * explainability panel. The app bar carries the dataset variant, a stepper
 * for the current chain, status chips and explicit drawer toggles.
 */
import { useCallback, useEffect, useMemo, useState } from 'react';
import type { ChangeEvent } from 'react';
import type {
  CrimeChain,
  CrimeChainData,
  CrimeChainNode,
  DataVariant,
  DisplayOptions,
} from '../types';
import type { ExplorerPreset } from '../lib/route';
import { GuideCard, guideSeen, markGuideSeen } from './GuideCard';
import {
  buildMergedGraph,
  buildResponsibilityRows,
  chainSets,
  hasPhiAsym,
  phiAsymMag,
  pivotGlobalOf,
  topRoots,
  unionChainSets,
} from '../lib/graph';
import { buildGraphView } from '../lib/neighbors';
import { useCrimeChains } from '../hooks/useCrimeChains';
import { useChainNeighbors } from '../hooks/useChainNeighbors';
import { useChainFilter } from '../hooks/useChainFilter';
import { Drawer } from '../components/Drawer';
import { GraphCanvas } from '../components/GraphCanvas';
import { ControlPanel } from '../components/ControlPanel';
import { ExplainPanel } from '../components/ExplainPanel';
import { TopBar, TOP_BAR_HEIGHT } from '../components/TopBar';

const LEFT_DRAWER_WIDTH = 340;
const BOTTOM_DRAWER_HEIGHT = 372;

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

export function ExplorerApp({ preset }: { preset: ExplorerPreset | null }) {
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
      <div className="flex h-screen w-screen items-center justify-center bg-ink-950 text-ink-300">
        <div className="flex items-center gap-3">
          <span className="relative flex h-2.5 w-2.5">
            <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-brand/50" />
            <span className="relative inline-flex h-2.5 w-2.5 rounded-full bg-brand" />
          </span>
          <span className="font-mono text-[12px]">載入犯罪鏈資料中…</span>
        </div>
      </div>
    );
  }

  if (!data) {
    return (
      <div className="flex h-screen w-screen items-center justify-center bg-ink-950 px-6">
        <div className="float w-full max-w-md p-6 text-ink-200 animate-slide-up">
          <div className="eyebrow mb-2">CI-RCT</div>
          <h1 className="text-lg font-semibold tracking-tight text-ink-100">詐欺資金鏈可解釋視覺化</h1>
          <p className="mt-2 text-[12px] leading-relaxed text-ink-300">
            找不到資料。請選取
            <span className="mx-1 font-mono text-[11px] text-brand">viz/crime_chains.json</span>
            （或相容的 crime_chains.csv）。
          </p>
          <label className="btn-primary mt-5 cursor-pointer">
            選取 crime_chains.json
            <input
              type="file"
              accept=".csv,application/json,.json"
              className="hidden"
              onChange={onFileInput}
            />
          </label>
          {error && (
            <p className="mt-4 break-words font-mono text-[11px] text-rose-300">{error}</p>
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
      preset={preset}
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
  /** Deep-link from the idea page: which of the three questions to open on. */
  preset: ExplorerPreset | null;
  onLoadFile: (file: File) => void;
  onVariantChange: (variant: DataVariant) => void;
}

/**
 * Data-present view. Lives at module scope (NOT nested in App) so it keeps a
 * stable component identity across re-renders — otherwise React would remount
 * the whole 3D graph on every state change, resetting the camera.
 */
function Viewer({ data, source, variant, dataError, preset, onLoadFile, onVariantChange }: ViewerProps) {
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

  const maxIdx = data.chains.length - 1;
  const safeIdx = clamp(selectedIdx, 0, maxIdx);
  const selectedChain = data.chains[safeIdx];

  // Search / φ-only filter shared by the app-bar stepper and the panel picker.
  const filter = useChainFilter(data, safeIdx, handleSelectedIdxChange);

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

  // Apply the idea page's deep-link once per preset: the three questions map to
  // convergence view / a single chain / the responsibility panel.
  useEffect(() => {
    if (!preset) return;
    if (preset === 'converge') {
      const top = rootGroups[0];
      if (top) handleConvergeChange(top.rootRealId);
      setLeftOpen(true);
      setBottomOpen(false);
    } else if (preset === 'responsibility') {
      setConvergeRootId(null);
      setLeftOpen(false);
      setBottomOpen(true);
    } else {
      setConvergeRootId(null);
      setLeftOpen(false);
      setBottomOpen(false);
    }
  }, [preset, rootGroups, handleConvergeChange]);

  // First-visit guide: offered only when no preset was requested.
  const [guideOpen, setGuideOpen] = useState<boolean>(() => !preset && !guideSeen());
  const dismissGuide = useCallback(() => {
    markGuideSeen();
    setGuideOpen(false);
  }, []);

  const merged = useMemo(() => buildMergedGraph(data), [data]);

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
    <div className="relative h-screen w-screen overflow-hidden bg-ink-950">
      <TopBar
        variant={variant}
        onVariantChange={onVariantChange}
        chain={selectedChain}
        selectedIdx={safeIdx}
        filter={filter}
        totalChains={data.chains.length}
        convergeCount={convergeCount}
        onExitConverge={() => handleConvergeChange(null)}
        datasetHasPhi={datasetHasPhi}
        chainHasPhi={chainHasPhi}
        leftOpen={leftOpen}
        bottomOpen={bottomOpen}
        onToggleLeft={() => setLeftOpen(o => !o)}
        onToggleBottom={() => setBottomOpen(o => !o)}
      />

      {/* Full-screen node-link graph, below the app bar */}
      <main className="absolute inset-x-0 bottom-0" style={{ top: TOP_BAR_HEIGHT }}>
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
      </main>

      {/* Left control drawer (hover or click) */}
      <Drawer
        side="left"
        open={leftOpen}
        onOpenChange={setLeftOpen}
        hoverOpen
        tab="控制台"
        size={LEFT_DRAWER_WIDTH}
        topOffset={TOP_BAR_HEIGHT}
      >
        <ControlPanel
          data={data}
          display={display}
          onDisplayChange={setDisplay}
          selectedIdx={safeIdx}
          onSelectedIdxChange={handleSelectedIdxChange}
          filter={filter}
          rootGroups={rootGroups}
          convergeRootId={convergeRootId}
          onConvergeChange={handleConvergeChange}
          dataError={dataError}
          visible={visible}
          neighborSource={view.neighborSource}
          source={source}
          onLoadFile={onLoadFile}
        />
      </Drawer>

      {guideOpen && <GuideCard onDismiss={dismissGuide} />}

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
