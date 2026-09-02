/**
 * GraphCanvas — the full-screen node-link "crime chain" graph (spec §3).
 *
 * Renders with either the 2D or 3D force-graph engine (display.dimensions), so
 * the two presentations can be compared. The selected fraud chain is coloured;
 * its 1-hop neighbours from the merged graph are shown GREYED (display.neighbors)
 * for context. Per-node id text labels are intentionally not drawn (the full id
 * is still available on hover and in the explainability panel).
 *
 * Visual encoding (spec §3.2):
 *  - node shape : wallet = circle, transaction = square (two shapes → two types)
 *  - node colour: type/fraud base — wallet 黃, transaction 藍, fraud 紅;
 *                 non-chain (neighbour) nodes are dimmed grey
 *  - node size  : nodeSize(n, sizeByPhi ? phiMax : 0)  → |φ_asym| or degree
 *  - root halo  : the chain's root cause gets a translucent violet ring
 *  - pivot      : the peak-|φ_asym| node gets an amber star/ring
 *  - edges      : chain edges coloured by signed CE (width scales with its strength), neighbours greyed
 */
import { lazy, Suspense, useCallback, useEffect, useMemo, useRef, useState } from 'react';
import * as THREE from 'three';
import { chainSets, idOf, linkKey } from '../lib/graph';
import { buildGraphView } from '../lib/neighbors';
import { GraphLegend } from './GraphLegend';
import {
  COLOR,
  linkColorByCe,
  linkWidthByCe,
  nodeBaseColor,
  nodeSize,
} from '../lib/render';
import type {
  CrimeChain,
  DisplayOptions,
  GraphLink,
  GraphNode,
  MergedGraph,
  NeighborIndex,
} from '../types';

// eslint-disable-next-line @typescript-eslint/no-explicit-any
const ForceGraph2D = lazy(() => import('react-force-graph-2d').then(m => ({ default: m.default as any })));
// eslint-disable-next-line @typescript-eslint/no-explicit-any
const ForceGraph3D = lazy(() => import('react-force-graph-3d').then(m => ({ default: m.default as any })));

const X_STEP = 80;
const Y_BAND = 70;
// react-force-graph-2d default nodeRelSize — keep custom shapes the same scale.
const NODE_REL = 4;

interface GraphCanvasProps {
  merged: MergedGraph;
  display: DisplayOptions;
  selectedChain: CrimeChain;
  selectedNodeId: number | null;
  phiMax: number;
  /** The selected chain's φ_asym pivot (peak responsibility); null when unscored. */
  pivotGlobal: number | null;
  /** Macro convergence override: highlight this union instead of the single chain. */
  highlight?: { nodeIds: Set<number>; linkKeys: Set<string> } | null;
  /** Real 1-hop neighbour overlay; null falls back to chain-union neighbours. */
  neighborIndex: NeighborIndex | null;
  onSelectNode: (id: number | null) => void;
}

export function GraphCanvas({
  merged,
  display,
  selectedChain,
  selectedNodeId,
  phiMax,
  pivotGlobal,
  highlight,
  neighborIndex,
  onSelectNode,
}: GraphCanvasProps) {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const graphRef = useRef<any>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const [dims, setDims] = useState({ width: 800, height: 600 });

  const is3D = display.dimensions === 3;
  const sel = useMemo(
    () => highlight ?? chainSets(selectedChain),
    [highlight, selectedChain],
  );
  const effPhiMax = display.sizeByPhi ? phiMax : 0;

  // Live refs so the per-frame accessor closures read current state without
  // re-instantiating the graph each render.
  const selRef = useRef(sel);
  const phiMaxRef = useRef(effPhiMax);
  const selectedNodeRef = useRef(selectedNodeId);
  const pivotRef = useRef(pivotGlobal);
  const layoutRef = useRef(display.layout);
  const heavyRef = useRef(false);
  useEffect(() => { selRef.current = sel; }, [sel]);
  useEffect(() => { phiMaxRef.current = effPhiMax; }, [effPhiMax]);
  useEffect(() => { selectedNodeRef.current = selectedNodeId; }, [selectedNodeId]);
  useEffect(() => { pivotRef.current = pivotGlobal; }, [pivotGlobal]);
  useEffect(() => { layoutRef.current = display.layout; }, [display.layout]);

  // Strongest CE over the merged graph — edge-width scale (spec §6.2).
  const ceMaxRef = useRef(1);
  useEffect(() => {
    ceMaxRef.current = merged.links.reduce((m, l) => Math.max(m, Math.abs(l.ce)), 0) || 1;
  }, [merged]);

  // Responsive sizing — fill the absolute inset-0 parent.
  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    const ro = new ResizeObserver(entries => {
      const { width, height } = entries[0].contentRect;
      if (width > 0 && height > 0) setDims({ width, height });
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  // Layout: temporal pins fx=time, fy=type-band; structure frees all axes.
  // Mutating the node objects in place is REQUIRED by the force engine.
  useEffect(() => {
    for (const n of merged.nodes) {
      if (display.layout === 'temporal') {
        n.fx = ((n.time == null ? merged.tMin - 1 : n.time) - merged.tMin) * X_STEP;
        n.fy = (n.type === 'transaction' ? Y_BAND : -Y_BAND) + ((n.id % 7) - 3) * 6;
        n.fz = undefined;
      } else {
        n.fx = undefined;
        n.fy = undefined;
        n.fz = undefined;
      }
    }
    const g = graphRef.current;
    g?.d3Force?.('charge')?.strength(display.layout === 'temporal' ? -45 : -34);
    g?.d3ReheatSimulation?.();
  }, [display.layout, merged]);

  // Selected chain (coloured) + its 1-hop neighbours (greyed), or showAll graph.
  const view = useMemo(
    () =>
      buildGraphView(merged, sel, neighborIndex, {
        showAll: display.showAll,
        neighbors: display.neighbors,
        typeFilter: display.typeFilter,
      }),
    [merged, sel, neighborIndex, display.showAll, display.neighbors, display.typeFilter],
  );
  const graphData = useMemo(() => ({ nodes: view.nodes, links: view.links }), [view]);

  // Focus the camera on the selected node (e.g. when a row is clicked in the L2
  // panel), not just enlarge it. Read graphData via a ref so this only fires on
  // selection change, not on every view rebuild.
  const graphDataRef = useRef(graphData);
  useEffect(() => { graphDataRef.current = graphData; }, [graphData]);
  useEffect(() => {
    if (selectedNodeId == null) return;
    const g = graphRef.current;
    if (!g) return;
    const node = graphDataRef.current.nodes.find(n => n.id === selectedNodeId) as
      | { x?: number; y?: number; z?: number }
      | undefined;
    if (!node) return;
    const { x = 0, y = 0, z = 0 } = node;
    if (is3D) {
      g.cameraPosition?.({ x: x + 220, y: y + 110, z: z + 220 }, { x, y, z }, 600);
    } else {
      // Pan to the node; only gently zoom in when very zoomed out — never force a
      // close zoom (that felt too tight).
      g.centerAt?.(x, y, 500);
      if ((g.zoom?.() ?? 1) < 1.4) g.zoom?.(1.4, 500);
    }
  }, [selectedNodeId, is3D]);

  // ----- per-frame accessors (read live refs) -----

  // Node colour = type / fraud (wallet 黃, transaction 藍, fraud 紅). φ_asym is
  // NOT encoded in colour (it drives size + the pivot marker) so the two node
  // types stay visually distinct.
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const nodeColor = useCallback((node: any): string => {
    const n = node as GraphNode;
    return selRef.current.nodeIds.has(n.id) ? nodeBaseColor(n) : COLOR.dim;
  }, []);

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const nodeVal = useCallback((node: any): number => {
    const n = node as GraphNode;
    const base = nodeSize(n, phiMaxRef.current);
    if (!selRef.current.nodeIds.has(n.id)) return Math.max(1.5, base * 0.5); // neighbours smaller
    return selectedNodeRef.current === n.id ? base * 1.6 : base;
  }, []);

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const linkColor = useCallback((link: any): string => {
    const l = link as GraphLink;
    return selRef.current.linkKeys.has(linkKey(idOf(l.source), idOf(l.target)))
      ? linkColorByCe(l)
      : COLOR.dim;
  }, []);

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const linkWidth = useCallback((link: any): number => {
    const l = link as GraphLink;
    return selRef.current.linkKeys.has(linkKey(idOf(l.source), idOf(l.target)))
      ? linkWidthByCe(l.ce, ceMaxRef.current)
      : 0.5;
  }, []);

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const linkParticles = useCallback((link: any): number => {
    const l = link as GraphLink;
    return selRef.current.linkKeys.has(linkKey(idOf(l.source), idOf(l.target))) ? 3 : 0;
  }, []);

  // Hover tooltip — the only place the full id surfaces (no on-canvas labels).
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const nodeLabel = useCallback((node: any): string => {
    const n = node as GraphNode;
    const pivot = pivotRef.current === n.id ? ' · ★pivot' : '';
    const role = n.is_target ? ' · target' : n.is_root ? ' · root' : '';
    const member = selRef.current.nodeIds.has(n.id) ? '' : ' · 鄰居';
    return `${n.real_id}\n${n.type} · 度數 ${n.deg}${role}${pivot}${member}${n.fraud ? ' · illicit' : ''}`;
  }, []);

  // Add the violet root halo + amber pivot ring sprites to a group (3D markers).
  const addMarkers = useCallback((group: THREE.Group, n: GraphNode) => {
    const base = nodeSize(n, phiMaxRef.current);
    if (n.is_root) {
      const halo = makeRingSprite(COLOR.root);
      const r = (base + 4) * 2.4;
      halo.scale.set(r, r, 1);
      group.add(halo);
    }
    if (pivotRef.current === n.id) {
      const halo = makeRingSprite(COLOR.pivot);
      const r = (base + 8) * 2.4;
      halo.scale.set(r, r, 1);
      group.add(halo);
    }
  }, []);

  // 3D node object: shape by type (wallet = sphere, transaction = box), coloured
  // by type/fraud, plus root/pivot markers. On heavy graphs we keep the engine's
  // default sphere (extend=true) and only add markers, to avoid building 6k meshes.
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const nodeThreeObject = useCallback((node: any) => {
    const n = node as GraphNode;
    const inChain = selRef.current.nodeIds.has(n.id);

    if (heavyRef.current) {
      if (!inChain || (!n.is_root && pivotRef.current !== n.id)) return undefined;
      const group = new THREE.Group();
      addMarkers(group, n);
      return group;
    }

    const val = Math.max(0.5, nodeVal(node));
    const r = Math.cbrt(val) * 4;
    const geom =
      n.type === 'transaction'
        ? new THREE.BoxGeometry(r * 1.7, r * 1.7, r * 1.7)
        : new THREE.SphereGeometry(r, 14, 14);
    const material = new THREE.MeshLambertMaterial({
      color: inChain ? nodeBaseColor(n) : COLOR.dim,
      transparent: true,
      opacity: inChain ? 0.96 : 0.72,
    });
    const group = new THREE.Group();
    group.add(new THREE.Mesh(geom, material));
    if (inChain) addMarkers(group, n);
    return group;
  }, [addMarkers, nodeVal]);

  // 2D node object: full custom draw so we can vary the SHAPE by type
  // (wallet = circle, transaction = square). Colour = type/fraud; neighbours dim.
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const nodeCanvasObject = useCallback((node: any, ctx: CanvasRenderingContext2D) => {
    const n = node as GraphNode;
    const inChain = selRef.current.nodeIds.has(n.id);
    const x = node.x ?? 0;
    const y = node.y ?? 0;
    const r = Math.sqrt(Math.max(0.5, nodeVal(node))) * NODE_REL;
    const selected = selectedNodeRef.current === n.id;

    // shape + fill
    ctx.beginPath();
    if (n.type === 'transaction') {
      const s = r * 1.8;
      ctx.rect(x - s / 2, y - s / 2, s, s);
    } else {
      ctx.arc(x, y, r, 0, Math.PI * 2);
    }
    ctx.globalAlpha = inChain ? 0.96 : 0.8;
    ctx.fillStyle = inChain ? nodeBaseColor(n) : COLOR.dim;
    ctx.fill();
    ctx.globalAlpha = 1;
    ctx.lineWidth = selected ? 2 : 0.7;
    ctx.strokeStyle = selected ? '#e2e8f0' : 'rgba(11,15,20,0.85)';
    ctx.stroke();

    if (!inChain) return;
    if (n.is_root) {
      ctx.beginPath();
      ctx.arc(x, y, r + 5, 0, Math.PI * 2);
      ctx.strokeStyle = COLOR.root;
      ctx.lineWidth = 2;
      ctx.stroke();
    }
    if (pivotRef.current === n.id) {
      drawStar(ctx, x, y, r + 9, COLOR.pivot);
    }
  }, [nodeVal]);

  // ----- camera / interaction (differ by engine) -----

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const handleNodeClick = useCallback(
    (node: any) => {
      const n = node as GraphNode;
      onSelectNode(n.id);
      const g = graphRef.current;
      if (!g) return;
      const { x = 0, y = 0, z = 0 } = node as { x?: number; y?: number; z?: number };
      if (is3D) {
        const distance = 120;
        g.cameraPosition({ x: x + distance, y: y + distance / 2, z: z + distance }, { x, y, z }, 800);
      } else {
        g.centerAt?.(x, y, 600);
        g.zoom?.(Math.max(g.zoom?.() ?? 1, 3), 600);
      }
    },
    [onSelectNode, is3D],
  );

  const handleEngineStop = useCallback(() => {
    const g = graphRef.current;
    if (!g) return;
    if (is3D && layoutRef.current === 'temporal') g.cameraPosition?.({ x: 0, y: 60, z: 540 });
    g.zoomToFit?.(600, 80);
  }, [is3D]);

  const zoomBy = useCallback(
    (dir: 'in' | 'out') => {
      const g = graphRef.current;
      if (!g) return;
      if (is3D) {
        const cam = g.camera?.();
        if (!cam) return;
        const f = dir === 'in' ? 0.65 : 1.5;
        g.cameraPosition(
          { x: cam.position.x * f, y: cam.position.y * f, z: cam.position.z * f },
          undefined,
          200,
        );
      } else {
        const z = g.zoom?.() ?? 1;
        g.zoom?.(z * (dir === 'in' ? 1.5 : 0.66), 200);
      }
    },
    [is3D],
  );
  const resetView = useCallback(() => graphRef.current?.zoomToFit(500, 80), []);

  // Heavy graph (e.g. 顯示全部鏈 ≈ 6k nodes): drop the per-frame costs so the
  // scene becomes static and stops repainting once settled — straight links, no
  // arrows, no animated particles, and a faster-cooling simulation.
  const heavy = view.nodes.length > 600;
  useEffect(() => { heavyRef.current = heavy; }, [heavy]);

  const shared = {
    width: dims.width,
    height: dims.height,
    graphData,
    backgroundColor: '#0b0d11',
    nodeColor,
    nodeVal,
    nodeLabel,
    linkColor,
    linkWidth,
    linkCurvature: heavy ? 0 : 0.18,
    linkDirectionalArrowLength: heavy ? 0 : 4,
    linkDirectionalArrowRelPos: 1,
    linkDirectionalParticles: display.particles && !heavy ? linkParticles : 0,
    linkDirectionalParticleWidth: 2.4,
    linkDirectionalParticleSpeed: 0.01,
    // Settle ~2.5× faster on big graphs (slightly looser layout, far less jank).
    cooldownTime: heavy ? 6000 : 15000,
    d3AlphaDecay: heavy ? 0.05 : 0.0228,
    d3VelocityDecay: heavy ? 0.55 : 0.4,
    onNodeClick: handleNodeClick,
    onEngineStop: handleEngineStop,
  };

  return (
    <div ref={containerRef} className="absolute inset-0 overflow-hidden">
      {/* Visual-encoding key — top right of the canvas (single source of truth). */}
      <GraphLegend />

      {/* Zoom / reset controls — bottom right, lifted above the bottom drawer grip */}
      <div className="float absolute bottom-12 right-3 z-10 flex flex-col overflow-hidden" role="group" aria-label="視角控制">
        <ZoomButton onClick={() => zoomBy('in')} label="放大">+</ZoomButton>
        <ZoomButton onClick={() => zoomBy('out')} label="縮小">−</ZoomButton>
        <ZoomButton onClick={resetView} label="重設視角" last>
          <svg width="13" height="13" viewBox="0 0 16 16" aria-hidden>
            <path d="M13.5 8a5.5 5.5 0 1 1-1.6-3.9" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" />
            <path d="M13.5 2.5v3.2h-3.2" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" />
          </svg>
        </ZoomButton>
      </div>

      <Suspense fallback={<div className="flex h-full items-center justify-center text-sm text-ink-400">載入圖形引擎…</div>}>
        {is3D ? (
          // eslint-disable-next-line @typescript-eslint/no-explicit-any
          <ForceGraph3D
            {...({ ref: graphRef } as any)}
            {...shared}
            nodeOpacity={0.95}
            nodeResolution={heavy ? 6 : 10}
            nodeThreeObject={nodeThreeObject}
            nodeThreeObjectExtend={heavy}
            linkOpacity={heavy ? 0.4 : 0.7}
            showNavInfo={false}
          />
        ) : (
          // eslint-disable-next-line @typescript-eslint/no-explicit-any
          <ForceGraph2D
            {...({ ref: graphRef } as any)}
            {...shared}
            nodeRelSize={NODE_REL}
            nodeCanvasObjectMode={() => 'replace'}
            nodeCanvasObject={nodeCanvasObject}
            linkDirectionalArrowColor={linkColor}
          />
        )}
      </Suspense>
    </div>
  );
}

function ZoomButton({
  onClick, label, last, children,
}: { onClick: () => void; label: string; last?: boolean; children: React.ReactNode }) {
  return (
    <button
      type="button"
      onClick={onClick}
      aria-label={label}
      title={label}
      className={`flex h-9 w-9 items-center justify-center text-[15px] font-semibold text-ink-300 transition-all duration-200 hover:bg-white/[0.06] hover:text-ink-100 active:scale-95 ${
        last ? '' : 'hairline border-b'
      }`}
    >
      {children}
    </button>
  );
}

/**
 * Build a translucent ring sprite via a canvas texture — the 3D root-cause halo.
 */
function makeRingSprite(color: string): THREE.Sprite {
  const dim = 128;
  const canvas = document.createElement('canvas');
  canvas.width = dim;
  canvas.height = dim;
  const ctx = canvas.getContext('2d');
  if (ctx) {
    ctx.strokeStyle = color;
    ctx.globalAlpha = 0.85;
    ctx.lineWidth = 9;
    ctx.beginPath();
    ctx.arc(dim / 2, dim / 2, dim / 2 - ctx.lineWidth, 0, Math.PI * 2);
    ctx.stroke();
  }
  const texture = new THREE.CanvasTexture(canvas);
  texture.minFilter = THREE.LinearFilter;
  const material = new THREE.SpriteMaterial({ map: texture, transparent: true, depthWrite: false });
  return new THREE.Sprite(material);
}

/** Draw a 5-point star outline — the 2D φ_asym pivot marker. */
function drawStar(ctx: CanvasRenderingContext2D, cx: number, cy: number, radius: number, color: string): void {
  const spikes = 5;
  const inner = radius * 0.45;
  ctx.beginPath();
  for (let i = 0; i < spikes * 2; i++) {
    const r = i % 2 === 0 ? radius : inner;
    const a = (Math.PI / spikes) * i - Math.PI / 2;
    const px = cx + Math.cos(a) * r;
    const py = cy + Math.sin(a) * r;
    if (i === 0) ctx.moveTo(px, py);
    else ctx.lineTo(px, py);
  }
  ctx.closePath();
  ctx.strokeStyle = color;
  ctx.lineWidth = 1.8;
  ctx.stroke();
}
