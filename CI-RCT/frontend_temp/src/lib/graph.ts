/**
 * Pure data helpers: validate crime-chain JSON, merge chains into a node-link
 * graph (shared real entities collapse so laundering hubs emerge), and build
 * the L2 dual-column rows (CE 追路徑 vs φ_asym 釘元兇 — descriptive, NOT
 * cumulative; see spec §7).
 */
import type {
  CrimeChain,
  CrimeChainData,
  CrimeChainNode,
  GraphLink,
  GraphNode,
  MergedGraph,
  ResponsibilityRow,
} from '../types';

/** Narrow untrusted JSON into CrimeChainData, or throw a clear error. */
export function parseCrimeChainData(raw: unknown): CrimeChainData {
  if (typeof raw !== 'object' || raw === null || !Array.isArray((raw as CrimeChainData).chains)) {
    throw new Error('JSON 缺少 chains 陣列 — 這不是 crime_chains 格式。');
  }
  const data = raw as CrimeChainData;
  if (data.chains.length === 0) throw new Error('chains 是空的(沒有可顯示的犯罪鏈)。');
  for (const c of data.chains) {
    if (!Array.isArray(c.nodes) || c.nodes.length === 0) {
      throw new Error('某條鏈缺少 nodes。');
    }
  }
  return data;
}

/** Key for a directed parent→child edge. */
export const linkKey = (source: number, target: number): string => `${source}->${target}`;

/**
 * |asymmetric φ| magnitude — the L1 spotlight signal. 0 when φ_asym is absent
 * (wallet-target chains), so those nodes simply get no spotlight rather than a
 * misleading fallback value.
 */
export const phiAsymMag = (node: { phi_asym?: number | null }): number =>
  node.phi_asym != null ? Math.abs(node.phi_asym) : 0;

/** True iff any chain carries an asymmetric φ (i.e. has a transaction-target chain). */
export function hasPhiAsym(data: CrimeChainData): boolean {
  return data.chains.some(c => c.nodes.some(n => n.phi_asym != null));
}

/** The chain's pivot = node with peak |φ_asym|; null when no node was scored. */
export function pivotGlobalOf(chain: CrimeChain): number | null {
  let best: CrimeChainNode | null = null;
  let bestMag = 0;
  for (const n of chain.nodes) {
    const mag = phiAsymMag(n);
    if (mag > bestMag) {
      bestMag = mag;
      best = n;
    }
  }
  return best ? best.global : null;
}

/**
 * Merge all chains into a single graph. Nodes are deduped by `global`, so a
 * wallet/tx reused across many chains becomes one node whose degree reflects
 * the reuse. `is_root` marks any node traced as a chain's root cause; `phiAsym`
 * keeps the max |φ_asym| responsibility seen across chains.
 */
export function buildMergedGraph(data: CrimeChainData): MergedGraph {
  const nodeById = new Map<number, GraphNode>();
  const linkBy = new Map<string, GraphLink & { _deg: number }>();
  const times: number[] = [];

  for (const chain of data.chains) {
    const rootGlobal = chain.nodes[chain.nodes.length - 1]?.global;
    for (let i = 1; i < chain.nodes.length; i++) {
      const parent = chain.nodes[i];
      const child = chain.nodes[i - 1];
      const key = linkKey(parent.global, child.global);
      const existing = linkBy.get(key);
      if (existing) existing._deg += 1;
      else linkBy.set(key, { source: parent.global, target: child.global, ce: parent.ce ?? 0, _deg: 1 });
    }
    for (const n of chain.nodes) {
      if (n.time != null) times.push(n.time);
      const mag = phiAsymMag(n);
      const existing = nodeById.get(n.global);
      if (!existing) {
        nodeById.set(n.global, {
          id: n.global,
          type: n.type,
          real_id: n.real_id,
          time: n.time,
          fraud: n.fraud,
          is_target: Boolean(n.is_target),
          is_root: n.global === rootGlobal,
          phiAsym: mag,
          deg: 0,
        });
      } else {
        if (n.is_target) existing.is_target = true;
        if (n.global === rootGlobal) existing.is_root = true;
        existing.phiAsym = Math.max(existing.phiAsym, mag);
      }
    }
  }

  for (const l of linkBy.values()) {
    const s = nodeById.get(l.source);
    const t = nodeById.get(l.target);
    if (s) s.deg += l._deg;
    if (t) t.deg += l._deg;
  }

  const nodes = [...nodeById.values()];
  const links: GraphLink[] = [...linkBy.values()].map(({ source, target, ce }) => ({ source, target, ce }));
  const tMin = times.length ? Math.min(...times) : 0;
  const tMax = times.length ? Math.max(...times) : 0;
  return { nodes, links, tMin, tMax };
}

/** The set of node ids and link keys belonging to one chain (for highlighting). */
export function chainSets(chain: CrimeChain): { nodeIds: Set<number>; linkKeys: Set<string> } {
  return unionChainSets([chain]);
}

/** Union of several chains' node ids + link keys — the macro convergence highlight. */
export function unionChainSets(chains: CrimeChain[]): { nodeIds: Set<number>; linkKeys: Set<string> } {
  const nodeIds = new Set<number>();
  const linkKeys = new Set<string>();
  for (const chain of chains) {
    chain.nodes.forEach(n => nodeIds.add(n.global));
    for (let i = 1; i < chain.nodes.length; i++) {
      linkKeys.add(linkKey(chain.nodes[i].global, chain.nodes[i - 1].global));
    }
  }
  return { nodeIds, linkKeys };
}

/** A traced root cause shared by ≥1 chains — the hub of a fan-in convergence. */
export interface RootGroup {
  rootRealId: string;
  rootType: string;
  rootGlobal: number;
  isFraud: boolean;
  count: number;
  chainIdxs: number[];
}

/**
 * Roots ranked by how many chains converge into them (spec §5 macro view).
 * Only multi-chain roots are returned (count ≥ 2) — single chains aren't fan-ins.
 */
export function topRoots(data: CrimeChainData, k = 15): RootGroup[] {
  const byRoot = new Map<string, RootGroup>();
  data.chains.forEach((c, i) => {
    const last = c.nodes[c.nodes.length - 1];
    if (!last) return;
    const g = byRoot.get(c.root_real_id);
    if (g) {
      g.count += 1;
      g.chainIdxs.push(i);
    } else {
      byRoot.set(c.root_real_id, {
        rootRealId: c.root_real_id,
        rootType: c.root_type,
        rootGlobal: last.global,
        isFraud: c.root_is_fraud,
        count: 1,
        chainIdxs: [i],
      });
    }
  });
  return [...byRoot.values()]
    .filter(g => g.count >= 2)
    .sort((a, b) => b.count - a.count)
    .slice(0, k);
}

/**
 * react-force-graph mutates link.source/target from id → node object after the
 * first tick, so accept either when reading an endpoint id.
 */
export const idOf = (e: number | { id: number }): number =>
  typeof e === 'object' ? e.id : e;

/** Default cap on how many 1-hop neighbours to pull in (hub nodes can be huge). */
export const NEIGHBOR_CAP = 250;

export interface VisibleOptions {
  /** Show the whole merged graph (laundering network). */
  showAll: boolean;
  /** Add the selected chain's 1-hop neighbours (greyed in the UI). */
  neighbors: boolean;
  /** Only keep nodes whose type is in this set; empty = all types. */
  typeFilter: Set<string>;
  neighborCap?: number;
}

/**
 * Compute the visible subgraph for a selected chain. Shared by GraphCanvas
 * (what to render) and App (the node/edge counts shown in the control panel) so
 * they never drift. `capped` is true when the neighbour cap truncated the set.
 */
export function visibleSubgraph(
  merged: MergedGraph,
  sel: { nodeIds: Set<number>; linkKeys: Set<string> },
  opts: VisibleOptions,
): { nodes: GraphNode[]; links: GraphLink[]; capped: boolean } {
  const typeOk = (t: string): boolean => opts.typeFilter.size === 0 || opts.typeFilter.has(t);
  let capped = false;

  const scope = new Set<number>(opts.showAll ? merged.nodes.map(n => n.id) : sel.nodeIds);
  if (!opts.showAll && opts.neighbors) {
    const cap = opts.neighborCap ?? NEIGHBOR_CAP;
    for (const l of merged.links) {
      const s = idOf(l.source);
      const t = idOf(l.target);
      const sIn = sel.nodeIds.has(s);
      const tIn = sel.nodeIds.has(t);
      if (sIn === tIn) continue; // both in chain, or both outside → not a boundary edge
      const outside = sIn ? t : s;
      if (scope.has(outside)) continue;
      if (scope.size - sel.nodeIds.size >= cap) {
        capped = true;
        continue;
      }
      scope.add(outside);
    }
  }

  const nodes = merged.nodes.filter(n => scope.has(n.id) && typeOk(n.type));
  const keep = new Set(nodes.map(n => n.id));
  const links = merged.links.filter(l => {
    const s = idOf(l.source);
    const t = idOf(l.target);
    if (!keep.has(s) || !keep.has(t)) return false;
    if (opts.showAll) return true;
    if (sel.linkKeys.has(linkKey(s, t))) return true;
    // 1-hop neighbour edges: keep only those that touch the chain.
    return opts.neighbors && (sel.nodeIds.has(s) || sel.nodeIds.has(t));
  });

  return { nodes, links, capped };
}

/**
 * L2 dual-column rows (spec §7). Rows are ordered upstream→downstream (root →
 * target), i.e. reverse of the stored [target..root] order, matching the L1
 * x-axis. Each row carries the signed CE (追路徑) and signed φ_asym (釘元兇);
 * DESCRIPTIVE, not cumulative — no efficiency total.
 */
export function buildResponsibilityRows(chain: CrimeChain): ResponsibilityRow[] {
  const ordered = [...chain.nodes].reverse(); // root first, target last
  const rootGlobal = chain.nodes[chain.nodes.length - 1]?.global;
  const pivotGlobal = pivotGlobalOf(chain);
  return ordered.map(n => ({
    global: n.global,
    real_id: n.real_id,
    type: n.type,
    pos: n.pos,
    time: n.time,
    is_root: n.global === rootGlobal,
    is_target: Boolean(n.is_target),
    is_pivot: pivotGlobal != null && n.global === pivotGlobal,
    ce: n.ce ?? null,
    phiAsym: n.phi_asym ?? null,
  }));
}
