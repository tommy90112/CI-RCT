/**
 * Real 1-hop neighbourhood overlay (from the FULL Elliptic++ graph).
 *
 * The chain-union graph only knows neighbours shared across the 2000 traced
 * chains. `chain_neighbors.json` (scripts/export_chain_neighbors.py) carries the
 * true 1-hop neighbours per chain node; here we parse it and graft those
 * neighbours onto the selected chain as greyed context nodes.
 */
import type {
  ChainNeighborData,
  GraphLink,
  GraphNode,
  MergedGraph,
  NeighborIndex,
  NeighborSource,
} from '../types';
import { visibleSubgraph } from './graph';

/** Narrow untrusted JSON into ChainNeighborData, or throw a clear error. */
export function parseNeighborData(raw: unknown): ChainNeighborData {
  if (typeof raw !== 'object' || raw === null || typeof (raw as ChainNeighborData).neighbors !== 'object') {
    throw new Error('chain_neighbors.json 缺少 neighbors 物件。');
  }
  return raw as ChainNeighborData;
}

/**
 * Build greyed neighbour nodes + links for a chain. Neighbour nodes get unique
 * NEGATIVE ids so they never collide with merged graph ids (0..N) and so the
 * existing accessors dim them (their id is not in the chain's nodeId set).
 */
function buildNeighborAdditions(
  chainNodes: GraphNode[],
  chainRealIds: Set<string>,
  index: NeighborIndex,
  typeFilter: Set<string>,
): { nodes: GraphNode[]; links: GraphLink[] } {
  const typeOk = (t: string): boolean => typeFilter.size === 0 || typeFilter.has(t);
  const idByReal = new Map<string, number>();
  let nextId = -1;
  const nodes: GraphNode[] = [];
  const links: GraphLink[] = [];

  for (const cn of chainNodes) {
    const entry = index[cn.real_id];
    if (!entry) continue;
    for (const nb of entry.nodes) {
      if (chainRealIds.has(nb.real_id) || !typeOk(nb.type)) continue;
      let nid = idByReal.get(nb.real_id);
      if (nid == null) {
        nid = nextId--;
        idByReal.set(nb.real_id, nid);
        nodes.push({
          id: nid,
          type: nb.type,
          real_id: nb.real_id,
          time: null,
          fraud: false,
          is_target: false,
          is_root: false,
          phiAsym: 0,
          deg: 1,
        });
      }
      links.push({ source: cn.id, target: nid, ce: 0 });
    }
  }
  return { nodes, links };
}

/**
 * Single source of truth for what the graph shows. Returns the chain (coloured)
 * plus its neighbours (greyed) — from the full graph when `index` is loaded,
 * otherwise from the chain-union fallback. `neighborSource` tells the UI which.
 */
export function buildGraphView(
  merged: MergedGraph,
  sel: { nodeIds: Set<number>; linkKeys: Set<string> },
  index: NeighborIndex | null,
  opts: { showAll: boolean; neighbors: boolean; typeFilter: Set<string> },
): { nodes: GraphNode[]; links: GraphLink[]; neighborSource: NeighborSource } {
  if (opts.showAll) {
    const v = visibleSubgraph(merged, sel, { showAll: true, neighbors: false, typeFilter: opts.typeFilter });
    return { nodes: v.nodes, links: v.links, neighborSource: 'none' };
  }

  const chainView = visibleSubgraph(merged, sel, {
    showAll: false,
    neighbors: false,
    typeFilter: opts.typeFilter,
  });
  if (!opts.neighbors) {
    return { nodes: chainView.nodes, links: chainView.links, neighborSource: 'none' };
  }

  if (index) {
    const chainReal = new Set(chainView.nodes.map(n => n.real_id));
    const add = buildNeighborAdditions(chainView.nodes, chainReal, index, opts.typeFilter);
    return {
      nodes: [...chainView.nodes, ...add.nodes],
      links: [...chainView.links, ...add.links],
      neighborSource: 'full',
    };
  }

  // Fallback: neighbours shared across the chain-union graph only.
  const v = visibleSubgraph(merged, sel, { showAll: false, neighbors: true, typeFilter: opts.typeFilter });
  return { nodes: v.nodes, links: v.links, neighborSource: 'union' };
}
