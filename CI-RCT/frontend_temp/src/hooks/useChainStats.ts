import { useEffect, useState } from 'react';
import { parseCrimeChainData } from '../lib/graph';
import { pivotGlobalOf } from '../lib/graph';
import type { CrimeChain } from '../types';

/**
 * Descriptive facts about the traced chains, computed live from the same
 * crime_chains.json the explorer loads — so the idea page never quotes a stale
 * number. All are proportions of what the tracer actually produced; none is a
 * detection metric.
 */
export interface ChainStats {
  nChains: number;
  /** Chains that carry an asymmetric-φ pivot (transaction-target chains). */
  nScored: number;
  /** Share of pivots that are wallets (vs transactions). */
  pivotWalletShare: number;
  /** Share of pivots that are NOT the target's direct upstream (pos ≥ 2). */
  pivotBeyondParentShare: number;
  /** Share of traced roots that are wallets. */
  rootWalletShare: number;
  /** Largest number of chains converging on one root. */
  maxFanIn: number;
  meanDepth: number;
}

function compute(chains: CrimeChain[]): ChainStats {
  const scored = chains
    .map(c => ({ c, pivot: pivotGlobalOf(c) }))
    .filter((x): x is { c: CrimeChain; pivot: number } => x.pivot != null);
  const pivotNodes = scored.map(({ c, pivot }) => c.nodes.find(n => n.global === pivot)!);
  const share = (n: number, d: number) => (d > 0 ? n / d : 0);
  const fanIn = new Map<string, number>();
  for (const c of chains) fanIn.set(c.root_real_id, (fanIn.get(c.root_real_id) ?? 0) + 1);
  return {
    nChains: chains.length,
    nScored: scored.length,
    pivotWalletShare: share(pivotNodes.filter(n => n.type === 'wallet').length, pivotNodes.length),
    pivotBeyondParentShare: share(pivotNodes.filter(n => n.pos >= 2).length, pivotNodes.length),
    rootWalletShare: share(chains.filter(c => c.root_type === 'wallet').length, chains.length),
    maxFanIn: Math.max(0, ...fanIn.values()),
    meanDepth: chains.length ? chains.reduce((s, c) => s + c.depth, 0) / chains.length : 0,
  };
}

export function useChainStats(): { stats: ChainStats | null; error: string | null } {
  const [stats, setStats] = useState<ChainStats | null>(null);
  const [error, setError] = useState<string | null>(null);
  useEffect(() => {
    let cancelled = false;
    fetch(`${import.meta.env.BASE_URL}crime_chains.json`)
      .then(res => {
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        return res.json();
      })
      .then((raw: unknown) => {
        if (!cancelled) setStats(compute(parseCrimeChainData(raw).chains));
      })
      .catch((e: unknown) => {
        if (!cancelled) setError(e instanceof Error ? e.message : '載入失敗');
      });
    return () => {
      cancelled = true;
    };
  }, []);
  return { stats, error };
}
