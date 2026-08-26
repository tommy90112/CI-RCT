import { useEffect, useState } from 'react';
import { parseNeighborData } from '../lib/neighbors';
import type { NeighborIndex } from '../types';

interface UseChainNeighborsResult {
  index: NeighborIndex | null;
  loading: boolean;
  /** True once a fetch attempt finished (success or not) — gates fallback UI. */
  ready: boolean;
}

/**
 * Load the real 1-hop neighbourhood overlay (chain_neighbors.json) served from
 * the repo's results/ dir. Optional: if it's absent the graph falls back to
 * chain-union neighbours, so failure is non-fatal.
 */
export function useChainNeighbors(): UseChainNeighborsResult {
  const [index, setIndex] = useState<NeighborIndex | null>(null);
  const [loading, setLoading] = useState(true);
  const [ready, setReady] = useState(false);

  useEffect(() => {
    let cancelled = false;
    const url = `${import.meta.env.BASE_URL}chain_neighbors.json`;
    fetch(url, { cache: 'no-store' })
      .then(res => {
        if (!res.ok) throw new Error('not found');
        return res.json();
      })
      .then((raw: unknown) => {
        if (cancelled) return;
        setIndex(parseNeighborData(raw).neighbors);
      })
      .catch(() => {
        if (!cancelled) setIndex(null);
      })
      .finally(() => {
        if (!cancelled) {
          setLoading(false);
          setReady(true);
        }
      });
    return () => {
      cancelled = true;
    };
  }, []);

  return { index, loading, ready };
}
