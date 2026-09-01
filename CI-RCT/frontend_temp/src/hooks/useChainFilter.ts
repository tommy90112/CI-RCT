import { useCallback, useMemo, useState } from 'react';
import type { CrimeChain, CrimeChainData } from '../types';

// Chain search / φ-only filter shared by the top-bar stepper and the control
// panel's chain picker, so both walk the same filtered list.

// Cap rendered <option> count so a 2000-chain CSV stays snappy; users narrow
// the list with the search box.
export const MAX_OPTIONS = 300;

/** A chain has a φ_asym responsibility spotlight iff it is transaction-target. */
export const chainHasPhi = (c: CrimeChain): boolean => c.nodes.some(n => n.phi_asym != null);

export interface ChainMatch {
  c: CrimeChain;
  i: number;
}

export interface ChainFilter {
  query: string;
  setQuery: (q: string) => void;
  onlyPhi: boolean;
  toggleOnlyPhi: () => void;
  /** Chains matching the search box (and the φ-only filter), in dataset order. */
  matches: ChainMatch[];
  /** First MAX_OPTIONS matches, always including the selected chain. */
  shown: ChainMatch[];
  /** Index of the selected chain inside `matches`, or -1 when filtered out. */
  posInMatches: number;
  /** Move selection ±delta within `matches` (wraps). */
  step: (delta: number) => void;
  nWithPhi: number;
  phiPct: number;
}

export function useChainFilter(
  data: CrimeChainData,
  selectedIdx: number,
  onSelect: (i: number) => void,
): ChainFilter {
  const [query, setQuery] = useState('');
  const [onlyPhi, setOnlyPhi] = useState(false);

  // φ_asym coverage across the loaded data (spec §3): transaction-target chains.
  const nWithPhi = useMemo(() => data.chains.filter(chainHasPhi).length, [data.chains]);
  const phiPct = data.chains.length ? Math.round((nWithPhi / data.chains.length) * 100) : 0;

  // Chains matching the search box (by target txid / root address / root type),
  // optionally restricted to those carrying φ_asym (transaction-target).
  const matches = useMemo(() => {
    const q = query.trim().toLowerCase();
    return data.chains
      .map((c, i) => ({ c, i }))
      .filter(({ c }) => {
        if (onlyPhi && !chainHasPhi(c)) return false;
        if (!q) return true;
        return (
          c.target_txid.toLowerCase().includes(q) ||
          c.root_real_id.toLowerCase().includes(q) ||
          c.root_type.toLowerCase().includes(q)
        );
      });
  }, [data.chains, query, onlyPhi]);

  // Turning the φ-only filter on jumps to the first φ chain if the current one lacks φ.
  const toggleOnlyPhi = useCallback(() => {
    const next = !onlyPhi;
    setOnlyPhi(next);
    if (next && !chainHasPhi(data.chains[selectedIdx])) {
      const firstPhi = data.chains.findIndex(chainHasPhi);
      if (firstPhi >= 0) onSelect(firstPhi);
    }
  }, [onlyPhi, data.chains, selectedIdx, onSelect]);

  const shown = useMemo(() => {
    const slice = matches.slice(0, MAX_OPTIONS);
    if (slice.some(m => m.i === selectedIdx)) return slice;
    const sel = data.chains[selectedIdx];
    return sel ? [{ c: sel, i: selectedIdx }, ...slice] : slice;
  }, [matches, selectedIdx, data.chains]);

  const posInMatches = matches.findIndex(m => m.i === selectedIdx);

  const step = useCallback(
    (delta: number) => {
      if (matches.length === 0) return;
      const base = posInMatches < 0 ? 0 : posInMatches;
      const next = (base + delta + matches.length) % matches.length;
      onSelect(matches[next].i);
    },
    [matches, posInMatches, onSelect],
  );

  return { query, setQuery, onlyPhi, toggleOnlyPhi, matches, shown, posInMatches, step, nWithPhi, phiPct };
}
