/**
 * One REAL traced chain from viz/crime_chains.json (joint variant, chain #33),
 * embedded so the idea page's concept demos run without loading the 2 MB dump.
 * Order: upstream → downstream (root → target), i.e. the money-flow direction.
 * `ce` on a node = signed causal effect of that node on its downstream child.
 */
export interface DemoNode {
  id: string;
  type: 'wallet' | 'transaction';
  role?: 'root' | 'pivot' | 'target';
  /** CE(this → downstream child); absent on the target. */
  ce?: number;
  /** Asymmetric causal Shapley φ; absent on the target (it is the readout). */
  phi?: number;
}

export const DEMO_CHAIN_INDEX = 33;
export const DEMO_TARGET_TXID = '279424082';

export const DEMO_CHAIN: readonly DemoNode[] = [
  { id: '1KfHxNrU48rdd5Vnh4X4C22gkS9oxGoiVV', type: 'wallet', role: 'root', ce: 0.3851, phi: 0.2767 },
  { id: '44995911', type: 'transaction', role: 'pivot', ce: 0.3518, phi: 0.6404 },
  { id: '1EX5vj3fbq7EjCyyfVC4U3bao2zSi1Rsrm', type: 'wallet', ce: 0.4059, phi: 0.1245 },
  { id: '279424082', type: 'transaction', role: 'target' },
];

/** Top causal feature effects on the pivot (L3, do-intervention). */
export const DEMO_PIVOT_FEATURES: readonly { name: string; value: number }[] = [
  { name: 'Local_feature_91', value: 0.149 },
  { name: 'Local_feature_3', value: -0.033 },
  { name: 'Local_feature_90', value: -0.019 },
];
