/**
 * Data contract for the CI-RCT explainability viewer.
 *
 * Mirrors the JSON produced by `evaluate.py --dump_chains` /
 * `scripts/export_crime_chains.py`. Each chain is a traced root-cause path
 * [target, ..., root] of real Elliptic++ entities (transaction txId / wallet
 * Bitcoin address).
 *
 * Fields marked OPTIONAL correspond to spec §7.4 known gaps: `phi`,
 * `feature_attribution`, and `features` may be absent in current exports, so
 * every consumer must degrade gracefully (see `lib/render.ts` fallbacks).
 */

export type NodeType = 'transaction' | 'wallet';

/** Which eval dump to load: joint (主結果) or a single-task variant. */
export type DataVariant = 'joint' | 'transaction' | 'wallet';

/** One feature's contribution to a node's local prediction (L3 drill-down). */
export interface FeatureContribution {
  /** Feature name (named for wallets; `Local_feature_*` is anonymised for tx). */
  name: string;
  /** Signed causal feature effect CFE = ΔP_fraud(target) under do(x=baseline). */
  value: number;
  /** Grad×Input saliency for the same feature (correlational; the fallback signal). */
  saliency?: number;
  /** Optional raw/normalised feature value for the tooltip. */
  raw?: number;
}

/** How L3 was computed: causal do-intervention, or the saliency fallback. */
export type FeatureAttributionMethod = 'causal_do' | 'saliency_fallback';

export interface CrimeChainNode {
  /** Position in the chain: 0 = flagged-fraud target, last = traced root. */
  pos: number;
  /** Global causal-graph id (stable across chains; merges shared nodes). */
  global: number;
  type: NodeType | string;
  /** Real identity: transaction txId, or wallet Bitcoin address. */
  real_id: string;
  /** Discrete Elliptic timestep (1–49), or null when unknown. */
  time: number | null;
  /** True iff this node's dataset label is illicit. */
  fraud: boolean;
  /** Present only on pos 0. */
  is_target?: boolean;
  /** Signed causal effect (CE) of this upstream node on the previous one. Full coverage; null on the target. */
  ce?: number;
  /** Additive Shapley φ (≈ damped CE). Full coverage; currently unused in the UI. */
  phi_add?: number;
  /**
   * Asymmetric Causal Shapley φ — the causal-responsibility "spotlight" signal.
   * Sparse & peaked (one node ≈ owns the chain). `null` when not computed
   * (wallet-target chains: the fraud readout head only scores transactions).
   */
  phi_asym?: number | null;
  /** OPTIONAL — L3 per-feature attribution (pivot node only). */
  feature_attribution?: FeatureContribution[];
  /** OPTIONAL — how L3 was computed (causal do-intervention vs saliency fallback). */
  feature_attribution_method?: FeatureAttributionMethod;
}

export interface CrimeChain {
  target_txid: string;
  depth: number;
  root_type: string;
  root_real_id: string;
  root_is_fraud: boolean;
  /** Target is actually illicit (not a classifier false positive). */
  is_true_positive?: boolean;
  nodes: CrimeChainNode[];
}

export interface CrimeChainMeta {
  dataset?: string;
  checkpoint?: string;
  n_chains?: number;
  n_true_positive?: number;
  n_fraud_root?: number;
  mean_depth?: number;
}

export interface CrimeChainData {
  meta?: CrimeChainMeta;
  chains: CrimeChain[];
}

/** Force-graph node (merged across chains by `global`). */
export interface GraphNode {
  id: number;
  type: string;
  real_id: string;
  time: number | null;
  fraud: boolean;
  is_target: boolean;
  /** Whether any chain traced this node as its root cause. */
  is_root: boolean;
  /** Max |asymmetric φ| seen across chains — drives L1 size+colour. 0 when never scored. */
  phiAsym: number;
  /** How many chain edges touch this node — bigger = laundering hub. */
  deg: number;
  // Pinned positions (temporal layout); undefined in structure mode.
  fx?: number;
  fy?: number;
  fz?: number;
}

export interface GraphLink {
  source: number;
  target: number;
  ce: number;
}

export interface MergedGraph {
  nodes: GraphNode[];
  links: GraphLink[];
  tMin: number;
  tMax: number;
}

export type LayoutMode = 'structure' | 'temporal';

/** Node-display options owned by the left control panel, consumed by the graph. */
export interface DisplayOptions {
  /** Render with the 2D or 3D force-graph engine. */
  dimensions: 2 | 3;
  layout: LayoutMode;
  /** Show every chain (laundering network) vs only the selected chain. */
  showAll: boolean;
  /** Also show the chain's 1-hop neighbours from the merged graph, greyed out. */
  neighbors: boolean;
  /** Animated money-flow particles along edges. */
  particles: boolean;
  /** Size/colour nodes by φ (vs by degree) — redundant φ encoding. */
  sizeByPhi: boolean;
  /** Only show nodes whose type is in this set; empty = show all types. */
  typeFilter: Set<string>;
}

/** One chain node's 1-hop neighbourhood in the FULL Elliptic++ graph. */
export interface NeighborEntry {
  /** Total neighbour count in the full graph (may exceed the capped list). */
  degree: number;
  /** Capped list of neighbour nodes (see scripts/export_chain_neighbors.py). */
  nodes: { real_id: string; type: string }[];
}

/** real_id → its 1-hop neighbourhood, loaded from chain_neighbors.json. */
export type NeighborIndex = Record<string, NeighborEntry>;

export interface ChainNeighborData {
  meta?: { cap?: number; n_chain_nodes?: number; n_with_neighbors?: number };
  neighbors: NeighborIndex;
}

/** Where the displayed neighbours came from. */
export type NeighborSource = 'full' | 'union' | 'none';

/**
 * One row of the L2 dual-column chart (spec §7): a chain node with its signed
 * edge CE (追路徑) and signed asymmetric φ (釘元兇). DESCRIPTIVE — bars are NOT
 * cumulative and make no efficiency claim (chain φ is only a path-subset of the
 * target's parents, so it does not sum to the prediction).
 */
export interface ResponsibilityRow {
  global: number;
  real_id: string;
  type: string;
  pos: number;
  time: number | null;
  is_root: boolean;
  is_target: boolean;
  /** This node carries the chain's peak |φ_asym| (the spotlight pivot). */
  is_pivot: boolean;
  /** Signed CE of the edge into this node's downstream child; null on target. */
  ce: number | null;
  /** Signed asymmetric φ; null when not computed (wallet-target chains). */
  phiAsym: number | null;
}
