/**
 * Parse `results/crime_chains.csv` into the viewer's CrimeChainData contract.
 *
 * CSV columns (one row = one fraud chain, ordered target → root):
 *   target_txid, depth, root_type, root_real_id, root_is_fraud,
 *   is_true_positive, n_nodes, chain_real_ids, chain_types, chain_ce
 *
 * `chain_real_ids` / `chain_types` / `chain_ce` are pipe(|)-separated, length
 * n_nodes. `chain_ce[0]` is empty (the target node has no upstream CE); ce[i]
 * is the causal effect of node i on node i-1 (matches CrimeChainNode.ce).
 *
 * The CSV carries no per-node `global` id, `time`, or per-node `fraud` label, so:
 *   - global  : a deterministic real_id → int map (shared real_id ⇒ same node,
 *               which is what lets laundering hubs merge across chains).
 *   - time    : null (no timestamp in this export).
 *   - fraud   : approximated from known labels — target = is_true_positive,
 *               root = root_is_fraud, relays = false (unknown).
 */
import type { CrimeChain, CrimeChainData, CrimeChainNode } from '../types';

const EXPECTED_HEADER = [
  'target_txid',
  'depth',
  'root_type',
  'root_real_id',
  'root_is_fraud',
  'is_true_positive',
  'n_nodes',
  'chain_real_ids',
  'chain_types',
  'chain_ce',
];

const parseBool = (s: string): boolean => s.trim().toLowerCase() === 'true';

const parseCe = (s: string): number | undefined => {
  const t = s.trim();
  if (t === '') return undefined;
  const v = Number(t);
  return Number.isFinite(v) ? v : undefined;
};

/** Split a CSV line on commas. The chain columns use `|` internally, so no field contains a comma. */
const splitRow = (line: string): string[] => line.split(',');

/**
 * Parse the crime-chains CSV text. Throws a clear error on malformed input.
 * `globalOf` assigns a stable integer id per distinct real_id across the file.
 */
export function parseCrimeChainsCsv(text: string): CrimeChainData {
  const lines = text.split(/\r?\n/).filter(l => l.trim() !== '');
  if (lines.length === 0) throw new Error('CSV 是空的(沒有任何資料列)。');

  const header = splitRow(lines[0]).map(h => h.trim());
  const headerOk = EXPECTED_HEADER.every((col, i) => header[i] === col);
  if (!headerOk) {
    throw new Error(
      `CSV 標頭不符。預期 ${EXPECTED_HEADER.join(',')},實際 ${header.join(',')}。`,
    );
  }

  const idByRealId = new Map<string, number>();
  const globalOf = (realId: string): number => {
    const existing = idByRealId.get(realId);
    if (existing != null) return existing;
    const id = idByRealId.size;
    idByRealId.set(realId, id);
    return id;
  };

  const chains: CrimeChain[] = [];

  for (let r = 1; r < lines.length; r++) {
    const cols = splitRow(lines[r]);
    if (cols.length < EXPECTED_HEADER.length) continue; // skip malformed/blank tail rows

    const target_txid = cols[0].trim();
    const depth = Number(cols[1]);
    const root_type = cols[2].trim();
    const root_real_id = cols[3].trim();
    const root_is_fraud = parseBool(cols[4]);
    const is_true_positive = parseBool(cols[5]);

    const realIds = cols[7].split('|');
    const types = cols[8].split('|');
    const ces = cols[9].split('|');
    const n = realIds.length;
    if (n === 0 || types.length !== n) continue;

    const lastIdx = n - 1;
    const nodes: CrimeChainNode[] = realIds.map((realId, i) => {
      const id = realId.trim();
      const isTarget = i === 0;
      const isRoot = i === lastIdx;
      const fraud = isTarget ? is_true_positive : isRoot ? root_is_fraud : false;
      const node: CrimeChainNode = {
        pos: i,
        global: globalOf(id),
        type: (types[i] ?? '').trim(),
        real_id: id,
        time: null,
        fraud,
      };
      if (isTarget) node.is_target = true;
      const ce = parseCe(ces[i] ?? '');
      if (ce != null) node.ce = ce;
      return node;
    });

    chains.push({
      target_txid,
      depth: Number.isFinite(depth) ? depth : nodes.length - 1,
      root_type,
      root_real_id,
      root_is_fraud,
      is_true_positive,
      nodes,
    });
  }

  if (chains.length === 0) throw new Error('CSV 沒有可解析的犯罪鏈。');

  const nTp = chains.filter(c => c.is_true_positive).length;
  const nFraudRoot = chains.filter(c => c.root_is_fraud).length;
  const meanDepth = chains.reduce((s, c) => s + c.depth, 0) / chains.length;

  return {
    meta: {
      dataset: 'elliptic++ (crime_chains.csv)',
      n_chains: chains.length,
      n_true_positive: nTp,
      n_fraud_root: nFraudRoot,
      mean_depth: Number(meanDepth.toFixed(2)),
    },
    chains,
  };
}
