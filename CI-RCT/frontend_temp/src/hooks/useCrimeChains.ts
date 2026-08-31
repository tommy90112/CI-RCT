import { useCallback, useEffect, useState } from 'react';
import { parseCrimeChainData } from '../lib/graph';
import { parseCrimeChainsCsv } from '../lib/parseCsv';
import type { CrimeChainData, DataVariant } from '../types';

interface UseCrimeChainsResult {
  data: CrimeChainData | null;
  loading: boolean;
  error: string | null;
  source: string;
  variant: DataVariant;
  loadFromFile: (file: File) => void;
  loadVariant: (variant: DataVariant) => void;
}

function getErrorMessage(error: unknown): string {
  return error instanceof Error ? error.message : '未知錯誤';
}

/** Per-variant dump filename: joint is the primary crime_chains.json. */
function fileForVariant(variant: DataVariant): string {
  return variant === 'joint' ? 'crime_chains.json' : `crime_chains_${variant}.json`;
}

/** Parse text as CSV (.csv) or JSON (anything else / by content sniff). */
function parseByName(text: string, name: string): CrimeChainData {
  const isCsv = name.toLowerCase().endsWith('.csv') || (!name.toLowerCase().endsWith('.json') && !text.trimStart().startsWith('{'));
  return isCsv ? parseCrimeChainsCsv(text) : parseCrimeChainData(JSON.parse(text) as unknown);
}

/**
 * Load crime-chain data. On mount loads the joint dump (主結果). `loadVariant`
 * switches between joint / transaction / wallet dumps served from the repo's
 * viz/ dir (see vite.config.ts); a missing per-variant file surfaces an error
 * but keeps the currently-loaded data. `loadFromFile` lets the user pick a file.
 */
export function useCrimeChains(): UseCrimeChainsResult {
  const [data, setData] = useState<CrimeChainData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [source, setSource] = useState('');
  const [variant, setVariant] = useState<DataVariant>('joint');

  const loadVariant = useCallback((next: DataVariant) => {
    const file = fileForVariant(next);
    setLoading(true);
    const url = `${import.meta.env.BASE_URL}${file}`;
    fetch(url, { cache: 'no-store' })
      .then(res => {
        if (!res.ok) throw new Error('not found');
        return res.text();
      })
      .then(text => {
        setData(parseCrimeChainData(JSON.parse(text) as unknown));
        setSource(file);
        setVariant(next);
        setError(null);
      })
      .catch(() => {
        // Keep the current data; just report the missing/failed variant.
        setError(`找不到 ${file} — 該變體資料可能尚未產生（請先在 server 跑出對應 dump）。`);
      })
      .finally(() => setLoading(false));
  }, []);

  useEffect(() => {
    loadVariant('joint');
  }, [loadVariant]);

  const loadFromFile = useCallback((file: File) => {
    setLoading(true);
    const reader = new FileReader();
    reader.onload = () => {
      try {
        setData(parseByName(String(reader.result), file.name));
        setSource(file.name);
        setError(null);
      } catch (error: unknown) {
        setError(getErrorMessage(error));
      } finally {
        setLoading(false);
      }
    };
    reader.onerror = () => {
      setError('讀檔失敗');
      setLoading(false);
    };
    reader.readAsText(file);
  }, []);

  return { data, loading, error, source, variant, loadFromFile, loadVariant };
}
