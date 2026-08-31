import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import fs from 'fs';
import path from 'path';

// Source of truth for the viewer lives one dir up in the repo:
//   viz/crime_chains.json    — traced fraud chains with per-node φ_asym (primary)
//   results/crime_chains.csv — flat CSV fallback (no nested per-node φ)
//   results/chain_neighbors.json — real 1-hop neighbours from the full graph
// Served live in dev (regenerating is picked up without copying) and bundled
// into dist/ on build.
const STATIC_FILES: { route: string; source: string; type: string }[] = [
  { route: '/crime_chains.json', source: path.resolve('../viz/crime_chains.json'), type: 'application/json; charset=utf-8' },
  { route: '/crime_chains_transaction.json', source: path.resolve('../viz/crime_chains_transaction.json'), type: 'application/json; charset=utf-8' },
  { route: '/crime_chains_wallet.json', source: path.resolve('../viz/crime_chains_wallet.json'), type: 'application/json; charset=utf-8' },
  { route: '/crime_chains.csv', source: path.resolve('../results/crime_chains.csv'), type: 'text/csv; charset=utf-8' },
  { route: '/chain_neighbors.json', source: path.resolve('../results/chain_neighbors.json'), type: 'application/json; charset=utf-8' },
];

interface ServerResponse {
  setHeader: (name: string, value: string) => void;
}

function serveResultFiles() {
  const middleware = (server: {
    middlewares: { use: (fn: (req: { url?: string }, res: ServerResponse, next: () => void) => void) => void };
  }) => {
    server.middlewares.use((req, res, next) => {
      const url = req.url?.split('?')[0];
      const match = STATIC_FILES.find(f => f.route === url);
      if (match && fs.existsSync(match.source)) {
        res.setHeader('Content-Type', match.type);
        res.setHeader('Access-Control-Allow-Origin', '*');
        // res is the real Node http.ServerResponse (a Writable) at runtime.
        fs.createReadStream(match.source).pipe(res as unknown as fs.WriteStream);
        return;
      }
      next();
    });
  };
  return {
    name: 'serve-result-files',
    configureServer: middleware,
    configurePreviewServer: middleware,
    writeBundle() {
      for (const f of STATIC_FILES) {
        if (!fs.existsSync(f.source)) continue;
        fs.copyFileSync(f.source, path.resolve('./dist', path.basename(f.source)));
      }
    },
  };
}

export default defineConfig({
  base: './',
  plugins: [react(), serveResultFiles()],
});
