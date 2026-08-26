/**
 * Visual encoding helpers (spec §3.2). Redundant encoding: φ drives BOTH node
 * size and colour depth (colour-blind safe). Shape = node type; edge width = CE.
 * Depends only on `three` (already a project dep).
 */
import * as THREE from 'three';
import type { GraphLink, GraphNode } from '../types';

export const COLOR = {
  tx: '#5aa9ff',
  wallet: '#f2b53a',
  fraud: '#ff5470',
  root: '#a78bfa',
  pivot: '#fbbf24',
  dim: '#39424d',
  ceNeg: '#ff5470',
  cePos: '#43d17a',
} as const;

export const nodeBaseColor = (n: GraphNode): string =>
  n.fraud ? COLOR.fraud : n.type === 'transaction' ? COLOR.tx : COLOR.wallet;

/**
 * Sequential φ colormap (spec §3.2: φ越大越深). Maps t∈[0,1] along a
 * dark-navy → violet → magenta ramp. Used for redundant φ encoding and to keep
 * L1 node colour ↔ L2 bar colour consistent for the same node.
 */
export function phiColor(t: number): string {
  const c = Math.max(0, Math.min(1, t));
  const lerp = (a: number, b: number) => Math.round(a + (b - a) * c);
  // #2a2a55 (low) → #c026d3 (high)
  const r = lerp(42, 192);
  const g = lerp(42, 38);
  const b = lerp(85, 211);
  return `rgb(${r},${g},${b})`;
}

/** Normalise a value against a max for [0,1] colormap input. */
export const phiNorm = (phi: number, phiMax: number): number => (phiMax > 0 ? phi / phiMax : 0);

/**
 * Node size (spec §6.2): |φ_asym| drives size when the chain was scored, else
 * degree (hub) so the graph still reads for wallet-target chains. Fraud nodes
 * get a mild boost. `phiMax` is the merged graph's max |φ_asym|.
 */
export const nodeSize = (n: GraphNode, phiMax = 0): number => {
  const base = phiMax > 0 ? 3 + phiNorm(n.phiAsym, phiMax) * 12 : 3 + Math.min(n.deg, 10) * 1.2;
  return n.fraud ? base * 1.4 : base;
};

export const linkColorByCe = (l: GraphLink): string => (l.ce < 0 ? COLOR.ceNeg : COLOR.cePos);

/** Edge width by |CE| (spec §6.2): stronger causal effect = thicker. */
export const linkWidthByCe = (ce: number, ceMax: number): number =>
  1 + (ceMax > 0 ? Math.min(1, Math.abs(ce) / ceMax) : 0) * 4;

export const shortId = (s: string): string =>
  s.length > 13 ? `${s.slice(0, 7)}…${s.slice(-3)}` : s;

/** Unicode glyph for a node type (wallet = ●, transaction = ◆). */
export const typeGlyph = (type: string): string => (type === 'transaction' ? '◆' : '●');

/**
 * Build a billboarded text label as a THREE.Sprite from a canvas texture, via
 * react-force-graph-3d's nodeThreeObject (extend=true keeps the sphere).
 */
export function makeTextSprite(text: string, color: string): THREE.Sprite {
  const fontSize = 44;
  const font = `bold ${fontSize}px ui-monospace, Menlo, monospace`;
  const measure = document.createElement('canvas').getContext('2d');
  if (measure) measure.font = font;
  const padding = 14;
  const textWidth = measure ? measure.measureText(text).width : text.length * fontSize * 0.6;
  const width = Math.ceil(textWidth) + padding * 2;
  const height = fontSize + padding * 2;

  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext('2d');
  if (ctx) {
    ctx.font = font;
    ctx.fillStyle = 'rgba(8,12,17,0.72)';
    ctx.fillRect(0, 0, width, height);
    ctx.fillStyle = color;
    ctx.textBaseline = 'middle';
    ctx.fillText(text, padding, height / 2);
  }

  const texture = new THREE.CanvasTexture(canvas);
  texture.minFilter = THREE.LinearFilter;
  const material = new THREE.SpriteMaterial({ map: texture, transparent: true, depthWrite: false });
  const sprite = new THREE.Sprite(material);
  const scale = 0.16;
  sprite.scale.set(width * scale, height * scale, 1);
  return sprite;
}
