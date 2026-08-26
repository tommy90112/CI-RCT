import { useState, type ReactNode } from 'react';

// Reusable collapsible drawer primitive.
//
// Interaction: hovering the edge tab PEEKS the panel open; clicking the tab
// PINS it open (stays open after the mouse leaves) — click again to unpin.
// `open` is the controlled pinned state; transient hover is internal. The panel
// and its tab live in one flex wrapper that translates as a unit, so collapsing
// slides the panel off-screen while the tab stays pinned to the edge.

interface DrawerProps {
  side: 'left' | 'bottom';
  /** Pinned-open state (controlled). Hover peeking is handled internally. */
  open: boolean;
  /** Tab click toggles the pin → onOpenChange(!open). */
  onOpenChange: (open: boolean) => void;
  tab: ReactNode; // label shown on the edge tab
  hoverOpen?: boolean; // peek open on hover
  size?: number; // open width(px) for left / height(px) for bottom
  children: ReactNode;
  className?: string;
}

const LEFT_DEFAULT_SIZE = 320;
const BOTTOM_DEFAULT_SIZE = 340;
const LEFT_TAB_WIDTH = 34;
const BOTTOM_TAB_HEIGHT = 32;

const PANEL_CLASS =
  'bg-slate-900/92 backdrop-blur-xl border border-slate-700/50 shadow-2xl shadow-black/40';

const TAB_BASE =
  'group relative flex items-center justify-center select-none transition-colors ' +
  'bg-slate-900/92 backdrop-blur-xl border border-slate-700/50 ' +
  'text-slate-400 hover:text-sky-300 hover:bg-slate-800/90';

export function Drawer({
  side,
  open,
  onOpenChange,
  tab,
  hoverOpen = false,
  size,
  children,
  className,
}: DrawerProps) {
  const [hovering, setHovering] = useState(false);
  const isLeft = side === 'left';
  const openSize = size ?? (isLeft ? LEFT_DEFAULT_SIZE : BOTTOM_DEFAULT_SIZE);
  const visible = open || (hoverOpen && hovering);

  const handlers = hoverOpen
    ? { onMouseEnter: () => setHovering(true), onMouseLeave: () => setHovering(false) }
    : {};
  const toggle = () => onOpenChange(!open);

  // Small pin dot: filled when pinned, ring when only peeking.
  const pinDot = (
    <span
      className={`inline-block h-1.5 w-1.5 rounded-full ${
        open ? 'bg-sky-400' : 'border border-slate-500'
      }`}
    />
  );

  if (isLeft) {
    return (
      <div
        className="absolute left-0 top-0 z-20 flex h-full items-center transition-transform duration-300 ease-out"
        style={{ transform: `translateX(${visible ? 0 : -openSize}px)` }}
      >
        <div
          {...handlers}
          className={[PANEL_CLASS, 'rounded-r-2xl flex flex-col overflow-hidden', className ?? ''].join(' ')}
          style={{ width: openSize, height: '100%' }}
        >
          <div className="flex-1 overflow-y-auto overflow-x-hidden px-3.5 pt-14 pb-4">{children}</div>
        </div>

        {/* Compact edge tab: text + arrow only, vertically centred (small hit area). */}
        <button
          {...handlers}
          type="button"
          onClick={toggle}
          aria-expanded={visible}
          className={[TAB_BASE, 'self-center rounded-r-xl flex-col gap-1.5 py-2.5'].join(' ')}
          style={{ width: LEFT_TAB_WIDTH }}
        >
          <span
            className="text-[11px] font-semibold tracking-[0.15em] text-sky-300/90"
            style={{ writingMode: 'vertical-rl', textOrientation: 'upright' }}
          >
            {tab}
          </span>
          <span className="text-slate-500 transition-transform group-hover:translate-x-0.5">
            {visible ? '‹' : '›'}
          </span>
          {pinDot}
        </button>
      </div>
    );
  }

  // side === 'bottom'
  return (
    <div
      className="absolute bottom-0 left-0 z-20 flex w-full flex-col items-center transition-transform duration-300 ease-out"
      style={{ transform: `translateY(${visible ? 0 : openSize}px)` }}
    >
      {/* Compact edge tab: text + arrow only, horizontally centred (small hit area). */}
      <button
        {...handlers}
        type="button"
        onClick={toggle}
        aria-expanded={visible}
        className={[TAB_BASE, 'self-center rounded-t-xl border-b-0 gap-2 px-4'].join(' ')}
        style={{ height: BOTTOM_TAB_HEIGHT }}
      >
        <span className="text-slate-500">{visible ? '▾' : '▴'}</span>
        <span className="text-[12px] font-semibold tracking-wider text-sky-300/90">{tab}</span>
        {pinDot}
      </button>

      <div
        {...handlers}
        className={[PANEL_CLASS, 'flex w-full flex-col overflow-hidden border-t-0', className ?? ''].join(' ')}
        style={{ height: openSize }}
      >
        <div className="flex-1 overflow-hidden">{children}</div>
      </div>
    </div>
  );
}
