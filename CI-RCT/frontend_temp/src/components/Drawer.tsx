import { useState, type ReactNode } from 'react';

// Reusable collapsible drawer primitive.
//
// Interaction: hovering the edge grip PEEKS the panel open; clicking the grip
// PINS it open (stays open after the mouse leaves) — click again to unpin.
// `open` is the controlled pinned state; transient hover is internal. The panel
// and its grip live in one wrapper that translates as a unit, so collapsing
// slides the panel off-screen while the grip stays fixed to the edge.

interface DrawerProps {
  side: 'left' | 'bottom';
  /** Pinned-open state (controlled). Hover peeking is handled internally. */
  open: boolean;
  /** Grip click toggles the pin → onOpenChange(!open). */
  onOpenChange: (open: boolean) => void;
  tab: ReactNode; // label shown on the edge grip
  hoverOpen?: boolean; // peek open on hover
  size?: number; // open width(px) for left / height(px) for bottom
  /** Distance from the top of the viewport reserved for the app bar (left side only). */
  topOffset?: number;
  children: ReactNode;
  className?: string;
}

const LEFT_DEFAULT_SIZE = 320;
const BOTTOM_DEFAULT_SIZE = 340;
const LEFT_GRIP_WIDTH = 22;
const BOTTOM_GRIP_HEIGHT = 26;

const GRIP_BASE =
  'group relative flex items-center justify-center select-none bg-ink-900/90 backdrop-blur-xl ' +
  'text-ink-400 ring-1 ring-white/[0.06] transition-colors duration-200 hover:bg-ink-800 hover:text-brand';

function Chevron({ dir }: { dir: 'left' | 'right' | 'up' | 'down' }) {
  const rotate = { right: 0, down: 90, left: 180, up: 270 }[dir];
  return (
    <svg width="10" height="10" viewBox="0 0 10 10" fill="none" style={{ transform: `rotate(${rotate}deg)` }} className="transition-transform duration-300">
      <path d="M3.5 1.5 7 5 3.5 8.5" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}

export function Drawer({
  side,
  open,
  onOpenChange,
  tab,
  hoverOpen = false,
  size,
  topOffset = 0,
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

  // Pin indicator: filled when pinned, hollow when only peeking.
  const pinDot = (
    <span
      className={`inline-block h-1.5 w-1.5 rounded-full transition-colors ${
        open ? 'bg-brand' : 'ring-1 ring-ink-400'
      }`}
      aria-hidden
    />
  );

  if (isLeft) {
    return (
      <div
        className="absolute left-0 z-20 flex items-center transition-transform duration-300 ease-out"
        style={{
          top: topOffset,
          height: `calc(100% - ${topOffset}px)`,
          transform: `translateX(${visible ? 0 : -openSize}px)`,
        }}
      >
        <aside
          {...handlers}
          aria-label={typeof tab === 'string' ? tab : undefined}
          className={['panel hairline flex h-full flex-col overflow-hidden border-r', className ?? ''].join(' ')}
          style={{ width: openSize }}
        >
          <div className="scroll-thin flex-1 overflow-y-auto overflow-x-hidden px-4 pb-6 pt-4">{children}</div>
        </aside>

        {/* Edge grip: vertical label + chevron, centred. */}
        <button
          {...handlers}
          type="button"
          onClick={toggle}
          aria-expanded={visible}
          aria-label={`${open ? '收合' : '展開'}${typeof tab === 'string' ? tab : ''}`}
          className={[GRIP_BASE, 'ml-[-1px] flex-col gap-2 rounded-r-lg py-3'].join(' ')}
          style={{ width: LEFT_GRIP_WIDTH }}
        >
          <span
            className="text-[10px] font-semibold tracking-[0.2em] text-ink-300 group-hover:text-brand"
            style={{ writingMode: 'vertical-rl', textOrientation: 'upright' }}
          >
            {tab}
          </span>
          <Chevron dir={visible ? 'left' : 'right'} />
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
      {/* Edge grip: pill with label + chevron, centred. */}
      <button
        {...handlers}
        type="button"
        onClick={toggle}
        aria-expanded={visible}
        aria-label={`${open ? '收合' : '展開'}${typeof tab === 'string' ? tab : ''}`}
        className={[GRIP_BASE, 'mb-[-1px] gap-2 rounded-t-lg px-4'].join(' ')}
        style={{ height: BOTTOM_GRIP_HEIGHT }}
      >
        <Chevron dir={visible ? 'down' : 'up'} />
        <span className="text-[11px] font-semibold tracking-[0.12em] text-ink-300 group-hover:text-brand">{tab}</span>
        {pinDot}
      </button>

      <section
        {...handlers}
        aria-label={typeof tab === 'string' ? tab : undefined}
        className={['panel hairline flex w-full flex-col overflow-hidden border-t', className ?? ''].join(' ')}
        style={{ height: openSize }}
      >
        <div className="flex-1 overflow-hidden">{children}</div>
      </section>
    </div>
  );
}
