import type { ReactNode } from 'react';

// Generic segmented control; `value` is compared by identity.
interface SegmentedProps<T extends string | number> {
  value: T;
  options: { value: T; label: ReactNode; title?: string }[];
  onChange: (value: T) => void;
  className?: string;
  ariaLabel?: string;
}

export function Segmented<T extends string | number>({
  value,
  options,
  onChange,
  className,
  ariaLabel,
}: SegmentedProps<T>) {
  return (
    <div className={['seg', className ?? ''].join(' ')} role="tablist" aria-label={ariaLabel}>
      {options.map(opt => {
        const on = opt.value === value;
        return (
          <button
            key={String(opt.value)}
            type="button"
            role="tab"
            aria-selected={on}
            title={opt.title}
            onClick={() => onChange(opt.value)}
            className={['seg-item flex-1 whitespace-nowrap', on ? 'seg-item-on' : ''].join(' ')}
          >
            {opt.label}
          </button>
        );
      })}
    </div>
  );
}
