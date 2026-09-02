import type { ReactNode } from 'react';

// Accessible toggle switch (role="switch") with an optional trailing hint.
interface SwitchProps {
  checked: boolean;
  onChange: (next: boolean) => void;
  label: ReactNode;
  hint?: ReactNode;
  disabled?: boolean;
}

export function Switch({ checked, onChange, label, hint, disabled }: SwitchProps) {
  return (
    <div className="flex flex-col gap-1">
      <button
        type="button"
        role="switch"
        aria-checked={checked}
        disabled={disabled}
        onClick={() => onChange(!checked)}
        className="group flex w-full items-center justify-between gap-3 rounded-lg px-2 py-1.5 text-left text-[13px] text-ink-200 transition-colors hover:bg-white/[0.04] disabled:cursor-not-allowed disabled:opacity-40"
      >
        <span className="min-w-0 flex-1 leading-snug">{label}</span>
        <span
          className={`relative inline-flex h-[18px] w-8 shrink-0 items-center rounded-full ring-1 transition-colors duration-200 ${
            checked ? 'bg-brand ring-brand/60' : 'bg-ink-600 ring-white/[0.06]'
          }`}
        >
          <span
            className={`absolute left-[2px] h-[14px] w-[14px] rounded-full shadow-sm shadow-black/50 transition-transform duration-200 ${
              checked ? 'translate-x-[14px] bg-ink-950' : 'translate-x-0 bg-ink-200'
            }`}
          />
        </span>
      </button>
      {hint && <div className="px-2 text-[11.5px] leading-relaxed text-ink-400">{hint}</div>}
    </div>
  );
}
