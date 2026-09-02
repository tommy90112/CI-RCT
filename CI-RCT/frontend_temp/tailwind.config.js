/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      fontFamily: {
        sans: ['Geist', 'Noto Sans TC', 'PingFang TC', 'Microsoft JhengHei', 'system-ui', 'sans-serif'],
        mono: ['Geist Mono', 'JetBrains Mono', 'Menlo', 'ui-monospace', 'monospace'],
      },
      colors: {
        // Chrome neutrals — one cool-tinted gray family for every surface.
        ink: {
          950: '#0b0d11',
          900: '#12151b',
          800: '#181c24',
          700: '#21262f',
          600: '#2c323d',
          500: '#3d4452',
          400: '#6b7382',
          300: '#98a1b3',
          200: '#c3c9d4',
          100: '#e7eaf0',
        },
        // Single interactive accent (selection, focus, active controls).
        brand: {
          DEFAULT: '#4fd1c5',
          soft: '#8ae4db',
          deep: '#2ba79c',
        },
        // Semantic risk colors — keep in sync with src/constants/risk.ts
        risk: {
          low:      '#10b981',
          midlow:   '#84cc16',
          mid:      '#f59e0b',
          high:     '#f97316',
          critical: '#ef4444',
        },
        relation: {
          r1: '#0ea5e9',
          r2: '#f59e0b',
          r3: '#10b981',
        },
        fraud: {
          positive: '#ef4444',
          negative: '#3b82f6',
          wallet:   '#8b5cf6',
        },
      },
      boxShadow: {
        panel: 'inset 0 1px 0 rgb(255 255 255 / 0.05), 0 30px 60px -30px rgb(4 8 16 / 0.9)',
        float: 'inset 0 1px 0 rgb(255 255 255 / 0.05), 0 12px 32px -12px rgb(4 8 16 / 0.85)',
      },
      keyframes: {
        'fade-in':   { '0%': { opacity: '0' }, '100%': { opacity: '1' } },
        'slide-up':  { '0%': { opacity: '0', transform: 'translateY(8px)' }, '100%': { opacity: '1', transform: 'translateY(0)' } },
        'pulse-ring': {
          '0%':   { transform: 'scale(0.95)', opacity: '0.7' },
          '100%': { transform: 'scale(1.6)',  opacity: '0' },
        },
      },
      animation: {
        'fade-in':   'fade-in 220ms ease-out both',
        'slide-up':  'slide-up 260ms ease-out both',
        'pulse-ring':'pulse-ring 1.8s ease-out infinite',
      },
    },
  },
  plugins: [],
}
