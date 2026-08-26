/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        // Semantic risk colors — keep in sync with src/constants/risk.ts
        risk: {
          low:      '#10b981', // emerald-500
          midlow:   '#84cc16', // lime-500
          mid:      '#f59e0b', // amber-500
          high:     '#f97316', // orange-500
          critical: '#ef4444', // red-500
        },
        relation: {
          r1: '#0ea5e9', // sky-500 wallet→user
          r2: '#f59e0b', // amber-500 user→user
          r3: '#10b981', // emerald-500 user→wallet
        },
        fraud: {
          positive: '#ef4444', // SHAP positive contribution / fraud
          negative: '#3b82f6', // SHAP negative contribution / normal
          wallet:   '#8b5cf6', // violet-500
        },
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
