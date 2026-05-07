/** @type {import('tailwindcss').Config} */
module.exports = {
  content: ["./app/**/*.{js,ts,jsx,tsx}", "./components/**/*.{js,ts,jsx,tsx}"],
  theme: {
    extend: {
      colors: {
        canvas: "var(--canvas)",
        surface: "var(--surface)",
        "surface-soft": "var(--surface-soft)",
        line: "var(--line)",
        ink: "var(--ink)",
        muted: "var(--muted)",
        teal: "var(--teal)",
        indigo: "var(--indigo)",
        amber: "var(--amber)",
        rose: "var(--rose)"
      },
      fontFamily: {
        mono: ["SFMono-Regular", "Consolas", "Liberation Mono", "monospace"],
        sans: ["Inter", "Segoe UI", "Arial", "sans-serif"]
      },
      boxShadow: {
        panel: "0 1px 2px rgba(16, 24, 40, 0.05)"
      }
    }
  },
  plugins: []
};
