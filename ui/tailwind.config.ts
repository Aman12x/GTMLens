import type { Config } from "tailwindcss";

export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        bg:      "#0d1117",
        surface: "#161b22",
        overlay: "#1c2128",
        border:  "#30363d",
        accent:  "#58a6ff",
        green:   "#3fb950",
        yellow:  "#d29922",
        red:     "#f85149",
        text:    "#e6edf3",
        muted:   "#8b949e",
        faint:   "#484f58",
      },
      fontFamily: {
        sans: ["Inter", "system-ui", "sans-serif"],
        mono: ["JetBrains Mono", "Fira Code", "monospace"],
      },
      borderRadius: {
        DEFAULT: "0.5rem",
      },
    },
  },
  plugins: [],
} satisfies Config;
