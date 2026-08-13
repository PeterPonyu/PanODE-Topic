import type { Config } from 'tailwindcss';

const config: Config = {
  content: ['./src/**/*.{js,ts,jsx,tsx,mdx}'],
  theme: {
    extend: {
      colors: {
        brand: '#2563eb',
        teal: '#0d9488',
      },
    },
  },
  plugins: [],
};

export default config;
