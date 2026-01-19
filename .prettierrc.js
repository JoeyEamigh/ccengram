/** @type {import('prettier').Config & import('prettier-plugin-tailwindcss').PluginOptions} */
export default {
  bracketSpacing: true,
  bracketSameLine: true,
  singleQuote: true,
  trailingComma: 'all',
  arrowParens: 'avoid',
  semi: true,
  plugins: ['prettier-plugin-organize-imports', 'prettier-plugin-tailwindcss'],
  tailwindStylesheet: './src/webui/globals.css',
  overrides: [
    {
      files: ['*.ts', '*.js', '*.tsx', '*.jsx', '*.cjs', '*.mjs'],
      options: {
        printWidth: 120,
      },
    },
  ],
};
