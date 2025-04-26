/** @type {import('tailwindcss').Config} */
// tailwind.config.js
module.exports = {
  content: [
    './app.vue',
    './pages/**/*.vue',
    './components/**/*.vue',
    './layouts/**/*.vue',
    './nuxt.config.{js,ts}',
  ],
  theme: { extend: {} },
  plugins: [],
}
