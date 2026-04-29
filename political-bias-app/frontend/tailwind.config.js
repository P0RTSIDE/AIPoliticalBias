/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,jsx}"],
  theme: {
    extend: {
      colors: {
        democrat: "#1a56db",
        republican: "#e02424",
        centrist: "#7e3af2",
      },
    },
  },
  plugins: [],
};
