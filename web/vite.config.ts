import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/runs": process.env.SIMBENCH_API_URL ?? "http://127.0.0.1:8000",
    },
  },
});
