import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Where the dev server forwards API calls. Override with SIMBENCH_API_URL to point at another API port.
const apiUrl = process.env.SIMBENCH_API_URL ?? "http://127.0.0.1:8000";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/runs": apiUrl,
      "/agents": apiUrl,
      "/scenarios": apiUrl,
    },
  },
});
