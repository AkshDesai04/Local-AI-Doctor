import { defineConfig, devices } from "@playwright/test";

export default defineConfig({
  testDir: "./tests",
  fullyParallel: true,
  // Starting every desktop/mobile project at once can saturate the Vite
  // transform server on Windows and make page.goto() time out before a test
  // reaches its assertions. Two workers still exercise projects in parallel
  // while keeping the release gate deterministic on local and hosted runners.
  workers: 2,
  retries: 0,
  reporter: "list",
  use: {
    baseURL: "http://127.0.0.1:4173",
    trace: "retain-on-failure",
  },
  projects: [
    { name: "desktop-chromium", use: { ...devices["Desktop Chrome"] } },
    { name: "mobile-chromium", use: { ...devices["Pixel 7"] } },
  ],
  webServer: {
    command: "npm run dev -- --host 127.0.0.1 --port 4173",
    url: "http://127.0.0.1:4173",
    reuseExistingServer: true,
  },
});
