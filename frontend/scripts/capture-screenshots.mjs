import { mkdir } from "node:fs/promises";
import { resolve } from "node:path";
import { chromium } from "@playwright/test";

const baseUrl = process.env.LAD_UI_URL ?? "http://127.0.0.1:8000";
const outputDirectory = resolve(import.meta.dirname, "../../docs/screenshots");
await mkdir(outputDirectory, { recursive: true });

const browser = await chromium.launch();
const page = await browser.newPage({ viewport: { width: 1440, height: 900 } });

try {
  await page.goto(baseUrl, { waitUntil: "networkidle" });
  await page.getByText("Local AI Doctor", { exact: true }).waitFor();

  const validatedChat = page.getByRole("button", { name: /Seeded arithmetic validation/ }).first();
  if (await validatedChat.count()) {
    await validatedChat.click();
    const details = page.getByRole("button", { name: "Open response details" }).last();
    if (await details.count()) {
      await details.click();
      await page.getByText("No run selected", { exact: true }).waitFor({ state: "hidden" });
    }
  }
  await page.screenshot({ path: resolve(outputDirectory, "chat-observability.png") });

  const closeInspector = page.getByRole("button", { name: "Close inspector", exact: true });
  if (await closeInspector.count()) await closeInspector.click();
  await page.getByRole("button", { name: "Embeddings", exact: true }).click();
  await page.getByRole("textbox", { name: "Text for Item 1" }).fill("Local model observability and reproducible inference");
  await page.getByRole("textbox", { name: "Text for Item 2" }).fill("Inspect reproducible local inference metrics");
  await page.getByRole("radio", { name: "64", exact: true }).click();
  await page.getByRole("button", { name: "Embed 2 items", exact: true }).click();
  await page.getByText("No embedding run yet", { exact: true }).waitFor({ state: "hidden", timeout: 120_000 });
  await page.screenshot({ path: resolve(outputDirectory, "embeddings-workspace.png") });

  await page.getByRole("button", { name: "Models", exact: true }).click();
  await page.waitForTimeout(250);
  await page.screenshot({ path: resolve(outputDirectory, "model-registry.png") });

  await page.setViewportSize({ width: 390, height: 844 });
  await page.waitForTimeout(250);
  await page.getByRole("button", { name: "Open navigation", exact: true }).click();
  await page.locator(".chat-list").evaluate((element) => {
    element.scrollTop = 0;
  });
  await page.waitForTimeout(350);
  await page.screenshot({ path: resolve(outputDirectory, "mobile-navigation.png") });
} finally {
  await browser.close();
}
