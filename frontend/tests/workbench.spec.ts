import { expect, test, type Page } from "@playwright/test";

const model = {
  id: "fixture-generation",
  name: "Deterministic UI Fixture",
  architecture: "FixtureForCausalLM",
  task: "text_generation",
  fingerprint: "fixture-only",
  lifecycle: "unloaded",
  effectiveContextLimit: 4096,
  capabilities: {
    text_generation: { state: "full" },
    embeddings: { state: "unsupported", reason: "Generation-only fixture." },
    vision: { state: "unsupported", reason: "Text-only fixture." },
    audio: { state: "unsupported", reason: "Text-only fixture." },
    video: { state: "unsupported", reason: "Text-only fixture." },
    raw_logits: { state: "full" },
    streaming: { state: "full" },
    reasoning_segments: { state: "full" },
    moe_routing: { state: "unsupported", reason: "Dense fixture." },
  },
};

async function fixtureApi(page: Page): Promise<void> {
  await page.route("**/api/v1/**", async (route) => {
    const url = route.request().url();
    if (url.endsWith("/health")) {
      await route.fulfill({ json: { status: "ok", version: "test", selectedBackend: "cpu" } });
    } else if (url.endsWith("/models")) {
      await route.fulfill({ json: [model] });
    } else if (url.includes("/chats?")) {
      await route.fulfill({ json: [] });
    } else {
      await route.fulfill({ status: 404, json: { message: "Not part of this UI fixture." } });
    }
  });
}

async function conversationFixtureApi(page: Page): Promise<void> {
  await page.route("**/api/v1/**", async (route) => {
    const url = route.request().url();
    if (url.endsWith("/health")) {
      await route.fulfill({ json: { status: "ok", worker: "ready" } });
    } else if (url.endsWith("/models")) {
      await route.fulfill({ json: { models: [{
        ...model,
        task: "encoder_decoder_generation",
        capabilities: {
          ...model.capabilities,
          text_generation: { state: "unsupported", reason: "Encoder-decoder checkpoint." },
          encoder_decoder_generation: { state: "full" },
        },
      }] } });
    } else if (url.includes("/chats?archived=false")) {
      await route.fulfill({ json: [{ id: "chat-1", title: "QA chat", created_at: "2026-09-12T00:00:00Z", updated_at: "2026-09-12T00:00:02Z" }] });
    } else if (url.includes("/chats?archived=true")) {
      await route.fulfill({ json: [] });
    } else if (url.endsWith("/chats/chat-1/messages")) {
      await route.fulfill({ json: [
        { id: "user-1", chat_id: "chat-1", role: "user", content: "Show the result", status: "complete", created_at: "2026-09-12T00:00:00Z" },
        { id: "assistant-1", chat_id: "chat-1", parent_id: "user-1", run_id: "run-1", role: "assistant", content: "<think>hidden trace</think>## Final answer\n\n| State | Value |\n| --- | --- |\n| Ready | Yes |<｜end▁of▁sentence｜>", status: "complete", created_at: "2026-09-12T00:00:02Z" },
      ] });
    } else if (url.endsWith("/runs/run-1")) {
      await route.fulfill({ json: {
        id: "run-1",
        message_id: "assistant-1",
        model_id: "fixture-generation",
        status: "complete",
        created_at: "2026-09-12T00:00:00Z",
        received_at: "2026-09-12T00:00:00.000Z",
        queue_entered_at: "2026-09-12T00:00:00.100Z",
        queue_exited_at: "2026-09-12T00:00:00.300Z",
        first_token_at: "2026-09-12T00:00:00.800Z",
        generated_token_count: 2,
        reproducibility: { effective_seed: "7", dtype: "bfloat16", device: { device_identifier: "cuda:0", reason: "Selected compatible accelerator." } },
        summary: { conditional_response_perplexity: 1.25, decode_tokens_per_second: 12.5 },
        tokens: [{ token_index: 0, token_id: 1, piece: "Final", display_text: "Final" }, { token_index: 1, token_id: 2, piece: " answer", display_text: " answer" }],
      } });
    } else if (url.endsWith("/configuration")) {
      await route.fulfill({ json: { effective: {}, precedence: [] } });
    } else {
      await route.fulfill({ status: 404, json: { message: "Not part of this UI fixture." } });
    }
  });
}

const systemPromptChat = { id: "chat-sp", title: "Prompted chat", created_at: "2026-09-12T00:00:00Z", updated_at: "2026-09-12T00:00:02Z" };
let systemPromptState: { stored: string | null; patches: unknown[] } = { stored: null, patches: [] };

async function systemPromptFixtureApi(page: Page): Promise<void> {
  systemPromptState = { stored: null, patches: [] };
  await page.route("**/api/v1/**", async (route) => {
    const url = route.request().url();
    if (url.endsWith("/health")) {
      await route.fulfill({ json: { status: "ok", worker: "ready" } });
    } else if (url.endsWith("/models")) {
      await route.fulfill({ json: [model] });
    } else if (url.includes("/chats?archived=false")) {
      await route.fulfill({ json: [{ ...systemPromptChat, system_prompt: systemPromptState.stored }] });
    } else if (url.includes("/chats?archived=true")) {
      await route.fulfill({ json: [] });
    } else if (url.endsWith("/chats/chat-sp/messages")) {
      await route.fulfill({ json: [
        { id: "user-1", chat_id: "chat-sp", role: "user", content: "Say hello", status: "complete", created_at: "2026-09-12T00:00:00Z" },
        { id: "assistant-1", chat_id: "chat-sp", parent_id: "user-1", role: "assistant", content: "Bonjour", status: "complete", created_at: "2026-09-12T00:00:02Z" },
      ] });
    } else if (url.endsWith("/chats/chat-sp") && route.request().method() === "PATCH") {
      const body = route.request().postDataJSON() as { systemPrompt?: string | null };
      systemPromptState.patches.push(body);
      if ("systemPrompt" in body) systemPromptState.stored = body.systemPrompt ?? null;
      await route.fulfill({ json: { ...systemPromptChat, system_prompt: systemPromptState.stored } });
    } else if (url.endsWith("/configuration")) {
      await route.fulfill({ json: { effective: {}, precedence: [] } });
    } else {
      await route.fulfill({ status: 404, json: { message: "Not part of this UI fixture." } });
    }
  });
}

test.beforeEach(async ({ page }, testInfo) => {
  if (testInfo.title.includes("mobile chat cleans")) await conversationFixtureApi(page);
  else if (testInfo.title.includes("system prompt")) await systemPromptFixtureApi(page);
  else await fixtureApi(page);
  await page.goto("/");
});

test("desktop workbench exposes model controls and truthful empty telemetry", async ({ page }, testInfo) => {
  test.skip(testInfo.project.name !== "desktop-chromium", "desktop-only assertion");
  await expect(page.getByRole("heading", { name: "What should we inspect?" })).toBeVisible();
  await expect(page.getByRole("combobox", { name: "Selected model" })).toHaveValue(model.id);
  await expect(page.getByText("No run selected", { exact: true })).toBeVisible();
  await page.getByRole("button", { name: /Nerd Mode/i }).click();
  await expect(page.getByText("Token boundaries are visible.")).toBeVisible();
});

test("mobile navigation opens without clipping the primary workspace", async ({ page }, testInfo) => {
  test.skip(testInfo.project.name !== "mobile-chromium", "mobile-only assertion");
  await page.getByRole("button", { name: "Open navigation" }).click();
  await expect(page.getByRole("navigation", { name: "Workspaces" })).toBeVisible();
  await page.getByRole("button", { name: "Model registry" }).click();
  await expect(page.getByRole("heading", { name: "Model registry" })).toBeVisible();
});

test("mobile chat cleans protocol text and wraps the run summary", async ({ page }, testInfo) => {
  test.skip(testInfo.project.name !== "mobile-chromium", "mobile-only assertion");

  await expect(page.getByRole("heading", { name: "Final answer" })).toBeVisible();
  await expect(page.getByText("Thinking…")).toBeVisible();
  await expect(page.getByText("hidden trace")).not.toBeVisible();
  await expect(page.getByRole("textbox", { name: "Message" })).toBeEnabled();
  await page.getByRole("button", { name: "Open response details" }).click();
  await expect(page.getByText("cuda:0 / bfloat16")).toBeVisible();
  await page.getByRole("button", { name: "Close inspector" }).click();

  const summary = page.locator(".run-summary-strip");
  await expect(summary).toBeVisible();
  await expect(summary).toHaveCSS("flex-wrap", "wrap");
  const bounds = await summary.boundingBox();
  const viewport = page.viewportSize();
  if (!bounds || !viewport) throw new Error("Expected a measurable mobile run summary.");
  expect(bounds.x + bounds.width).toBeLessThanOrEqual(viewport.width + 0.5);
});

test("mobile prompt controls stay inside a narrow viewport", async ({ page }, testInfo) => {
  test.skip(testInfo.project.name !== "mobile-chromium", "mobile-only assertion");
  await page.setViewportSize({ width: 320, height: 360 });

  await page.getByRole("button", { name: "Prompt controls" }).click();
  await expect(page.getByRole("slider", { name: "Temperature" })).toBeVisible();
  await expect(page.getByRole("slider", { name: "Top K" })).toBeVisible();
  await expect(page.getByRole("slider", { name: "Top P" })).toBeVisible();

  const panel = page.getByLabel("Prompt controls panel");
  const bounds = await panel.boundingBox();
  const viewport = page.viewportSize();
  if (!bounds || !viewport) throw new Error("Expected measurable prompt controls.");
  expect(bounds.x).toBeGreaterThanOrEqual(-0.5);
  expect(bounds.x + bounds.width).toBeLessThanOrEqual(viewport.width + 0.5);
  expect(bounds.y).toBeGreaterThanOrEqual(-0.5);
  expect(bounds.y + bounds.height).toBeLessThanOrEqual(viewport.height + 0.5);
});

test("system prompt is saved per chat from the prompt controls and shown in the conversation", async ({ page }) => {
  await expect(page.getByText("Bonjour")).toBeVisible();
  await expect(page.getByRole("img", { name: "System prompt active" })).toHaveCount(0);
  await expect(page.locator(".system-prompt-card")).toHaveCount(0);

  await page.getByRole("button", { name: "Prompt controls" }).click();
  const field = page.getByRole("textbox", { name: "System prompt" });
  await expect(field).toHaveAttribute("placeholder", /applied to every response/);
  await field.fill("Answer in French.");
  await expect.poll(() => systemPromptState.patches).toEqual([{ systemPrompt: "Answer in French." }]);
  await expect(page.getByRole("img", { name: "System prompt active" })).toBeVisible();

  const card = page.locator(".system-prompt-card");
  await expect(card).toBeVisible();
  await expect(card).not.toHaveAttribute("open", "");
  await card.locator("summary").click();
  await expect(card.getByText("Answer in French.")).toBeVisible();

  await page.reload();
  await expect(page.getByRole("img", { name: "System prompt active" })).toBeVisible();
  await expect(page.locator(".system-prompt-card")).toHaveCount(1);

  await page.getByRole("button", { name: "Prompt controls" }).click();
  await page.getByRole("textbox", { name: "System prompt" }).fill("");
  await expect.poll(() => systemPromptState.patches.at(-1)).toEqual({ systemPrompt: null });
  await expect(page.getByRole("img", { name: "System prompt active" })).toHaveCount(0);
  await expect(page.locator(".system-prompt-card")).toHaveCount(0);
});
