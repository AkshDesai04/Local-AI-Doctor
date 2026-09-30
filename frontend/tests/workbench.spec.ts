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

const GiB = 1024 ** 3;
const MiB = 1024 ** 2;

const emptyResidency = {
  models: [],
  memory: { device: "cpu", total_bytes: null, free_bytes: null, torch_allocated_bytes: null, torch_reserved_bytes: null, cap_bytes: null, process_rss_bytes: 512 * MiB, system_available_bytes: 8 * GiB, safety_margin_bytes: 512 * MiB, ledger_age_seconds: 0.5, stale: false },
  max_loaded_models: 4,
};

async function fixtureApi(page: Page): Promise<void> {
  await page.route("**/api/v1/**", async (route) => {
    const url = route.request().url();
    if (url.endsWith("/health")) {
      await route.fulfill({ json: { status: "ok", version: "test", selectedBackend: "cpu", loaded_model: null, loaded_models: [] } });
    } else if (url.endsWith("/models/resident")) {
      await route.fulfill({ json: emptyResidency });
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

const secondModel = { ...model, id: "fixture-second", name: "Second UI Fixture", fingerprint: "fixture-second" };
const residents = [
  { model_key: "key-first", model_id: model.id, display_name: model.name, device: "cuda:0", dtype: "bfloat16", quantization: "none", strict_vram: true, placement: "gpu", gpu_bytes: 2 * GiB, cpu_bytes: 96 * MiB, kv_reserve_bytes: 256 * MiB, load_seconds: 4.2, last_used_at: "2026-10-01T09:00:00Z", in_use: false },
  { model_key: "key-second", model_id: secondModel.id, display_name: secondModel.name, device: "cuda:0", dtype: "float16", quantization: "none", strict_vram: false, placement: "offload", gpu_bytes: 1.5 * GiB, cpu_bytes: GiB, kv_reserve_bytes: 256 * MiB, load_seconds: 9.8, last_used_at: "2026-10-01T09:05:00Z", in_use: false },
];
let loadRequests: unknown[] = [];

async function residencyFixtureApi(page: Page): Promise<void> {
  loadRequests = [];
  await page.route("**/api/v1/**", async (route) => {
    const url = route.request().url();
    if (url.endsWith("/health")) {
      await route.fulfill({ json: {
        status: "ok",
        worker: "ready",
        loaded_model: { model_id: secondModel.id, device: "cuda:0" },
        loaded_models: residents.map(({ model_key, model_id, device, dtype, quantization, placement, strict_vram }) => ({ model_key, model_id, device, dtype, quantization, placement, strict_vram })),
      } });
    } else if (url.endsWith("/models/resident")) {
      await route.fulfill({ json: {
        models: residents,
        memory: { device: "cuda:0", total_bytes: 8 * GiB, free_bytes: 3.5 * GiB, torch_allocated_bytes: 3.5 * GiB, torch_reserved_bytes: 4 * GiB, cap_bytes: 7 * GiB, process_rss_bytes: 1.2 * GiB, system_available_bytes: 10 * GiB, safety_margin_bytes: 512 * MiB, ledger_age_seconds: 1.2, stale: false },
        max_loaded_models: 4,
      } });
    } else if (url.endsWith(`/models/${model.id}/load`)) {
      loadRequests.push(route.request().postDataJSON());
      await route.fulfill({ json: { ...model, lifecycle: "loaded", loaded_device: "cuda:0", model_key: "key-first", placement: "gpu", strict_vram: false, evicted_model_keys: [] } });
    } else if (url.endsWith("/models")) {
      await route.fulfill({ json: [model, secondModel] });
    } else if (url.includes("/chats?")) {
      await route.fulfill({ json: [] });
    } else if (url.endsWith("/configuration")) {
      await route.fulfill({ json: { effective: { runtime: { device: "auto", dtype: "auto", strict_vram: true, max_loaded_models: 4 } }, precedence: [] } });
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
  else if (testInfo.title.includes("GPU memory ledger")) await residencyFixtureApi(page);
  else await fixtureApi(page);
  await page.goto("/");
});

test("desktop workbench exposes model controls and truthful empty telemetry", async ({ page }, testInfo) => {
  test.skip(testInfo.project.name !== "desktop-chromium", "desktop-only assertion");
  await expect(page.getByRole("heading", { name: "What should we inspect?" })).toBeVisible();
  await expect(page.getByRole("combobox", { name: "Selected model" })).toHaveValue(model.id);
  await expect(page.getByText("No run selected", { exact: true })).toBeVisible();
  await page.getByRole("switch", { name: "Nerd Mode" }).click();
  await expect(page.getByText("Token boundaries are visible.")).toBeVisible();
});

test("mobile navigation opens without clipping the primary workspace", async ({ page }, testInfo) => {
  test.skip(testInfo.project.name !== "mobile-chromium", "mobile-only assertion");
  await page.getByRole("button", { name: "Open navigation" }).click();
  await expect(page.getByRole("navigation", { name: "Workspaces" })).toBeVisible();
  await page.getByRole("button", { name: "Models" }).click();
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

test("laptop layout keeps the context meter inside the composer and inspector tab labels visible", async ({ page }, testInfo) => {
  test.skip(testInfo.project.name !== "desktop-chromium", "desktop-only assertion");
  await page.setViewportSize({ width: 1280, height: 800 });

  const composer = await page.locator(".composer").boundingBox();
  const meter = await page.getByRole("button", { name: /Context usage/ }).boundingBox();
  if (!composer || !meter) throw new Error("Expected a measurable composer and context meter.");
  expect(meter.x).toBeGreaterThanOrEqual(composer.x);
  expect(meter.y).toBeGreaterThanOrEqual(composer.y);
  expect(meter.x + meter.width).toBeLessThanOrEqual(composer.x + composer.width + 0.5);
  expect(meter.y + meter.height).toBeLessThanOrEqual(composer.y + composer.height + 0.5);

  await expect(page.getByRole("tab", { name: "Overview" })).toHaveText("Overview");
  await expect(page.getByRole("tab", { name: "Overview" })).toBeVisible();
  await expect(page.getByRole("tab", { name: "Experts" })).toHaveAttribute("aria-disabled", "true");
});

test("model registry shows the GPU memory ledger, load options, and every resident copy", async ({ page }, testInfo) => {
  const mobile = testInfo.project.name === "mobile-chromium";
  await expect(page.getByText("cuda:0 · fp16 · offload")).toHaveCount(1);
  if (!mobile) await expect(page.getByRole("meter", { name: "GPU memory in use" })).toHaveAttribute("aria-valuetext", "4.5 GiB of 8 GiB");
  if (mobile) await page.getByRole("button", { name: "Open navigation" }).click();
  await page.getByRole("button", { name: "Models" }).click();

  const ledger = page.getByRole("region", { name: "Resident memory" });
  await expect(ledger.getByRole("heading", { name: "GPU memory" })).toBeVisible();
  await expect(ledger.getByText("2 of 4 resident")).toBeVisible();
  const breakdown = ledger.getByRole("list", { name: "GPU memory breakdown" });
  await expect(breakdown.getByRole("listitem").filter({ hasText: model.name })).toContainText("2 GiB");
  await expect(breakdown.getByRole("listitem").filter({ hasText: "Safety margin" })).toContainText("512 MiB");
  await breakdown.getByRole("listitem").filter({ hasText: secondModel.name }).focus();
  await expect(ledger.getByText(/^Second UI Fixture: 1.5 GiB/)).toBeVisible();

  await expect(page.getByRole("list", { name: /^Resident copies of/ })).toHaveCount(2);
  const secondCopies = page.getByRole("list", { name: `Resident copies of ${secondModel.name}` });
  await expect(secondCopies.getByText("Offloaded to system RAM")).toBeVisible();
  await expect(secondCopies.getByRole("button", { name: /^Unload Second UI Fixture/ })).toBeEnabled();

  const panel = page.getByRole("group", { name: `Load options for ${model.name}` });
  await expect(panel.getByRole("combobox", { name: "Device" })).toHaveValue("auto");
  await expect(panel.getByRole("combobox", { name: "Data type" })).toHaveValue("auto");
  const strict = panel.getByRole("switch", { name: /Strict VRAM/ });
  await expect(strict).toBeChecked();
  await strict.click();
  await panel.getByRole("button", { name: "Load" }).click();
  await expect.poll(() => loadRequests).toEqual([{ device: "auto", dtype: "auto", strictVram: false }]);

  const bounds = await ledger.boundingBox();
  const viewport = page.viewportSize();
  if (!bounds || !viewport) throw new Error("Expected a measurable memory ledger.");
  expect(bounds.x + bounds.width).toBeLessThanOrEqual(viewport.width + 0.5);
});
