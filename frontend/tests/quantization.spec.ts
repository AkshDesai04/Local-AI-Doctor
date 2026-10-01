import { expect, test, type Page } from "@playwright/test";

const GiB = 1024 ** 3;
const MiB = 1024 ** 2;

const model = {
  id: "fixture-generation",
  display_name: "Fixture-Model",
  architectures: ["FixtureForCausalLM"],
  task: "text_generation",
  fingerprint: { value: "fixture-only" },
  effective_context_limit: 4096,
  root_index: 0,
  metadata: { weight_quantization: null },
  capabilities: {
    entries: {
      text_generation: { state: "full" },
      cuda: { state: "full" },
      weight_quantization: { state: "partial", reason: "bitsandbytes NF4 4-bit / LLM.int8 8-bit at load time; CUDA only" },
    },
  },
};

const derived = {
  ...model,
  id: "fixture-generation-bnb-nf4",
  display_name: "Fixture-Model-nf4",
  fingerprint: { value: "fixture-derived" },
  root_index: 1,
  parameter_count: null,
  metadata: { weight_quantization: { method: "bitsandbytes", bits: 4, quant_type: "nf4" }, parameter_count_note: "packed quantized tensors" },
  derivation: { source_model_id: model.id, source_display_name: model.display_name, quantization: "bitsandbytes-4bit", created_at: "2026-10-01T12:00:00+00:00" },
  capabilities: {
    entries: {
      ...model.capabilities.entries,
      weight_quantization: { state: "partial", reason: "pre-quantized bitsandbytes checkpoint; loads as-is; cannot be re-quantized" },
    },
  },
};

const nf4Resident = {
  model_key: "key-nf4",
  model_id: model.id,
  display_name: model.display_name,
  device: "cuda:0",
  dtype: "bfloat16",
  quantization: "bitsandbytes-4bit",
  strict_vram: true,
  placement: "gpu",
  gpu_bytes: 1.5 * GiB,
  cpu_bytes: 0,
  kv_reserve_bytes: 256 * MiB,
  load_seconds: 26,
  last_used_at: "2026-10-01T12:00:00Z",
  in_use: false,
};

interface FixtureState {
  residents: typeof nf4Resident[];
  models: unknown[];
  loads: unknown[];
  flushes: unknown[];
}

let state: FixtureState = { residents: [], models: [], loads: [], flushes: [] };

async function quantizationFixtureApi(page: Page): Promise<void> {
  state = { residents: [], models: [model], loads: [], flushes: [] };
  await page.route("**/api/v1/**", async (route) => {
    const url = route.request().url();
    const method = route.request().method();
    if (url.endsWith("/health")) {
      await route.fulfill({ json: { status: "ok", worker: "ready", loaded_model: null, loaded_models: state.residents.map(({ model_key, model_id, device, dtype, quantization, placement, strict_vram }) => ({ model_key, model_id, device, dtype, quantization, placement, strict_vram })) } });
    } else if (url.endsWith("/models/resident")) {
      await route.fulfill({ json: {
        models: state.residents,
        memory: { device: "cuda:0", total_bytes: 8 * GiB, free_bytes: 6 * GiB, torch_allocated_bytes: 1.5 * GiB, torch_reserved_bytes: 1.6 * GiB, cap_bytes: 7 * GiB, process_rss_bytes: GiB, system_available_bytes: 10 * GiB, safety_margin_bytes: 512 * MiB, ledger_age_seconds: 0.5, stale: false },
        max_loaded_models: 4,
      } });
    } else if (url.endsWith(`/models/${model.id}/load`) && method === "POST") {
      state.loads.push(route.request().postDataJSON());
      state.residents = [nf4Resident];
      await route.fulfill({ json: { ...model, lifecycle: "loaded", loaded_device: "cuda:0", model_key: "key-nf4", placement: "gpu", quantization: "bitsandbytes-4bit", strict_vram: true, evicted_model_keys: [] } });
    } else if (url.endsWith("/models/resident/key-nf4/flush") && method === "POST") {
      state.flushes.push(route.request().postDataJSON());
      state.models = [model, derived];
      await route.fulfill({ status: 201, json: { model: derived, folder: "<model-root:1>/Fixture-Model-nf4", bytes_written: 1.4 * GiB, derivation: derived.derivation } });
    } else if (url.endsWith("/configuration/model-roots")) {
      await route.fulfill({ json: { model_roots: ["/models", "/scratch/models"], writable: true, source: "user-local configuration", reason: null, containerized: false } });
    } else if (url.endsWith("/models")) {
      await route.fulfill({ json: { models: state.models, roots: [] } });
    } else if (url.includes("/chats?")) {
      await route.fulfill({ json: [] });
    } else if (url.endsWith("/configuration")) {
      await route.fulfill({ json: { effective: { runtime: { device: "auto", dtype: "auto", strict_vram: true, quantization: "none", max_loaded_models: 4 } }, precedence: [] } });
    } else {
      await route.fulfill({ status: 404, json: { message: "Not part of this UI fixture." } });
    }
  });
}

test.beforeEach(async ({ page }) => {
  await quantizationFixtureApi(page);
  await page.goto("/");
});

test("quantize at load time, then flush the VRAM-only copy into a new model folder", async ({ page }, testInfo) => {
  if (testInfo.project.name === "mobile-chromium") await page.getByRole("button", { name: "Open navigation" }).click();
  await page.getByRole("button", { name: "Models" }).click();

  const panel = page.getByRole("group", { name: `Load options for ${model.display_name}` });
  await panel.getByRole("combobox", { name: "Quantization" }).selectOption("bitsandbytes-4bit");
  await panel.getByRole("button", { name: "Load" }).click();
  await expect.poll(() => state.loads).toEqual([{ device: "auto", dtype: "auto", strictVram: true, quantization: "bitsandbytes-4bit" }]);

  const copies = page.getByRole("list", { name: `Resident copies of ${model.display_name}` });
  await expect(copies.getByText("4-bit NF4 · VRAM only")).toBeVisible();
  await copies.getByRole("button", { name: `Flush ${model.display_name} 4-bit NF4 to storage` }).click();

  const dialog = page.getByRole("dialog", { name: "Flush to storage" });
  await expect(dialog.getByRole("combobox", { name: "Model root" })).toHaveValue("0");
  const folder = dialog.getByRole("textbox", { name: "Folder name" });
  await expect(folder).toHaveValue("Fixture-Model-bnb-nf4");
  await expect(dialog.getByText("About 1.5 GiB")).toBeVisible();
  await folder.fill("NUL");
  await expect(dialog.getByRole("alert")).toContainText("reserved on Windows");
  await expect(dialog.getByRole("button", { name: "Flush to storage" })).toBeDisabled();
  await folder.fill("Fixture-Model-nf4");
  await dialog.getByRole("combobox", { name: "Model root" }).selectOption("1");
  await dialog.getByRole("button", { name: "Flush to storage" }).click();

  await expect.poll(() => state.flushes).toEqual([{ targetRootIndex: 1, folderName: "Fixture-Model-nf4" }]);
  await expect(dialog).toHaveCount(0);
  await expect(page.getByText(/Saved Fixture-Model to <model-root:1>\/Fixture-Model-nf4/)).toBeVisible();
  const card = page.getByRole("article", { name: derived.display_name });
  await expect(card.getByText("Derived from Fixture-Model · 4-bit NF4")).toBeVisible();
  await expect(card.getByText("Pre-quantized 4-bit nf4")).toBeVisible();
  await expect(card.getByRole("combobox", { name: "Quantization" })).toBeDisabled();

  const bounds = await card.boundingBox();
  const viewport = page.viewportSize();
  if (!bounds || !viewport) throw new Error("Expected a measurable model card.");
  expect(bounds.x + bounds.width).toBeLessThanOrEqual(viewport.width + 0.5);
});
