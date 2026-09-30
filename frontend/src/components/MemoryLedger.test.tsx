import { fireEvent, render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import type { MemoryLedger, ModelSummary, ResidentModel } from "../api/types";
import { ledgerSegments } from "../domain/residency";
import { MemoryLedgerCard } from "./MemoryLedger";

const gib = 1024 ** 3;
const mib = 1024 ** 2;

const ledger: MemoryLedger = {
  device: "cuda:0",
  totalBytes: 8 * gib,
  freeBytes: 3 * gib,
  torchAllocatedBytes: 3.5 * gib,
  torchReservedBytes: 4 * gib,
  capBytes: 7 * gib,
  processRssBytes: gib,
  systemAvailableBytes: 12 * gib,
  safetyMarginBytes: 512 * mib,
  ledgerAgeSeconds: 1.5,
  stale: false,
};

function resident(key: string, modelId: string, gpuBytes: number, overrides: Partial<ResidentModel> = {}): ResidentModel {
  return {
    modelKey: key,
    modelId,
    displayName: null,
    device: "cuda:0",
    dtype: "bfloat16",
    quantization: "none",
    strictVram: true,
    placement: "gpu",
    gpuBytes,
    cpuBytes: 0,
    kvReserveBytes: 0,
    loadSeconds: 1,
    lastUsedAt: null,
    inUse: false,
    ...overrides,
  };
}

const models = [
  { id: "model-a", name: "Model A" },
  { id: "model-b", name: "Model B" },
] as ModelSummary[];

const residents = [resident("key-a", "model-a", 2 * gib), resident("key-b", "model-b", 1.5 * gib, { placement: "offload", dtype: "float16" })];

describe("memory ledger segments", () => {
  it("splits device memory into residents, other use, reserved cache, margin, and free", () => {
    const segments = ledgerSegments(ledger, residents, models);
    expect(segments?.map((segment) => [segment.kind, segment.label, segment.bytes])).toEqual([
      ["resident", "Model A", 2 * gib],
      ["resident", "Model B", 1.5 * gib],
      ["other", "Other GPU use", 1 * gib],
      ["reserved", "Reserved by PyTorch", 0.5 * gib],
      ["margin", "Safety margin", 512 * mib],
      ["free", "Free", 2.5 * gib],
    ]);
    expect(segments?.reduce((sum, segment) => sum + segment.percent, 0)).toBeCloseTo(100);
    expect(segments?.map((segment) => segment.color)).toEqual([1, 2, undefined, undefined, undefined, undefined]);
    expect(segments?.[1]?.detail).toBe("cuda:0 · fp16 · offloaded");
  });

  it("drops empty segments, scales an overshooting sum, and needs a total", () => {
    const tight = ledgerSegments({ ...ledger, freeBytes: 0, torchReservedBytes: null, torchAllocatedBytes: null }, [resident("key-a", "model-a", 9 * gib)], models);
    expect(tight?.map((segment) => segment.kind)).toEqual(["resident"]);
    expect(tight?.[0]?.percent).toBeCloseTo(100);
    expect(ledgerSegments({ ...ledger, totalBytes: null }, residents, models)).toBeNull();
  });
});

describe("MemoryLedgerCard", () => {
  it("renders the bar legend with sizes, details on focus, and the resident count", async () => {
    const user = userEvent.setup();
    render(<MemoryLedgerCard connected maxLoadedModels={4} memory={ledger} models={models} onUnloadAll={vi.fn()} residents={residents} />);

    const card = screen.getByRole("region", { name: "Resident memory" });
    expect(within(card).getByText("GPU memory")).toBeInTheDocument();
    expect(within(card).getByText("2 of 4 resident")).toBeInTheDocument();
    const legend = within(card).getAllByRole("listitem");
    expect(legend.map((item) => item.textContent)).toEqual([
      "Model A2 GiB",
      "Model B1.5 GiB",
      "Other GPU use1 GiB",
      "Reserved by PyTorch512 MiB",
      "Safety margin512 MiB",
      "Free2.5 GiB",
    ]);

    await user.tab();
    await user.tab();
    await user.tab();
    expect(document.activeElement).toBe(legend[1]);
    expect(within(card).getByText(/^Model B: 1.5 GiB \(18.8%\) · cuda:0 · fp16 · offloaded$/)).toBeInTheDocument();
    fireEvent.mouseEnter(legend[5] as HTMLElement);
    expect(within(card).getByText(/^Free: 2.5 GiB/)).toBeInTheDocument();
    expect(card.querySelector(".ledger-cap")).toHaveStyle({ left: "87.5%" });
  });

  it("asks before unloading every resident model", async () => {
    const user = userEvent.setup();
    const onUnloadAll = vi.fn();
    render(<MemoryLedgerCard connected maxLoadedModels={4} memory={ledger} models={models} onUnloadAll={onUnloadAll} residents={residents} />);

    await user.click(screen.getByRole("button", { name: "Unload all" }));
    const dialog = screen.getByRole("alertdialog", { name: "Unload every resident model?" });
    expect(dialog).toHaveTextContent("Unload 2 resident copies");
    await user.click(within(dialog).getByRole("button", { name: "Cancel" }));
    expect(onUnloadAll).not.toHaveBeenCalled();

    await user.click(screen.getByRole("button", { name: "Unload all" }));
    await user.click(within(screen.getByRole("alertdialog")).getByRole("button", { name: "Unload all" }));
    expect(onUnloadAll).toHaveBeenCalledOnce();
  });

  it("explains a CPU-only or missing ledger and disables Unload all with nothing resident", () => {
    const { rerender } = render(<MemoryLedgerCard connected maxLoadedModels={null} memory={{ ...ledger, device: "cpu", totalBytes: null }} models={models} onUnloadAll={vi.fn()} residents={[]} />);
    expect(screen.getByText("No CUDA device ledger; models run on the CPU.")).toBeInTheDocument();
    expect(screen.getByText("0 resident")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Unload all" })).toBeDisabled();
    expect(screen.queryByRole("list", { name: "GPU memory breakdown" })).not.toBeInTheDocument();

    rerender(<MemoryLedgerCard connected maxLoadedModels={null} memory={null} models={models} onUnloadAll={vi.fn()} residents={[]} />);
    expect(screen.getByText("This backend does not report a memory ledger.")).toBeInTheDocument();
  });
});
