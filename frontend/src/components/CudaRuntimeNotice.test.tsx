import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { CudaRuntimeNotice } from "./CudaRuntimeNotice";

function hardware(options: { requested?: string; runtimeAvailable: boolean }): Response {
  return new Response(JSON.stringify({
    inventory: {
      accelerators: [{
        backend: "cuda",
        index: 0,
        name: "NVIDIA GeForce RTX 4060 Laptop GPU",
        runtime_available: options.runtimeAvailable,
        runtime_reason: options.runtimeAvailable ? null : "device found by nvidia-smi but no usable PyTorch CUDA runtime was detected",
      }],
    },
    selection: { requested: options.requested ?? "auto", selected_backend: options.runtimeAvailable ? "cuda" : "cpu" },
  }), { status: 200, headers: { "Content-Type": "application/json" } });
}

describe("CUDA runtime notice", () => {
  it("warns when nvidia-smi finds a GPU that PyTorch cannot use, and can be dismissed", async () => {
    const user = userEvent.setup();
    const fetchMock = vi.fn<typeof fetch>().mockResolvedValue(hardware({ runtimeAvailable: false }));
    vi.stubGlobal("fetch", fetchMock);
    render(<CudaRuntimeNotice connected />);

    expect(await screen.findByText("This build's PyTorch has no usable CUDA runtime, so models run on the CPU.")).toBeInTheDocument();
    expect(screen.getByRole("status")).toHaveTextContent("NVIDIA GeForce RTX 4060 Laptop GPU was found by nvidia-smi");
    expect(fetchMock).toHaveBeenCalledWith(expect.stringMatching(/\/hardware$/u), expect.anything());

    await user.click(screen.getByRole("button", { name: "Dismiss CUDA runtime warning" }));
    expect(screen.queryByRole("status")).not.toBeInTheDocument();
  });

  it("stays hidden when the CUDA runtime is usable", async () => {
    const fetchMock = vi.fn<typeof fetch>().mockResolvedValue(hardware({ runtimeAvailable: true }));
    vi.stubGlobal("fetch", fetchMock);
    render(<CudaRuntimeNotice connected />);
    await waitFor(() => expect(fetchMock).toHaveBeenCalledOnce());
    expect(screen.queryByRole("status")).not.toBeInTheDocument();
  });

  it("stays hidden for a CPU-only configuration, which skips the runtime probe", async () => {
    const fetchMock = vi.fn<typeof fetch>().mockResolvedValue(hardware({ requested: "cpu", runtimeAvailable: false }));
    vi.stubGlobal("fetch", fetchMock);
    render(<CudaRuntimeNotice connected />);
    await waitFor(() => expect(fetchMock).toHaveBeenCalledOnce());
    expect(screen.queryByRole("status")).not.toBeInTheDocument();
  });

  it("tolerates a failed hardware request and does not fetch while offline", async () => {
    const fetchMock = vi.fn<typeof fetch>().mockRejectedValue(new TypeError("connection refused"));
    vi.stubGlobal("fetch", fetchMock);
    const { rerender } = render(<CudaRuntimeNotice connected={false} />);
    expect(fetchMock).not.toHaveBeenCalled();

    rerender(<CudaRuntimeNotice connected />);
    await waitFor(() => expect(fetchMock).toHaveBeenCalledOnce());
    expect(screen.queryByRole("status")).not.toBeInTheDocument();
  });
});
