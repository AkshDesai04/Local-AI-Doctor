import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import App from "./App";

describe("workbench boot states", () => {
  it("shows an honest offline state when every local endpoint is unreachable", async () => {
    vi.stubGlobal("fetch", vi.fn().mockRejectedValue(new TypeError("connection refused")));
    render(<App />);

    await waitFor(() => expect(screen.getByRole("heading", { name: "The backend is offline" })).toBeInTheDocument());
    expect(screen.getByText(/will not fall back to a remote inference provider/i)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Send message" })).toBeDisabled();
    expect(screen.getByText(/No placeholder measurements are shown/i)).toBeInTheDocument();
  });

  it("opens tab-scoped token settings when the backend returns 401", async () => {
    vi.stubGlobal("fetch", vi.fn().mockImplementation(() => Promise.resolve(new Response(
      JSON.stringify({ error: { message: "authentication required" } }),
      { status: 401, headers: { "Content-Type": "application/json" } },
    ))));
    render(<App />);

    expect(await screen.findByLabelText("Authentication token")).toHaveAttribute("type", "password");
    expect(screen.getByText(/stays in memory and session storage for this browser tab only/i)).toBeInTheDocument();
  });

  it("shows an embedding model as loaded after inference auto-loads it", async () => {
    let loaded = false;
    let healthGets = 0;
    let modelGets = 0;
    const json = (value: unknown): Response => new Response(JSON.stringify(value), {
      status: 200,
      headers: { "Content-Type": "application/json" },
    });
    const fetchMock = vi.fn((input: RequestInfo | URL, init?: RequestInit): Promise<Response> => {
      const url = input instanceof Request ? input.url : String(input);
      const method = init?.method ?? (input instanceof Request ? input.method : "GET");
      if (method === "GET" && url.endsWith("/health")) {
        healthGets += 1;
        return Promise.resolve(json({ status: "ok", loaded_model: loaded ? { model_id: "embedding-fixture", device: "cpu" } : null }));
      }
      if (method === "GET" && url.endsWith("/models")) {
        modelGets += 1;
        return Promise.resolve(json({ models: [{
          id: "embedding-fixture",
          display_name: "Embedding fixture",
          architectures: ["SentenceTransformer"],
          task: "embedding",
          fingerprint: { value: "fixture-fingerprint" },
          capabilities: { entries: { embeddings: { state: "full" }, cpu: { state: "full" } } },
          effective_context_limit: null,
        }] }));
      }
      if (method === "GET" && url.includes("/chats?archived=")) return Promise.resolve(json({ chats: [] }));
      if (method === "GET" && url.endsWith("/configuration")) return Promise.resolve(json({ effective: {}, precedence: [] }));
      if (method === "POST" && url.endsWith("/embeddings")) {
        loaded = true;
        return Promise.resolve(json({
          run_id: "embedding-run",
          model_id: "embedding-fixture",
          status: "complete",
          results: [{ input_id: "input-1", vector: [1, 0], output_dimension: 2, output_dtype: "float32", l2_norm: 1 }],
          output_dimensions: 2,
          normalized: true,
        }));
      }
      return Promise.reject(new Error(`Unexpected request: ${method} ${url}`));
    });
    vi.stubGlobal("fetch", fetchMock);
    render(<App />);

    await screen.findByText("Embedding fixture");
    fireEvent.click(screen.getByRole("button", { name: "Embeddings" }));
    fireEvent.change(await screen.findByLabelText("Text for Item 1"), { target: { value: "local vectors" } });
    fireEvent.click(screen.getByRole("button", { name: "Embed 1 item" }));

    await waitFor(() => expect(screen.getByText("cpu · ready")).toBeInTheDocument());
    expect(screen.getByRole("button", { name: "Unload" })).toBeEnabled();
    expect(healthGets).toBe(2);
    expect(modelGets).toBe(2);
  });
});
