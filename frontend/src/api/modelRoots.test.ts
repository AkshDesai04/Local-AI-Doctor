import { describe, expect, it, vi } from "vitest";
import { saveSessionAuthToken } from "./auth";
import { getModelRootSettings, updateModelRootSettings } from "./modelRoots";

function response(value: unknown, status = 200): Response {
  return new Response(JSON.stringify(value), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

describe("model root configuration boundary", () => {
  it("normalizes settings and authenticates durable updates", async () => {
    saveSessionAuthToken("local-token");
    const fetchMock = vi.fn<typeof fetch>()
      .mockResolvedValueOnce(response({
        model_roots: ["/models"],
        writable: true,
        source: "user-local configuration",
        containerized: true,
      }))
      .mockResolvedValueOnce(response({
        model_roots: ["/models/updated"],
        writable: true,
        source: "user-local configuration",
        containerized: true,
      }));
    vi.stubGlobal("fetch", fetchMock);

    await expect(getModelRootSettings()).resolves.toMatchObject({
      modelRoots: ["/models"],
      containerized: true,
    });
    await expect(updateModelRootSettings(["/models/updated"])).resolves.toMatchObject({
      modelRoots: ["/models/updated"],
    });

    const [url, options] = fetchMock.mock.calls.at(-1) ?? [];
    expect(url).toBe("/api/v1/configuration/model-roots");
    expect(options?.method).toBe("PUT");
    expect(new Headers(options?.headers).get("Authorization")).toBe("Bearer local-token");
    expect(options?.body).toBe(JSON.stringify({ model_roots: ["/models/updated"] }));
  });
});
