import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import {
  getModelRootSettings,
  updateModelRootSettings,
} from "../api/modelRoots";
import { ModelRootSettings } from "./ModelRootSettings";

vi.mock("../api/modelRoots", () => ({
  getModelRootSettings: vi.fn(),
  updateModelRootSettings: vi.fn(),
}));

const mockedGet = vi.mocked(getModelRootSettings);
const mockedUpdate = vi.mocked(updateModelRootSettings);

describe("model directory settings", () => {
  beforeEach(() => {
    mockedGet.mockReset().mockResolvedValue({
      modelRoots: ["/models"],
      writable: true,
      source: "user-local configuration",
      containerized: true,
    });
    mockedUpdate.mockReset().mockResolvedValue({
      modelRoots: ["/models/updated"],
      writable: true,
      source: "user-local configuration",
      containerized: true,
    });
  });

  it("saves backend-visible paths and explains Docker path semantics", async () => {
    const user = userEvent.setup();
    const onRefresh = vi.fn();
    render(<ModelRootSettings connected onRefresh={onRefresh} />);

    const input = await screen.findByLabelText("Backend-visible absolute paths, one per line");
    expect(input).toHaveValue("/models");
    expect(screen.getByText(/mounted container path, normally \/models/i)).toBeInTheDocument();

    await user.clear(input);
    await user.type(input, "/models/updated");
    await user.click(screen.getByRole("button", { name: "Save & rescan" }));

    await waitFor(() => expect(mockedUpdate).toHaveBeenCalledWith(["/models/updated"]));
    expect(onRefresh).toHaveBeenCalledOnce();
    expect(screen.getByText("Saved to the user-local configuration and rescanned.")).toBeInTheDocument();
  });

  it("disables edits when the configured source is not writable", async () => {
    mockedGet.mockResolvedValue({
      modelRoots: ["/models"],
      writable: false,
      source: "user-local configuration",
      reason: "The user-local configuration file is not writable by the backend.",
      containerized: true,
    });

    render(<ModelRootSettings connected onRefresh={vi.fn()} />);

    expect(await screen.findByRole("button", { name: "Save & rescan" })).toBeDisabled();
    expect(screen.getByRole("status")).toHaveTextContent("not writable");
  });

  it("requires the active model to be unloaded before changing roots", async () => {
    render(<ModelRootSettings connected modelLoaded onRefresh={vi.fn()} />);

    expect(await screen.findByRole("button", { name: "Save & rescan" })).toBeDisabled();
    expect(screen.getByText("Unload the active model before changing model directories.")).toBeInTheDocument();
  });
});
