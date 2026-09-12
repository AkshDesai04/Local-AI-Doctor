import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { AUTH_SESSION_KEY } from "../api/auth";
import { defaultGenerationSettings } from "../hooks/useWorkbench";
import { GenerationControls } from "./GenerationControls";

describe("runtime authentication settings", () => {
  it("stores a write-only token for the tab and clears it explicitly", async () => {
    const user = userEvent.setup();
    render(
      <GenerationControls
        authRequired
        model={null}
        onChange={vi.fn()}
        onClose={vi.fn()}
        open
        settings={defaultGenerationSettings}
      />,
    );

    const input = screen.getByLabelText("Authentication token");
    await user.type(input, "private-local-token");
    await user.click(screen.getByRole("button", { name: "Save for tab" }));

    expect(input).toHaveValue("");
    expect(window.sessionStorage.getItem(AUTH_SESSION_KEY)).toBe("private-local-token");
    expect(screen.queryByText("private-local-token")).not.toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "Clear" }));
    expect(window.sessionStorage.getItem(AUTH_SESSION_KEY)).toBeNull();
  });
});
