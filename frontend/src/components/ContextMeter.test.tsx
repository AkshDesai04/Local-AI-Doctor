import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { ContextMeter } from "./ContextMeter";

describe("ContextMeter", () => {
  it("reports unknown rather than inventing a context limit", () => {
    render(<ContextMeter context={undefined} fallbackLimit={null} onOpen={vi.fn()} />);
    expect(screen.getByRole("button", { name: "Context usage unknown" })).toBeInTheDocument();
    expect(screen.getByText("Context unknown")).toBeInTheDocument();
  });

  it("accounts for prompt, template, media, output, and reserve", () => {
    render(
      <ContextMeter
        context={{
          renderedPromptTokens: 100,
          templateTokens: 10,
          multimodalPositions: 20,
          generatedTokens: 30,
          reservedOutputTokens: 40,
          effectiveLimit: 1000,
        }}
        fallbackLimit={null}
        onOpen={vi.fn()}
      />,
    );
    expect(screen.getByRole("button", { name: "Context usage 20 percent" })).toBeInTheDocument();
    expect(screen.getByText("200 / 1,000")).toBeInTheDocument();
  });
});
