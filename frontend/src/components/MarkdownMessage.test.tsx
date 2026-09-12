import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { cleanAssistantOutput } from "../utils/markdown";
import { MarkdownMessage } from "./MarkdownMessage";

describe("MarkdownMessage", () => {
  it("hides reasoning delimiters and protocol terminators from the normal response", () => {
    const raw = "<think>private scratch text</think>## Answer\n\n**Ready.**<｜end▁of▁sentence｜>";
    expect(cleanAssistantOutput(raw)).toBe("## Answer\n\n**Ready.**");
    render(<MarkdownMessage assistant content={raw} />);
    expect(screen.queryByText(/private scratch text/i)).not.toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "Answer" })).toBeInTheDocument();
    expect(screen.getByText("Ready.").tagName).toBe("STRONG");
  });

  it("renders GFM and math without executing raw HTML", () => {
    const { container } = render(<MarkdownMessage content={"<script>alert('no')</script>\n\n- [x] done\n\n$E = mc^2$\n\n\\[x^2 + y^2\\]"} />);
    expect(container.querySelector("script")).toBeNull();
    expect(screen.getByRole("checkbox")).toBeChecked();
    expect(container.querySelectorAll(".katex")).toHaveLength(2);
  });

  it("withholds a template-primed reasoning stream until an answer segment exists", () => {
    const raw = "private emitted reasoning without an opening delimiter";
    expect(cleanAssistantOutput(raw, true)).toBe("");
    render(<MarkdownMessage assistant content={raw} reasoningPrimed />);
    expect(screen.queryByText(raw)).not.toBeInTheDocument();
    expect(screen.getByText(/no answer segment was emitted/i)).toBeInTheDocument();
  });
});
