import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it } from "vitest";
import { cleanAssistantOutput, splitAssistantOutput } from "../utils/markdown";
import { MarkdownMessage } from "./MarkdownMessage";

describe("MarkdownMessage", () => {
  it("places reasoning in a collapsed disclosure while hiding protocol tokens", async () => {
    const user = userEvent.setup();
    const raw = "<think>private scratch text</think>## Answer\n\n**Ready.**<｜end▁of▁sentence｜>";
    expect(cleanAssistantOutput(raw)).toBe("## Answer\n\n**Ready.**");
    render(<MarkdownMessage assistant content={raw} />);
    const summary = screen.getByText("Thinking…");
    const disclosure = summary.closest("details");
    expect(disclosure).not.toHaveAttribute("open");
    expect(screen.getByText(/private scratch text/i)).toBeInTheDocument();
    await user.click(summary);
    expect(disclosure).toHaveAttribute("open");
    expect(screen.getByRole("heading", { name: "Answer" })).toBeInTheDocument();
    expect(screen.getByText("Ready.").tagName).toBe("STRONG");
    expect(screen.queryByText(/end▁of▁sentence/i)).not.toBeInTheDocument();
  });

  it("renders GFM and math without executing raw HTML", () => {
    const { container } = render(<MarkdownMessage content={"<script>alert('no')</script>\n\n- [x] done\n\n$E = mc^2$\n\n\\[x^2 + y^2\\]"} />);
    expect(container.querySelector("script")).toBeNull();
    expect(screen.getByRole("checkbox")).toBeChecked();
    expect(container.querySelectorAll(".katex")).toHaveLength(2);
  });

  it("puts a template-primed reasoning stream in the Thinking disclosure", () => {
    const raw = "private emitted reasoning without an opening delimiter";
    expect(cleanAssistantOutput(raw, true)).toBe("");
    render(<MarkdownMessage assistant content={raw} reasoningPrimed />);
    expect(screen.getByText(raw)).toBeInTheDocument();
    expect(screen.getByText("Thinking…").closest("details")).not.toHaveAttribute("open");
  });

  it("splits prompt-primed reasoning, answer text, and termination tokens", () => {
    expect(splitAssistantOutput("step one</think>Answer<|eot_id|>", true)).toEqual({
      answer: "Answer",
      closingReasoningToken: "</think>",
      hasReasoning: true,
      openingReasoningToken: null,
      rawAnswer: "Answer<|eot_id|>",
      reasoning: "step one",
      terminationTokens: ["<|eot_id|>"],
    });
  });

  it("preserves inline literal think markup unless reasoning was explicitly primed", () => {
    const raw = "Describe `<think>` and then write <think>literally</think>.";
    const output = splitAssistantOutput(raw);

    expect(output.hasReasoning).toBe(false);
    expect(output.answer).toBe(raw);
    const { container } = render(<MarkdownMessage assistant content={raw} />);
    expect(container).toHaveTextContent("Describe <think> and then write <think>literally</think>.");
    expect(screen.queryByText("Thinking…")).not.toBeInTheDocument();
  });

  it("does not treat a stray closing delimiter as reasoning evidence", () => {
    const raw = "This example includes </think> as literal text.";
    expect(splitAssistantOutput(raw)).toMatchObject({ answer: raw, hasReasoning: false, reasoning: "" });
  });

  it("recognizes common termination-token spellings consistently", () => {
    const tokens = ["<|endoftext|>", "<|end|>", "<|eom_id|>", "<|end_of_text|>", "<|end_of_turn|>", "<end_of_turn>", "</s>"];
    const output = splitAssistantOutput(`Answer${tokens.join("")}`);

    expect(output.answer).toBe("Answer");
    expect(output.terminationTokens).toEqual(tokens);
  });

  it("does not invent a closing reasoning marker for an unfinished stream", () => {
    const output = splitAssistantOutput("<think>still working");

    expect(output).toMatchObject({
      closingReasoningToken: null,
      hasReasoning: true,
      openingReasoningToken: "<think>",
      reasoning: "still working",
    });
  });
});
