import { describe, expect, it } from "vitest";
import { displayTokenText, tokenTextHint } from "./format";

describe("token display formatting", () => {
  it("hides tokenizer whitespace marker glyphs from human-facing labels", () => {
    expect(displayTokenText("Ġsimple")).toBe("simple");
    expect(displayTokenText("▁sentence")).toBe("sentence");
    expect(displayTokenText("Ċanswer")).toBe("answer");
    expect(displayTokenText("<｜end▁of▁sentence｜>")).toBe("<｜end▁of▁sentence｜>");
  });

  it("keeps whitespace-only tokens compact and single-line", () => {
    expect(displayTokenText("\n")).toBe("↵");
    expect(displayTokenText("ĊĊ")).toBe("↵↵");
    expect(displayTokenText(" ")).toBe("␠");
    expect(displayTokenText("\t")).toBe("⇥");
  });

  it("preserves the exact raw tokenizer piece in hover metadata", () => {
    expect(tokenTextHint("Ġsimple", " simple")).toContain("Displayed text: simple");
    expect(tokenTextHint("Ġsimple", " simple")).toContain("Raw tokenizer piece: Ġsimple");
  });
});
