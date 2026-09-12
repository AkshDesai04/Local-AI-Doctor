import ReactMarkdown from "react-markdown";
import rehypeKatex from "rehype-katex";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import "katex/dist/katex.min.css";
import { cleanAssistantOutput, normalizeMathDelimiters } from "../utils/markdown";

export function MarkdownMessage({ content, assistant = false, reasoningPrimed = false }: { content: string; assistant?: boolean; reasoningPrimed?: boolean }): React.ReactNode {
  const cleaned = assistant ? cleanAssistantOutput(content, reasoningPrimed) : content;
  const visible = normalizeMathDelimiters(cleaned);
  if (assistant && reasoningPrimed && content.trim() && !visible.trim()) {
    return <div className="markdown-message"><p className="reasoning-withheld">Reasoning trace hidden; no answer segment was emitted.</p></div>;
  }
  return (
    <div className="markdown-message">
      <ReactMarkdown rehypePlugins={[rehypeKatex]} remarkPlugins={[remarkGfm, remarkMath]} skipHtml>{visible}</ReactMarkdown>
    </div>
  );
}
