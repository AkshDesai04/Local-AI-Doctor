import ReactMarkdown from "react-markdown";
import rehypeKatex from "rehype-katex";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import "katex/dist/katex.min.css";
import { normalizeMathDelimiters, splitAssistantOutput } from "../utils/markdown";

function escapeLiteralReasoningTags(content: string): string {
  return content.split(/(```[\s\S]*?```|`[^`\r\n]*`)/gu).map((part, index) => index % 2 === 1
    ? part
    : part.replace(/<\/?think(?:\s[^>]*)?>/giu, (token) => token.replace("<", "\\<").replace(">", "\\>"))).join("");
}

function MarkdownBlock({ content }: { content: string }): React.ReactNode {
  const visibleContent = escapeLiteralReasoningTags(content);
  return <ReactMarkdown rehypePlugins={[rehypeKatex]} remarkPlugins={[remarkGfm, remarkMath]} skipHtml>{normalizeMathDelimiters(visibleContent)}</ReactMarkdown>;
}

export function MarkdownMessage({ content, assistant = false, reasoningPrimed = false }: { content: string; assistant?: boolean; reasoningPrimed?: boolean }): React.ReactNode {
  if (!assistant) return <div className="markdown-message"><MarkdownBlock content={content} /></div>;

  const output = splitAssistantOutput(content, reasoningPrimed);
  return (
    <div className="markdown-message">
      {output.hasReasoning && (
        <details className="reasoning-disclosure">
          <summary className="reasoning-summary">Thinking…</summary>
          <div className="reasoning-content">
            {output.reasoning ? <MarkdownBlock content={output.reasoning} /> : <p>Reasoning tokens are still arriving…</p>}
          </div>
        </details>
      )}
      {output.answer && <MarkdownBlock content={output.answer} />}
    </div>
  );
}
