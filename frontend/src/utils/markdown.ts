const terminationTokenSource = String.raw`(?:<\|(?:endoftext|eot_id|im_end|end_of_turn|end_of_text|end|eom_id|eom|end_of_message|eos)\|>|<(?:end_of_turn|end_of_text|eos)>|<｜(?:end▁of▁sentence|end▁of▁turn)｜>|<\/s>)`;
const specialTokenSource = String.raw`(?:${terminationTokenSource}|<\|(?:assistant|user|system|begin_of_text)\|>|<｜begin▁of▁sentence｜>|<s>)`;
const leadingReasoningOpenPattern = /^(\s*)(<think(?:\s[^>]*)?>)/iu;
const reasoningClosePattern = /<\/think\s*>/iu;

function protocolTokenPattern(): RegExp {
  return new RegExp(specialTokenSource, "giu");
}

function terminationTokenPattern(): RegExp {
  return new RegExp(terminationTokenSource, "giu");
}

export function isTerminationToken(value: string): boolean {
  return new RegExp(terminationTokenSource, "iu").test(value);
}

export interface AssistantOutputSegments {
  answer: string;
  closingReasoningToken: string | null;
  hasReasoning: boolean;
  openingReasoningToken: string | null;
  rawAnswer: string;
  reasoning: string;
  terminationTokens: string[];
}

export function splitAssistantOutput(value: string, reasoningPrimed = false): AssistantOutputSegments {
  const openingMatch = leadingReasoningOpenPattern.exec(value);
  const openingReasoningToken = openingMatch?.[2] ?? null;
  const reasoningStart = openingMatch ? openingMatch[0].length : 0;
  const closeMatch = reasoningClosePattern.exec(value.slice(reasoningStart));
  const closingIndex = closeMatch ? reasoningStart + (closeMatch.index ?? 0) : -1;
  const closingReasoningToken = closeMatch?.[0] ?? null;
  const hasReasoning = reasoningPrimed || openingMatch !== null;
  const terminationTokens = Array.from(value.matchAll(terminationTokenPattern()), (match) => match[0]);

  if (!hasReasoning) {
    return {
      answer: value.replace(protocolTokenPattern(), "").trimStart(),
      closingReasoningToken: null,
      hasReasoning: false,
      openingReasoningToken: null,
      rawAnswer: value,
      reasoning: "",
      terminationTokens,
    };
  }

  const reasoningEnd = closingIndex >= 0 ? closingIndex : value.length;
  const rawAnswer = closeMatch
    ? value.slice(closingIndex + closeMatch[0].length)
    : "";
  return {
    answer: rawAnswer.replace(protocolTokenPattern(), "").trimStart(),
    closingReasoningToken,
    hasReasoning: true,
    openingReasoningToken,
    rawAnswer,
    reasoning: value
      .slice(Math.min(reasoningStart, reasoningEnd), reasoningEnd)
      .replace(protocolTokenPattern(), "")
      .trim(),
    terminationTokens,
  };
}

export function cleanAssistantOutput(value: string, reasoningPrimed = false): string {
  return splitAssistantOutput(value, reasoningPrimed).answer;
}

export function normalizeMathDelimiters(value: string): string {
  return value.split(/(```[\s\S]*?```)/gu).map((part, index) => {
    if (index % 2 === 1) return part;
    return part
      .replace(/\\\[([\s\S]*?)\\\]/gu, (_match, expression: string) => `\n$$\n${expression.trim()}\n$$\n`)
      .replace(/\\\((.*?)\\\)/gu, (_match, expression: string) => `$${expression}$`);
  }).join("");
}
