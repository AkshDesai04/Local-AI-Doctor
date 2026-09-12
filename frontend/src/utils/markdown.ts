const specialTokenPattern = /<\|(?:assistant|user|system|endoftext|eot_id|im_end|end_of_turn|begin_of_text)\|>|<｜(?:end▁of▁sentence|begin▁of▁sentence)｜>|<\/?s>/giu;

export function cleanAssistantOutput(value: string, reasoningPrimed = false): string {
  const closingThink = /<\/think\s*>/giu;
  let lastClose = -1;
  for (const match of value.matchAll(closingThink)) lastClose = (match.index ?? -1) + match[0].length;
  if (lastClose >= 0) return value.slice(lastClose).replace(specialTokenPattern, "").trimStart();
  if (reasoningPrimed || /^\s*<think(?:\s[^>]*)?>/iu.test(value)) return "";
  return value.replace(/<\/?think(?:\s[^>]*)?>/giu, "").replace(specialTokenPattern, "").trimStart();
}

export function normalizeMathDelimiters(value: string): string {
  return value.split(/(```[\s\S]*?```)/gu).map((part, index) => {
    if (index % 2 === 1) return part;
    return part
      .replace(/\\\[([\s\S]*?)\\\]/gu, (_match, expression: string) => `\n$$\n${expression.trim()}\n$$\n`)
      .replace(/\\\((.*?)\\\)/gu, (_match, expression: string) => `$${expression}$`);
  }).join("");
}
