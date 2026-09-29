import type { GenerationSettings } from "../api/types";

export interface NumericRange {
  min: number;
  max: number;
  step: number;
}

type NumericSetting = {
  [K in keyof GenerationSettings]-?: GenerationSettings[K] extends number ? K : never;
}[keyof GenerationSettings];

/**
 * Editable bounds for every numeric generation setting. The composer's quick
 * controls and the full generation controls both read these, so the two editors
 * can never disagree. Each range stays inside the backend's request validation
 * (for example Top-P and the repetition penalty must be greater than zero).
 */
export const SAMPLING_LIMITS: Readonly<Record<NumericSetting, NumericRange>> = {
  maxOutputTokens: { min: 1, max: 32768, step: 1 },
  temperature: { min: 0, max: 5, step: 0.05 },
  topK: { min: 0, max: 1000, step: 1 },
  topP: { min: 0.01, max: 1, step: 0.01 },
  minP: { min: 0, max: 1, step: 0.01 },
  repetitionPenalty: { min: 0.05, max: 3, step: 0.05 },
  frequencyPenalty: { min: -2, max: 2, step: 0.05 },
  presencePenalty: { min: -2, max: 2, step: 0.05 },
  alternatives: { min: 0, max: 1000, step: 1 },
};
