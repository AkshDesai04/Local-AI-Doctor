import type { Capability, CapabilityKey, ModelSummary } from "../api/types";

const labels: Record<CapabilityKey, string> = {
  text_generation: "Text generation",
  encoder_decoder_generation: "Encoder-decoder generation",
  embeddings: "Embeddings",
  vision: "Vision",
  audio: "Audio",
  video: "Video",
  native_file_input: "Native file input",
  extracted_text_input: "Extracted text",
  reasoning_segments: "Reasoning segments",
  moe_routing: "Expert routing",
  raw_logits: "Raw logits",
  processed_logits: "Processed logits",
  top_k_alternatives: "Top-K alternatives",
  prompt_scoring: "Prompt scoring",
  attention_capture: "Attention capture",
  hidden_state_capture: "Hidden states",
  streaming: "Streaming",
  batching: "Batching",
  deterministic_seed: "Deterministic seeding",
  cpu: "CPU",
  cuda: "CUDA",
  cpu_offload: "CPU offload",
};

export function capabilityLabel(key: CapabilityKey): string {
  return labels[key];
}

export function capabilityOf(model: ModelSummary | null | undefined, key: CapabilityKey): Capability {
  if (!model) return { state: "unsupported", reason: "Select a model to inspect this capability." };
  return model.capabilities[key] ?? { state: "unsupported", reason: "The model adapter did not report this capability." };
}

export function isUsable(model: ModelSummary | null | undefined, key: CapabilityKey): boolean {
  const state = capabilityOf(model, key).state;
  return state === "full" || state === "partial";
}

export function supportsGeneration(model: ModelSummary | null | undefined): boolean {
  return isUsable(model, "text_generation") || isUsable(model, "encoder_decoder_generation");
}

export function generationCapabilityReason(model: ModelSummary | null | undefined): string {
  if (isUsable(model, "text_generation")) return capabilityReason(model, "text_generation");
  if (isUsable(model, "encoder_decoder_generation")) return capabilityReason(model, "encoder_decoder_generation");
  return capabilityReason(model, model?.task === "encoder_decoder_generation" ? "encoder_decoder_generation" : "text_generation");
}

export function capabilityReason(model: ModelSummary | null | undefined, key: CapabilityKey): string {
  const capability = capabilityOf(model, key);
  if (capability.state === "full") return "Supported by this model and backend.";
  return capability.reason ?? capability.state.replaceAll("_", " ");
}

export const capabilityKeys = Object.keys(labels) as CapabilityKey[];
