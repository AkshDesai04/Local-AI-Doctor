import { useEffect, useState } from "react";
import { api } from "../api/client";
import type { HardwareSummary } from "../api/types";
import { Callout } from "./ui";

/**
 * Warns when nvidia-smi reports an NVIDIA GPU but the backend's PyTorch has no
 * usable CUDA runtime, so the backend falls back to the CPU. Hardware discovery is
 * fetched once per connection; a failure simply shows nothing.
 */
export function CudaRuntimeNotice({ connected }: { connected: boolean }): React.ReactNode {
  const [hardware, setHardware] = useState<HardwareSummary | null>(null);
  const [dismissed, setDismissed] = useState(false);

  useEffect(() => {
    if (!connected) return undefined;
    let active = true;
    api.hardware().then((summary) => { if (active) setHardware(summary); }).catch(() => undefined);
    return () => { active = false; };
  }, [connected]);

  const nvidia = hardware?.accelerators.filter((device) => device.backend === "cuda") ?? [];
  // A CPU-only configuration skips the runtime probe, so its devices never report a runtime.
  const runtimeMissing = hardware !== null
    && hardware.requestedDevice !== "cpu"
    && nvidia.length > 0
    && nvidia.every((device) => !device.runtimeAvailable);
  if (!connected || dismissed || !runtimeMissing) return null;

  return (
    <Callout
      className="cuda-runtime-notice"
      dismissLabel="Dismiss CUDA runtime warning"
      onDismiss={() => setDismissed(true)}
      role="status"
      title="This build's PyTorch has no usable CUDA runtime, so models run on the CPU."
      tone="warning"
    >
      {nvidia.map((device) => device.name).join(", ")} was found by nvidia-smi. Install a CUDA-enabled PyTorch build that matches your driver, then restart the backend to run models on the GPU.
    </Callout>
  );
}
