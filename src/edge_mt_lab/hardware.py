"""Hardware inspection and 2-4GB VRAM budget estimator for RTX 3060 8GB research."""

from __future__ import annotations

import os
import platform
import subprocess
from dataclasses import dataclass

from edge_mt_lab.schemas import ModelSpec, OCRPlacement, VRAMBreakdown

BYTES_PER_MIB = 1024.0 * 1024.0

OCR_VRAM_PRESETS_MIB: dict[OCRPlacement, float] = {
    "cpu_onnx": 0.0,
    "gpu_light_vlm": 1400.0,  # e.g., quantized PaddleOCR-VL (~0.9B) + runtime
    "unified_vlm": 450.0,  # vision encoder + projector overhead inside shared VLM
}


@dataclass(frozen=True)
class HostHardware:
    """Detected host hardware summary."""

    os_name: str
    machine: str
    cpu_cores: int
    gpu_name: str | None
    gpu_vram_total_mib: int | None
    gpu_vram_used_mib: int | None


def detect_host_hardware() -> HostHardware:
    """Inspect local CPU and NVIDIA GPU (if nvidia-smi is available)."""
    gpu_name: str | None = None
    vram_total: int | None = None
    vram_used: int | None = None

    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.used",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=3,
        )
        first_line = out.strip().splitlines()[0]
        parts = [part.strip() for part in first_line.split(",")]
        if len(parts) == 3:
            gpu_name = parts[0]
            vram_total = int(parts[1])
            vram_used = int(parts[2])
    except (FileNotFoundError, subprocess.SubprocessError, IndexError, ValueError):
        pass

    return HostHardware(
        os_name=platform.system(),
        machine=platform.machine(),
        cpu_cores=os.cpu_count() or 1,
        gpu_name=gpu_name,
        gpu_vram_total_mib=vram_total,
        gpu_vram_used_mib=vram_used,
    )


def estimate_vram_breakdown(
    spec: ModelSpec,
    *,
    llm_budget_gb: float = 4.0,
    total_gpu_vram_gb: float = 8.0,
    ocr_placement: OCRPlacement = "cpu_onnx",
    quant_metadata_overhead: float = 1.08,
    cuda_context_mib: float = 350.0,
    activation_overhead_mib: float = 220.0,
) -> VRAMBreakdown:
    """Estimate realistic VRAM usage (MiB) for weights, vocab pruning, KV/MLA, and OCR."""
    bytes_per_weight = (spec.weight_bits / 8.0) * quant_metadata_overhead
    raw_weight_bytes = spec.params_billions * 1e9 * bytes_per_weight

    vocab_savings_bytes = 0.0
    if spec.pruned_vocab_size is not None and spec.pruned_vocab_size < spec.vocab_size:
        removed_tokens = spec.vocab_size - spec.pruned_vocab_size
        embedding_matrices = 1 if spec.tied_embeddings else 2
        # Embeddings & LM head are typically stored at higher precision (~6-8 bit)
        embed_bytes_per_param = max(bytes_per_weight, 0.75)
        vocab_savings_bytes = (
            removed_tokens * spec.hidden_size * embedding_matrices * embed_bytes_per_param
        )

    effective_weight_mib = max(0.0, raw_weight_bytes - vocab_savings_bytes) / BYTES_PER_MIB
    vocab_savings_mib = vocab_savings_bytes / BYTES_PER_MIB

    if spec.attention_type == "mla":
        # DeepSeek-style MLA caches only compressed latent vector (+ small RoPE key slice)
        elements_per_token_per_layer = spec.mla_latent_dim + 64
    else:
        # Standard GQA/MHA stores both K and V tensors across kv heads
        elements_per_token_per_layer = 2 * spec.num_kv_heads * spec.head_dim

    kv_bytes = (
        spec.num_layers * spec.context_length * elements_per_token_per_layer * (spec.kv_bits / 8.0)
    )
    kv_cache_mib = kv_bytes / BYTES_PER_MIB

    llm_total_mib = effective_weight_mib + kv_cache_mib + activation_overhead_mib + cuda_context_mib
    ocr_vram_mib = OCR_VRAM_PRESETS_MIB[ocr_placement]
    system_total_mib = llm_total_mib + ocr_vram_mib

    return VRAMBreakdown(
        model_name=spec.name,
        weight_mib=round(effective_weight_mib, 1),
        vocab_savings_mib=round(vocab_savings_mib, 1),
        kv_cache_mib=round(kv_cache_mib, 1),
        activation_overhead_mib=round(activation_overhead_mib, 1),
        cuda_context_mib=round(cuda_context_mib, 1),
        llm_total_mib=round(llm_total_mib, 1),
        ocr_vram_mib=round(ocr_vram_mib, 1),
        system_total_mib=round(system_total_mib, 1),
        fits_llm_budget=llm_total_mib <= llm_budget_gb * 1024.0,
        fits_total_vram=system_total_mib <= total_gpu_vram_gb * 1024.0,
    )


def default_reference_profiles() -> list[ModelSpec]:
    """Curated Qwen / DeepSeek reference profiles under a 2-4GB LLM budget."""
    return [
        ModelSpec(
            name="Qwen3-4B @ 4-bit (GQA, KV8, 2K ctx)",
            params_billions=4.0,
            weight_bits=4.0,
            vocab_size=151_936,
            hidden_size=2560,
            num_layers=36,
            num_kv_heads=8,
            head_dim=128,
            attention_type="gqa",
            kv_bits=8.0,
            context_length=2048,
        ),
        ModelSpec(
            name="Qwen3-4B @ 4-bit + VocabTrim(48K) + TransMLA + KV4",
            params_billions=4.0,
            weight_bits=4.0,
            vocab_size=151_936,
            pruned_vocab_size=48_000,
            hidden_size=2560,
            num_layers=36,
            num_kv_heads=8,
            head_dim=128,
            attention_type="mla",
            mla_latent_dim=512,
            kv_bits=4.0,
            context_length=4096,
        ),
        ModelSpec(
            name="Qwen3-8B @ 2.5-bit (QTIP/Dynamic2.0) + KV4",
            params_billions=8.2,
            weight_bits=2.5,
            vocab_size=151_936,
            pruned_vocab_size=64_000,
            hidden_size=4096,
            num_layers=36,
            num_kv_heads=8,
            head_dim=128,
            attention_type="gqa",
            kv_bits=4.0,
            context_length=2048,
            tied_embeddings=False,
        ),
        ModelSpec(
            name="BitNet-b1.58-2B4T @ 1.58-bit (Ultra-Low Baseline)",
            params_billions=2.0,
            weight_bits=1.58,
            vocab_size=128_256,
            hidden_size=2560,
            num_layers=30,
            num_kv_heads=5,
            head_dim=128,
            attention_type="gqa",
            kv_bits=8.0,
            context_length=2048,
        ),
    ]
