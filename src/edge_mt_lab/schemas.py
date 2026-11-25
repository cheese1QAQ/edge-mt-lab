"""Structured Pydantic schemas for datasets, VRAM budgets, and benchmark reports."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

SampleCategory = Literal["terminology", "ocr_noise", "formatting", "discourse"]
AttentionType = Literal["gqa", "mha", "mla"]
OCRPlacement = Literal["cpu_onnx", "gpu_light_vlm", "unified_vlm"]


class EvalSample(BaseModel):
    """Single multilingual evaluation sample for low-VRAM translation experiments."""

    id: str = Field(min_length=1)
    source_lang: str = Field(min_length=2)
    target_lang: str = Field(min_length=2)
    category: SampleCategory
    source: str = Field(min_length=1)
    reference: str = Field(min_length=1)
    glossary: dict[str, str] = Field(default_factory=dict)
    context_notes: str = ""
    ocr_corrupted: bool = False


class ModelSpec(BaseModel):
    """Parameter and architecture specification used for VRAM budget modeling."""

    name: str
    params_billions: float = Field(gt=0)
    weight_bits: float = Field(gt=0, le=32)
    vocab_size: int = Field(gt=0, default=151_936)
    pruned_vocab_size: int | None = Field(default=None, gt=0)
    hidden_size: int = Field(gt=0, default=2560)
    num_layers: int = Field(gt=0, default=36)
    num_kv_heads: int = Field(gt=0, default=8)
    head_dim: int = Field(gt=0, default=128)
    attention_type: AttentionType = "gqa"
    mla_latent_dim: int = Field(gt=0, default=512)
    kv_bits: float = Field(gt=0, le=16, default=8.0)
    context_length: int = Field(gt=0, default=2048)
    tied_embeddings: bool = True


class VRAMBreakdown(BaseModel):
    """Estimated VRAM consumption breakdown (in MiB) for an LLM + OCR co-deployment."""

    model_name: str
    weight_mib: float
    vocab_savings_mib: float
    kv_cache_mib: float
    activation_overhead_mib: float
    cuda_context_mib: float
    llm_total_mib: float
    ocr_vram_mib: float
    system_total_mib: float
    fits_llm_budget: bool
    fits_total_vram: bool


class SampleMetric(BaseModel):
    """Per-sample latency, glossary adherence, and placeholder fidelity result."""

    sample_id: str
    category: SampleCategory
    ocr_corrupted: bool
    ttft_ms: float
    total_ms: float
    output_tokens: int
    decode_tps: float
    glossary_expected: int
    glossary_matched: int
    placeholders_preserved: bool
    output_preview: str


class BenchmarkReport(BaseModel):
    """Structured JSON benchmark report written to experiments/reports/."""

    run_id: str
    timestamp_utc: str
    backend: str
    model_name: str
    dataset_path: str
    llm_budget_gb: float
    sample_count: int
    mean_ttft_ms: float
    mean_decode_tps: float
    glossary_adherence_rate: float
    placeholder_fidelity_rate: float
    peak_vram_mib: int | None
    samples: list[SampleMetric]
