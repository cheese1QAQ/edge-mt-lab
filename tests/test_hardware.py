"""Unit tests for RTX 3060 8GB / 2-4GB LLM VRAM budget modeling."""

from __future__ import annotations

from edge_mt_lab.hardware import default_reference_profiles, estimate_vram_breakdown
from edge_mt_lab.schemas import ModelSpec


def test_vocab_pruning_and_mla_significantly_reduce_vram() -> None:
    baseline = ModelSpec(
        name="Qwen3-4B baseline",
        params_billions=4.0,
        weight_bits=4.0,
        vocab_size=151_936,
        hidden_size=2560,
        num_layers=36,
        num_kv_heads=8,
        head_dim=128,
        attention_type="gqa",
        kv_bits=8.0,
        context_length=4096,
    )
    optimized = baseline.model_copy(
        update={
            "name": "Qwen3-4B pruned + MLA",
            "pruned_vocab_size": 48_000,
            "attention_type": "mla",
            "mla_latent_dim": 512,
            "kv_bits": 4.0,
        }
    )

    est_base = estimate_vram_breakdown(baseline, llm_budget_gb=4.0)
    est_opt = estimate_vram_breakdown(optimized, llm_budget_gb=4.0)

    assert est_opt.vocab_savings_mib > 150.0
    assert est_opt.weight_mib < est_base.weight_mib
    assert est_opt.kv_cache_mib < est_base.kv_cache_mib * 0.25
    assert est_opt.llm_total_mib < est_base.llm_total_mib


def test_default_reference_profiles_fit_within_4gb_llm_and_8gb_total() -> None:
    for spec in default_reference_profiles():
        est = estimate_vram_breakdown(
            spec,
            llm_budget_gb=4.0,
            total_gpu_vram_gb=8.0,
            ocr_placement="cpu_onnx",
        )
        assert est.fits_llm_budget, f"{spec.name} exceeded 4GB LLM budget: {est.llm_total_mib} MiB"
        assert est.fits_total_vram, f"{spec.name} exceeded 8GB total VRAM: {est.system_total_mib}"


def test_gpu_light_vlm_ocr_adds_coexistence_vram() -> None:
    spec = default_reference_profiles()[0]
    cpu_ocr = estimate_vram_breakdown(spec, ocr_placement="cpu_onnx")
    gpu_ocr = estimate_vram_breakdown(spec, ocr_placement="gpu_light_vlm")
    assert gpu_ocr.ocr_vram_mib > 0.0
    assert gpu_ocr.system_total_mib > cpu_ocr.system_total_mib
