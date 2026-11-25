"""Command-line entry point for Edge-MT-Lab."""

from __future__ import annotations

import argparse
from pathlib import Path

from edge_mt_lab.benchmark import default_dataset_path, run_benchmark
from edge_mt_lab.hardware import (
    default_reference_profiles,
    detect_host_hardware,
    estimate_vram_breakdown,
)
from edge_mt_lab.schemas import OCRPlacement


def _cmd_doctor(args: argparse.Namespace) -> int:
    host = detect_host_hardware()
    ocr_mode: OCRPlacement = args.ocr
    llm_budget_gb: float = args.llm_budget_gb
    total_vram_gb: float = args.total_vram_gb

    print("=== Edge-MT-Lab Hardware & VRAM Budget Doctor ===")
    print(f"Host OS / Arch : {host.os_name} ({host.machine}), CPU cores={host.cpu_cores}")
    if host.gpu_name is not None:
        print(
            f"Detected GPU   : {host.gpu_name} "
            f"(used={host.gpu_vram_used_mib} MiB / total={host.gpu_vram_total_mib} MiB)"
        )
    else:
        print("Detected GPU   : none (running analytical RTX 3060 8GB budget model)")
    print(
        f"Target Envelope: LLM <= {llm_budget_gb:.1f} GB | "
        f"Total GPU <= {total_vram_gb:.1f} GB | OCR={ocr_mode}"
    )
    print()

    header = (
        f"{'Profile':<54} {'Weight':>8} {'Vocab-':>8} {'KV':>7} "
        f"{'LLM Tot':>9} {'Sys Tot':>9} {'Fit':>5}"
    )
    print(header)
    print("-" * len(header))
    for spec in default_reference_profiles():
        est = estimate_vram_breakdown(
            spec,
            llm_budget_gb=llm_budget_gb,
            total_gpu_vram_gb=total_vram_gb,
            ocr_placement=ocr_mode,
        )
        fit_flag = "OK" if (est.fits_llm_budget and est.fits_total_vram) else "OVER"
        print(
            f"{est.model_name:<54} "
            f"{est.weight_mib:>7.0f}M "
            f"{est.vocab_savings_mib:>7.0f}M "
            f"{est.kv_cache_mib:>6.0f}M "
            f"{est.llm_total_mib:>8.0f}M "
            f"{est.system_total_mib:>8.0f}M "
            f"{fit_flag:>5}"
        )
    return 0


def _cmd_bench(args: argparse.Namespace) -> int:
    backend = "dry_run" if args.dry_run else args.backend
    dataset = Path(args.dataset) if args.dataset else default_dataset_path()
    output = Path(args.output) if args.output else None
    report = run_benchmark(
        dataset,
        backend=backend,
        model_name=args.model_name,
        llm_budget_gb=args.llm_budget_gb,
        output_json=output,
    )
    print("=== Edge-MT-Lab Benchmark Summary ===")
    print(f"Run ID               : {report.run_id}")
    print(f"Backend / Model      : {report.backend} / {report.model_name}")
    print(f"Samples Validated    : {report.sample_count}")
    print(f"Glossary Adherence   : {report.glossary_adherence_rate * 100:.1f}%")
    print(f"Placeholder Fidelity : {report.placeholder_fidelity_rate * 100:.1f}%")
    print(f"Mean TTFT / Decode   : {report.mean_ttft_ms:.2f} ms / {report.mean_decode_tps:.1f} tps")
    if output is not None:
        print(f"Report Written       : {output}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="edge-mt-lab",
        description="Low-VRAM (2-4GB) frontier LLM lightweighting & MT research CLI.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_doc = sub.add_parser("doctor", help="Inspect host hardware & 2-4GB VRAM budgets.")
    p_doc.add_argument("--llm-budget-gb", type=float, default=4.0)
    p_doc.add_argument("--total-vram-gb", type=float, default=8.0)
    p_doc.add_argument(
        "--ocr",
        choices=["cpu_onnx", "gpu_light_vlm", "unified_vlm"],
        default="cpu_onnx",
    )
    p_doc.set_defaults(func=_cmd_doctor)

    p_bench = sub.add_parser("bench", help="Validate eval set and run MT benchmark.")
    p_bench.add_argument("--dry-run", action="store_true", help="Run deterministic zero-GPU check.")
    p_bench.add_argument("--backend", type=str, default="dry_run")
    p_bench.add_argument("--model-name", type=str, default="dry-run-mock")
    p_bench.add_argument("--llm-budget-gb", type=float, default=3.0)
    p_bench.add_argument("--dataset", type=str, default=None)
    p_bench.add_argument("--output", type=str, default=None)
    p_bench.set_defaults(func=_cmd_bench)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    func = args.func
    return int(func(args))


if __name__ == "__main__":
    raise SystemExit(main())
