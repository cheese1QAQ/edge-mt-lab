"""Unit tests for dataset validation, prompt construction, and dry-run benchmark."""

from __future__ import annotations

import json
from pathlib import Path

from edge_mt_lab.benchmark import (
    build_translation_prompt,
    check_placeholders_preserved,
    default_dataset_path,
    load_eval_dataset,
    run_benchmark,
)
from edge_mt_lab.cli import main


def test_default_eval_dataset_loads_and_covers_all_categories() -> None:
    samples = load_eval_dataset(default_dataset_path())
    categories = {s.category for s in samples}
    assert len(samples) >= 6
    assert categories == {"terminology", "ocr_noise", "formatting", "discourse"}
    assert any(s.ocr_corrupted for s in samples)


def test_prompt_and_placeholder_checker() -> None:
    samples = load_eval_dataset(default_dataset_path())
    fmt_sample = next(s for s in samples if s.category == "formatting")
    system, user = build_translation_prompt(fmt_sample)
    assert "[GLOSSARY]" in system
    assert user == fmt_sample.source
    assert check_placeholders_preserved(fmt_sample.source, fmt_sample.reference)
    assert not check_placeholders_preserved(fmt_sample.source, "missing placeholders")


def test_dry_run_benchmark_and_json_export(tmp_path: Path) -> None:
    out_file = tmp_path / "report.json"
    report = run_benchmark(
        default_dataset_path(),
        backend="dry_run",
        output_json=out_file,
    )
    assert report.sample_count >= 6
    assert report.glossary_adherence_rate == 1.0
    assert report.placeholder_fidelity_rate == 1.0
    assert out_file.exists()
    parsed = json.loads(out_file.read_text(encoding="utf-8"))
    assert parsed["run_id"] == report.run_id


def test_cli_doctor_and_bench_dry_run() -> None:
    assert main(["doctor", "--llm-budget-gb", "4.0", "--ocr", "cpu_onnx"]) == 0
    assert main(["bench", "--dry-run"]) == 0
