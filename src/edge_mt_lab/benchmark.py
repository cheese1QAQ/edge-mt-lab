"""Reproducible translation benchmark harness with dry-run CI validation."""

from __future__ import annotations

import json
import re
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

from edge_mt_lab.hardware import detect_host_hardware
from edge_mt_lab.schemas import BenchmarkReport, EvalSample, SampleMetric

PLACEHOLDER_PATTERN = re.compile(r"(`[^`]+`|\{[a-zA-Z0-9_]+\})")


def default_dataset_path() -> Path:
    """Return repository-root datasets/eval_sample.jsonl when run from source."""
    return Path(__file__).resolve().parents[2] / "datasets" / "eval_sample.jsonl"


def load_eval_dataset(path: Path) -> list[EvalSample]:
    """Load and validate JSONL translation evaluation samples."""
    if not path.exists():
        raise FileNotFoundError(f"Evaluation dataset not found: {path}")

    samples: list[EvalSample] = []
    seen_ids: set[str] = set()
    for line_no, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = raw_line.strip()
        if not stripped:
            continue
        payload = json.loads(stripped)
        sample = EvalSample.model_validate(payload)
        if sample.id in seen_ids:
            raise ValueError(f"Duplicate sample id '{sample.id}' at {path}:{line_no}")
        seen_ids.add(sample.id)
        samples.append(sample)

    if not samples:
        raise ValueError(f"Dataset is empty: {path}")
    return samples


def build_translation_prompt(sample: EvalSample) -> tuple[str, str]:
    """Construct standardized system and user prompts for RAG/glossary-guided MT."""
    glossary_block = (
        "\n".join(f"- {src} => {dst}" for src, dst in sample.glossary.items())
        if sample.glossary
        else "(none)"
    )
    system = (
        f"You are a precise technical translator ({sample.source_lang} -> {sample.target_lang}).\n"
        "Rules:\n"
        "1. Strictly use the provided glossary terms.\n"
        "2. Preserve all inline code spans, flags, and `{placeholder}` tokens verbatim.\n"
        "3. Silently fix obvious OCR typos when meaning is clear.\n"
        "4. Output ONLY the translated text.\n"
        f"[GLOSSARY]\n{glossary_block}\n"
        f"[CONTEXT]\n{sample.context_notes or '(none)'}"
    )
    return system, sample.source


def check_placeholders_preserved(source: str, translation: str) -> bool:
    """Verify every inline code span or `{placeholder}` in source appears in translation."""
    expected = PLACEHOLDER_PATTERN.findall(source)
    return all(token in translation for token in expected)


def count_glossary_hits(glossary: dict[str, str], translation: str) -> tuple[int, int]:
    """Count expected vs matched glossary target terms in translation output."""
    expected = len(glossary)
    matched = sum(1 for target_term in glossary.values() if target_term in translation)
    return expected, matched


def _dry_run_translate(sample: EvalSample) -> tuple[str, float, float, int]:
    """Deterministic zero-GPU mock translation for CI pipeline validation."""
    start = time.perf_counter()
    _system, _user = build_translation_prompt(sample)
    output = sample.reference
    elapsed_ms = max(0.2, (time.perf_counter() - start) * 1000.0)
    ttft_ms = round(elapsed_ms * 0.4, 3)
    total_ms = round(elapsed_ms, 3)
    output_tokens = max(1, len(output) // 2)
    return output, ttft_ms, total_ms, output_tokens


def run_benchmark(
    dataset_path: Path,
    *,
    backend: str = "dry_run",
    model_name: str = "dry-run-mock",
    llm_budget_gb: float = 3.0,
    output_json: Path | None = None,
) -> BenchmarkReport:
    """Run benchmark suite and optionally persist a JSON report."""
    if backend != "dry_run":
        raise NotImplementedError(
            f"Backend '{backend}' is not wired yet; run with --dry-run or add an adapter in "
            "experiments/tracks/."
        )

    samples = load_eval_dataset(dataset_path)
    metrics: list[SampleMetric] = []

    for sample in samples:
        translated, ttft_ms, total_ms, out_tokens = _dry_run_translate(sample)
        expected, matched = count_glossary_hits(sample.glossary, translated)
        placeholders_ok = check_placeholders_preserved(sample.source, translated)
        decode_tps = round(out_tokens / max(total_ms / 1000.0, 1e-6), 2)
        metrics.append(
            SampleMetric(
                sample_id=sample.id,
                category=sample.category,
                ocr_corrupted=sample.ocr_corrupted,
                ttft_ms=ttft_ms,
                total_ms=total_ms,
                output_tokens=out_tokens,
                decode_tps=decode_tps,
                glossary_expected=expected,
                glossary_matched=matched,
                placeholders_preserved=placeholders_ok,
                output_preview=translated[:120],
            )
        )

    total_expected = sum(m.glossary_expected for m in metrics)
    total_matched = sum(m.glossary_matched for m in metrics)
    glossary_rate = round(total_matched / total_expected, 4) if total_expected > 0 else 1.0
    placeholder_rate = round(
        sum(1 for m in metrics if m.placeholders_preserved) / len(metrics),
        4,
    )

    host = detect_host_hardware()
    now = datetime.now(timezone.utc)
    report = BenchmarkReport(
        run_id=f"{now.strftime('%Y%m%dT%H%M%SZ')}_{backend}",
        timestamp_utc=now.isoformat(),
        backend=backend,
        model_name=model_name,
        dataset_path=str(dataset_path),
        llm_budget_gb=llm_budget_gb,
        sample_count=len(metrics),
        mean_ttft_ms=round(statistics.mean(m.ttft_ms for m in metrics), 3),
        mean_decode_tps=round(statistics.mean(m.decode_tps for m in metrics), 2),
        glossary_adherence_rate=glossary_rate,
        placeholder_fidelity_rate=placeholder_rate,
        peak_vram_mib=host.gpu_vram_used_mib,
        samples=metrics,
    )

    if output_json is not None:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(
            json.dumps(report.model_dump(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    return report
