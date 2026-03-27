"""
0. сбилдите образ: cd runtime/triton_trtllm && docker build . -f Dockerfile.server -t soar97/triton-cosyvoice:25.06
1. запустите сервис: docker compose -f docker-compose.cosyvoice3.yml up
2. запустите прогрев и тест: bash benchmark.sh
"""
import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path

try:
    import pandas as pd
except ImportError as exc:
    raise SystemExit("pandas is required to generate report.xlsx") from exc

ROOT = Path(__file__).resolve().parent
LOG_NAME = "rtf-wenetspeech4tts.txt"
OUTPUT_COLUMNS = [
    "Model",
    "Text (sym.)",
    "Conc",
    "Calls",
    "RTF",
    "RPS",
    "TTFB p50",
    "TTFB p90",
    "ITL p50",
    "ITL p90",
    "E2E p50",
    "E2E p90",
    "Throughput (×RT)",
    "Quality ОК?",
    "Additional",
    "Folder",
]

LOG_PATTERNS = {
    "RTF": re.compile(r"^RTF:\s*([0-9.]+)", re.MULTILINE),
    "TTFB p50": re.compile(r"^first_chunk_latency_50_percentile_ms:\s*([0-9.]+)", re.MULTILINE),
    "TTFB p90": re.compile(r"^first_chunk_latency_90_percentile_ms:\s*([0-9.]+)", re.MULTILINE),
    "ITL p50": re.compile(r"^second_chunk_latency_50_percentile_ms:\s*([0-9.]+)", re.MULTILINE),
    "ITL p90": re.compile(r"^second_chunk_latency_90_percentile_ms:\s*([0-9.]+)", re.MULTILINE),
    "E2E p50": re.compile(r"^total_request_latency_50_percentile_ms:\s*([0-9.]+)", re.MULTILINE),
    "E2E p90": re.compile(r"^total_request_latency_90_percentile_ms:\s*([0-9.]+)", re.MULTILINE),
}

PERCENT_RE = re.compile(r"^(?P<key>[A-Za-z0-9_]+):\s*(?P<value>.+)$", re.MULTILINE)
TEXT_RE = re.compile(r"text(?P<value>\d+)")
CONC_RE = re.compile(r"threads(?P<value>\d+)")
THREADS_RE = re.compile(r"threads(?P<value>\d+)")
CALLS_RE = re.compile(r"threads(?P<value>\d+)")
MODEL_HINTS = ["CosyVoice3"]
DEFAULT_TEXT = "100"
PROCESSING_TIME_RE = re.compile(r"^processing time:\s*([0-9.]+) seconds", re.MULTILINE)


def parse_log_metrics(text: str) -> dict[str, str]:
    values = {}
    for column, pattern in LOG_PATTERNS.items():
        match = pattern.search(text)
        if match:
            values[column] = match.group(1)
    return values


def infer_from_folder(folder_name: str) -> dict[str, str]:
    result = defaultdict(str)

    result["Model"] = "CosyVoice3"
    text_match = TEXT_RE.search(folder_name)
    if text_match:
        result["Text (sym.)"] = int(text_match.group("value"))
    else:
        result["Text (sym.)"] = int(DEFAULT_TEXT)

    conc_match = CONC_RE.search(folder_name)
    if conc_match:
        result["Conc"] = conc_match.group("value")

    calls_match = CALLS_RE.search(folder_name)
    if calls_match:
        result["Calls"] = calls_match.group("value")

    threads_match = THREADS_RE.search(folder_name)
    if threads_match:
        result["Additional"] = f"threads={threads_match.group('value')}"

    parts = []
    if result["Text (sym.)"]:
        parts.append(f"text={result['Text (sym.)']}")
    if result["Conc"]:
        parts.append(f"conc={result['Conc']}")
    if result["Calls"]:
        parts.append(f"calls={result['Calls']}")
    if result["Additional"]:
        parts.append(result["Additional"])
    result["Additional"] = "; ".join(parts)
    return result


def build_rows() -> list[dict[str, str]]:
    rows = []
    for child in sorted(ROOT.iterdir()):
        if not child.is_dir():
            continue
        log_path = child / LOG_NAME
        if not log_path.is_file():
            continue
        text = log_path.read_text(encoding="utf-8")

        inferred = infer_from_folder(child.name)
        row = {column: "" for column in OUTPUT_COLUMNS}
        row["Folder"] = child.name
        threads_match = THREADS_RE.search(child.name)
        processing_time_match = PROCESSING_TIME_RE.search(text)
        total_duration_match = re.search(r"^total_duration:\s*([0-9.]+)", text, re.MULTILINE)
        if threads_match and processing_time_match:
            threads = float(threads_match.group("value"))
            processing_time = float(processing_time_match.group(1))
            if processing_time:
                row["RPS"] = threads / processing_time
        if processing_time_match and total_duration_match:
            processing_time = float(processing_time_match.group(1))
            total_duration = float(total_duration_match.group(1))
            if processing_time:
                row["Throughput (×RT)"] = total_duration / processing_time
        row.update(inferred)
        row.update(parse_log_metrics(text))
        rows.append(row)
    rows.sort(key=lambda r: (
        int(r["Text (sym.)"]),
        int(r["Conc"]) if r["Conc"] else -1,
    ))
    return rows


def write_reports(rows: list[dict[str, str]]) -> None:
    csv_path = ROOT / "report_trt.csv"
    xlsx_path = ROOT / "report_trt.xlsx"

    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS, delimiter=";")
        writer.writeheader()
        writer.writerows(rows)
    df = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)

    df.to_excel(xlsx_path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build benchmark report from Triton TRT-LLM logs")
    parser.parse_args()
    rows = build_rows()
    write_reports(rows)
    print(f"Wrote {ROOT / 'report.csv'}")
    print(f"Wrote {ROOT / 'report.xlsx'}")


if __name__ == "__main__":
    main()
