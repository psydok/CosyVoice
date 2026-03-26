#!/usr/bin/env python3
"""Benchmark for CosyVoice3 gRPC synthesis server."""

import argparse
import csv
from datetime import datetime
import io
import os
import sys
import statistics
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

import grpc
import numpy as np
import torch
import torchaudio
import yaml

try:
    import pandas as pd
except ImportError:
    pd = None

try:
    from openpyxl import Workbook
except ImportError:
    Workbook = None

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cosyvoice_pb2
import cosyvoice_pb2_grpc

DEFAULT_CONFIG = {
    "server": {"host": "localhost", "port": 50000},
    "warmup": {"enabled": True, "calls": 2},
    "repeats": 3,
    "voices": [
        {
            "name": "ref",
            "path": "ref.wav",
            "prompt_text": "В лесу родилась ёлочка, в лесу она росла. Зимой и летом стройная, зелёная была. Метель ей пела песенку, спи ёлочка бай-бай. Мороз снежком укутывал, смотри не замерзай",
        },
    ],
    "texts": {
        "20": "Источники добавлены."[:20],
        "100": "Когда вы добавили источники, нужно проиндексировать их. После этого база знаний сможет отвечать на вопросы по этим источникам."[:100],
        "200": "Когда вы добавили источники, нужно проиндексировать их. После этого база знаний сможет отвечать на вопросы по этим источникам. Для проверки качества и стабильности мы также используем более длинные фразы, чтобы увидеть, как меняются задержки и скорость генерации на коротких, средних и длинных текстах."[:200],
    },
    "modes": [
        {"mode": "zero_shot"},
    ],
    "concurrency": {
        "levels": [1, 2, 4, 8, 16, 32],
        "calls_per_level": 20,
        "rps_limit": 0,
    },
    "skip": {
        "texts": False,
        "modes": False,
        "concurrency": False,
    },
    "max_duration_s": 0,
    "save_audio_dir": "",
}


@dataclass
class VoiceSample:
    name: str
    path: str
    prompt_text: str
    audio_bytes: bytes = field(default=b"", repr=False)

    def load(self):
        p = Path(self.path)
        if not p.exists():
            raise FileNotFoundError(f"Voice sample not found: {p}")
        self.audio_bytes = p.read_bytes()


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def make_stub(host: str, port: int):
    channel = grpc.insecure_channel(
        f"{host}:{port}",
        options=[("grpc.max_receive_message_length", 50 * 1024 * 1024)],
    )
    return cosyvoice_pb2_grpc.CosyVoiceStub(channel)


class RateLimiter:
    """Token-bucket rate limiter for RPS control."""

    def __init__(self, rps: float):
        self.rps = rps
        self.interval = 1.0 / rps if rps > 0 else 0
        self.lock = threading.Lock()
        self.last = 0.0

    def acquire(self):
        if self.interval <= 0:
            return
        with self.lock:
            now = time.monotonic()
            wait = self.last + self.interval - now
            if wait > 0:
                time.sleep(wait)
            self.last = time.monotonic()


def _build_request(text: str, voice: VoiceSample, mode: str = "zero_shot",
                   instruct_text: str = ""):
    request = cosyvoice_pb2.Request()
    zero_shot_request = cosyvoice_pb2.zeroshotRequest()
    zero_shot_request.tts_text = text
    zero_shot_request.prompt_text = voice.prompt_text
    zero_shot_request.prompt_audio = voice.audio_bytes
    request.zero_shot_request.CopyFrom(zero_shot_request)

    return request


def run_one(stub, text: str, voice: VoiceSample, mode: str = "zero_shot",
            instruct_text: str = "") -> dict:
    """Single synthesis call. Returns metrics dict."""
    req = _build_request(text, voice, mode=mode, instruct_text=instruct_text)
    return _run_one_streaming_inference(stub, req, text, voice.name)


def _run_one_streaming_inference(stub, req, text, voice_name) -> dict:
    t0 = time.perf_counter()
    itl_start_time = None 
    ttfb_ms = None
    tts_audio = b""

    itl_ms = []  # perf_counter timestamps for each streamed piece with non-empty tts_audio
    chunk_sizes = []

    try:
        for r in stub.Inference(req):
            now = time.perf_counter()
            if ttfb_ms is None:
                ttfb_ms = (now - t0) * 1000.0
            else:
                itl_ms.append((now - itl_start_time) * 1000.0)
            piece = getattr(r, "tts_audio", b"")
            if piece:
                chunk_sizes.append(len(piece))
                tts_audio += piece
            itl_start_time = time.perf_counter()

    except Exception:
        e2e_ms = (time.perf_counter() - t0) * 1000.0
        return {"error": True, "e2e_ms": e2e_ms, "ttfb_ms": ttfb_ms or e2e_ms}

    e2e_ms = (time.perf_counter() - t0) * 1000.0
    if ttfb_ms is None:
        ttfb_ms = e2e_ms

    if not tts_audio:
        return {"error": True, "e2e_ms": e2e_ms, "ttfb_ms": ttfb_ms}

    tts_speech = torch.from_numpy(np.array(np.frombuffer(tts_audio, dtype=np.int16))).unsqueeze(dim=0)
    wav_buf = io.BytesIO()
    torchaudio.save(wav_buf, tts_speech, 24000, format="wav")

    audio_s = tts_speech.shape[-1] / 24000
    server_ms = e2e_ms

    first_chunk_s = 0.0
    second_chunk_after_first_s = 0.0

    if len(chunk_sizes) >= 1:
        # Approx: chunk duration from PCM bytes assuming int16 mono at 24kHz.
        # duration_s = bytes / (2 bytes/sample) / sr
        first_chunk_s = chunk_sizes[0] / 2.0 / 24000.0

    if len(chunk_sizes) >= 2:
        second_chunk_after_first_s = chunk_sizes[1] / 2.0 / 24000.0

    return {
        "error": False,
        "ttfb_ms": ttfb_ms,
        "e2e_ms": e2e_ms,
        "server_ms": server_ms,
        "audio_s": audio_s,
        "server_rtf": (server_ms / 1000.0) / max(audio_s, 1e-6),
        "e2e_rtf": (e2e_ms / 1000.0) / max(audio_s, 1e-6),
        "text_len": len(text),
        "audio_bytes": len(tts_audio),
        "voice": voice_name,
        "wav_data": wav_buf.getvalue(),
        # New chunk timing metrics
        "chunk_count": len(chunk_sizes),
        "chunk_count_hint": "non-empty tts_audio pieces",
        "itl_ms_p50": percentile(itl_ms, 50),
        "itl_ms_p90": percentile(itl_ms, 90),
        "first_chunk_s": first_chunk_s,
        "second_chunk_after_first_s": second_chunk_after_first_s,
    }


def warmup(stub, voice: VoiceSample, n: int = 2):
    print(f"Warming up ({n} calls)...", flush=True)
    for _ in range(n):
        run_one(stub, "这是一个预热测试。Warmup test sentence.", voice)
    print("Warmup done.\n", flush=True)


def percentile(data, p):
    s = sorted(data)
    k = (len(s) - 1) * p / 100.0
    f = int(k)
    c = f + 1 if f + 1 < len(s) else f
    return s[f] + (k - f) * (s[c] - s[f])


def percentile_or_zero(data, p):
    if not data:
        return 0.0
    return percentile(data, p)


def summarize_results(results: list[dict]) -> dict:
    ok = [r for r in results if not r.get("error")]
    if not ok:
        return {"ok": 0, "errors": len(results)}
    n = len(ok)
    server_ms = [r["server_ms"] for r in ok]
    e2e_ms = [r["e2e_ms"] for r in ok]
    ttfb_ms = [r["ttfb_ms"] for r in ok]
    itl_ms_p50 = [r["itl_ms_p50"] for r in ok]
    itl_ms_p90 = [r["itl_ms_p90"] for r in ok]
    audio_s = [r["audio_s"] for r in ok]
    rtf = [r["server_rtf"] for r in ok]

    chunk_counts = [r.get("chunk_count", 0) for r in ok]
    audio_durations_s = [r.get("audio_s", 0.0) for r in ok]
    first_chunk_s = [r.get("first_chunk_s", 0.0) for r in ok]
    second_chunk_after_first_s = [r.get("second_chunk_after_first_s", 0.0) for r in ok]

    def _p(values, p):
        values = [v for v in values if v is not None]
        if not values:
            return 0.0
        return percentile(values, p)

    return {
        "ok": n,
        "errors": len(results) - n,
        "audio_s_mean": statistics.mean(audio_s),
        "server_ms_mean": statistics.mean(server_ms),
        "server_ms_p50": percentile(server_ms, 50),
        "server_ms_p90": percentile(server_ms, 90),
        "e2e_ms_mean": statistics.mean(e2e_ms),
        "e2e_ms_p50": percentile(e2e_ms, 50),
        "e2e_ms_p90": percentile(e2e_ms, 90),
        "ttfb_ms_mean": statistics.mean(ttfb_ms),
        "ttfb_ms_p50": percentile(ttfb_ms, 50),
        "ttfb_ms_p90": percentile(ttfb_ms, 90),
        "rtf_mean": statistics.mean(rtf),
        "rtf_min": min(rtf),
        "rtf_max": max(rtf),
        # New requested metrics
        "chunk_count_mean": statistics.mean(chunk_counts) if chunk_counts else 0.0,
        "chunk_count_p50": _p(chunk_counts, 50),
        "chunk_count_p90": _p(chunk_counts, 90),
        "chunk_count_p95": _p(chunk_counts, 95),
        "audio_dur_s_mean": statistics.mean(audio_durations_s) if audio_durations_s else 0.0,
        "audio_dur_s_p50": _p(audio_durations_s, 50),
        "audio_dur_s_p90": _p(audio_durations_s, 90),
        "audio_dur_s_p95": _p(audio_durations_s, 95),
        "first_chunk_s_mean": statistics.mean(first_chunk_s) if first_chunk_s else 0.0,
        "second_chunk_after_first_s_mean": statistics.mean(second_chunk_after_first_s) if second_chunk_after_first_s else 0.0,

        "itl_ms_p50": percentile(itl_ms_p50, 50),
        "itl_ms_p90": percentile(itl_ms_p90, 90),
    }


# ---------------------------------------------------------------------------
# Benchmark suites
# ---------------------------------------------------------------------------

def _check_deadline(deadline: float) -> bool:
    """Return True if deadline exceeded (0 = no deadline)."""
    return deadline > 0 and time.monotonic() > deadline


def bench_texts(stub, voices: list[VoiceSample], texts: list,
                repeats: int, save_dir: str = "",
                deadline: float = 0) -> list[dict]:
    """Benchmark each text snippet with each voice."""
    rows = []
    for voice in voices:
        if _check_deadline(deadline):
            print("  (time limit reached, stopping)", flush=True)
            break
        print(f"\n  Voice: {voice.name}", flush=True)
        for label, text in texts:
            if _check_deadline(deadline):
                break
            results = []
            for rep in range(repeats):
                r = run_one(stub, text, voice)
                results.append(r)
                if save_dir and not r.get("error") and r.get("wav_data"):
                    p = Path(save_dir) / f"{voice.name}_{label}_{rep}.wav"
                    p.write_bytes(r["wav_data"])
            s = summarize_results(results)
            if s["ok"] == 0:
                print(f"    {label:15s}  FAILED", flush=True)
                continue
            row = {"voice": voice.name, "label": label,
                   "text_len": len(text), **s}
            print(f"    {label:15s}  chars={len(text):3d}  "
                  f"audio={s['audio_s_mean']:.2f}s  "
                  f"server={s['server_ms_mean']:.0f}ms  "
                  f"e2e_p50={s['e2e_ms_p50']:.0f}ms  "
                  f"e2e_p90={s['e2e_ms_p90']:.0f}ms  "
                  f"RTF={s['rtf_mean']:.3f}", flush=True)
            rows.append(row)
    return rows


def bench_modes(stub, voice: VoiceSample, text: str,
                modes_cfg: list[dict], repeats: int,
                deadline: float = 0) -> list[dict]:
    """Compare inference modes."""
    rows = []
    for mcfg in modes_cfg:
        if _check_deadline(deadline):
            print("  (time limit reached, stopping)", flush=True)
            break
        mode = mcfg["mode"]
        it = mcfg.get("instruct_text", "")
        results = []
        for _ in range(repeats):
            r = run_one(stub, text, voice, mode=mode, instruct_text=it)
            results.append(r)
        s = summarize_results(results)
        if s["ok"] == 0:
            print(f"  {mode:16s}  FAILED", flush=True)
            continue
        row = {"mode": mode, **s}
        print(f"  {mode:16s}  audio={s['audio_s_mean']:.2f}s  "
              f"server={s['server_ms_mean']:.0f}ms  "
              f"TTFB_p50={s['ttfb_ms_p50']:.0f}ms  "
              f"RTF={s['rtf_mean']:.3f}", flush=True)
        rows.append(row)
    return rows


def bench_concurrency(host: str, port: int, voice: VoiceSample, text: str,
                      levels: list[int], rps_limit: float,
                      calls_per_level: int,
                      text_label: str = "",
                      deadline: float = 0) -> list[dict]:
    """Benchmark concurrent requests with optional RPS limiting."""
    rows = []
    limiter = RateLimiter(rps_limit) if rps_limit > 0 else None

    for conc in levels:
        if _check_deadline(deadline):
            print("  (time limit reached, stopping)", flush=True)
            break
        print(f"\n  text={text_label or text[:24]!r}  concurrency={conc}" +
              (f"  rps_limit={rps_limit}" if limiter else "") + ":",
              flush=True)

        def _call(_i):
            if limiter:
                limiter.acquire()
            s = make_stub(host, port)
            return run_one(s, text, voice)

        results = []
        t_batch = time.perf_counter()
        with ThreadPoolExecutor(max_workers=conc) as pool:
            futs = [pool.submit(_call, i) for i in range(calls_per_level)]
            for fut in as_completed(futs):
                results.append(fut.result())
        duration_s = time.perf_counter() - t_batch

        s = summarize_results(results)
        if s["ok"] == 0:
            print("    all calls failed", flush=True)
            continue

        ok_results = [r for r in results if not r.get("error")]
        total_audio = sum(r["audio_s"] for r in ok_results)
        throughput = total_audio / duration_s
        actual_rps = s["ok"] / duration_s

        row = {
            "text_label": text_label,
            "text_len": len(text),
            "concurrency": conc,
            "total_calls": calls_per_level,
            "duration_s": duration_s,
            "actual_rps": actual_rps,
            "throughput_x_rt": throughput,
            **s,
        }
        print(f"    text={text_label or len(text)}  ok={s['ok']}/{calls_per_level}  "
              f"duration={duration_s:.1f}s  "
              f"rps={actual_rps:.1f}  "
              f"e2e_mean={s['e2e_ms_mean']:.0f}ms  "
              f"e2e_p95={s['e2e_ms_p90']:.0f}ms  "
              f"TTFB_p50={s['ttfb_ms_p50']:.0f}ms  "
              f"RTF={s['rtf_mean']:.3f}  "
              f"throughput={throughput:.2f}x RT", flush=True)
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def print_summary(text_rows, mode_rows, conc_rows):
    W = 105
    print("\n" + "=" * W)
    print("BENCHMARK SUMMARY")
    print("=" * W)

    if text_rows:
        print("\n--- Text Length / Language / Voice ---")
        print(f"{'Voice':>12s} {'Label':>15s} {'Chars':>5s} {'Audio':>7s} "
              f"{'Srv_avg':>8s} {'E2E_avg':>8s} {'E2E_p90':>8s} "
              f"{'TTFB_avg':>9s} {'TTFB_p95':>9s} {'RTF_avg':>8s} {'RTF_max':>8s}")
        for r in text_rows:
            print(f"{r['voice']:>12s} {r['label']:>15s} {r['text_len']:>5d} "
                  f"{r['audio_s_mean']:>6.2f}s "
                  f"{r['server_ms_mean']:>6.0f}ms "
                  f"{r['e2e_ms_mean']:>6.0f}ms "
                  f"{r['e2e_ms_p90']:>6.0f}ms "
                  f"{r['ttfb_ms_mean']:>7.0f}ms "
                  f"{r['ttfb_ms_p90']:>7.0f}ms "
                  f"{r['rtf_mean']:>8.3f} "
                  f"{r['rtf_max']:>8.3f}")

    if mode_rows:
        print("\n--- Mode Comparison ---")
        print(f"{'Mode':>16s} {'Audio':>7s} {'Srv_avg':>8s} "
              f"{'E2E_avg':>8s} {'E2E_p90':>8s} "
              f"{'TTFB_avg':>9s} {'TTFB_p95':>9s} {'RTF_avg':>8s}")
        for r in mode_rows:
            print(f"{r['mode']:>16s} {r['audio_s_mean']:>6.2f}s "
                  f"{r['server_ms_mean']:>6.0f}ms "
                  f"{r['e2e_ms_mean']:>6.0f}ms "
                  f"{r['e2e_ms_p90']:>6.0f}ms "
                  f"{r['ttfb_ms_mean']:>7.0f}ms "
                  f"{r['ttfb_ms_p90']:>7.0f}ms "
                  f"{r['rtf_mean']:>8.3f}")

    if conc_rows:
        print("\n--- Concurrency / Throughput ---")
        print(f"{'Text':>8s} {'Conc':>5s} {'Calls':>6s} {'RPS':>6s} {'Duration':>7s} "
              f"{'E2E_avg':>8s} {'E2E_p90':>8s} "
              f"{'TTFB_avg':>9s} {'TTFB_p90':>9s} "
              f"{'RTF_avg':>8s} {'Throughput':>12s}")
        for r in conc_rows:
            print(f"{str(r.get('text_label', ''))[:8]:>8s} {r['concurrency']:>5d} {r['ok']:>6d} "
                  f"{r['actual_rps']:>5.1f} "
                  f"{r['duration_s']:>6.1f}s "
                  f"{r['e2e_ms_mean']:>6.0f}ms "
                  f"{r['e2e_ms_p90']:>6.0f}ms "
                  f"{r['ttfb_ms_mean']:>7.0f}ms "
                  f"{r['ttfb_ms_p90']:>7.0f}ms "
                  f"{r['rtf_mean']:>8.3f} "
                  f"{r['throughput_x_rt']:>10.2f}x RT")

    print("\n" + "=" * W)


def _stretch_text(text: str, target_len: int) -> str:
    if target_len <= 0:
        return text
    if len(text) >= target_len:
        return text[:target_len].rstrip()
    filler = " Это помогает проверить поведение модели на разных длинах входного текста."
    out = text
    while len(out) < target_len:
        out = f"{out}{filler}"
    return out[:target_len].rstrip()


def _format_num(value, digits=0):
    if value in (None, ""):
        return ""
    if isinstance(value, (int, float)) and value == 0:
        return "0" if digits == 0 else f"{value:.{digits}f}"
    if digits == 0:
        return f"{value:.0f}"
    return f"{value:.{digits}f}"



def _safe_row(r: dict, key: str, default=""):
    return r.get(key, default)


def build_export_rows(text_rows, mode_rows, conc_rows):
    header = [
        "Model", "Text (sym.)", "Conc", "Calls", "RTF", "RPS",
        "TTFB p50", "TTFB p90",
        "ITL p50", "ITL p90",
        "E2E p50", "E2E p90",
        "Throughput (×RT)",
        "Quality ОК?", "Additional",
    ]

    def make_row(model, text="", conc="", calls="", rtf="", rps="", ttfb_p50="", ttfb_p90="", itl_p50="", itl_p90="", e2e_p50="", e2e_p90="", throughput="", quality="", additional=""):
        return [
            model,
            text,
            conc,
            calls,
            rtf,
            rps,
            ttfb_p50,
            ttfb_p90,
            itl_p50,
            itl_p90,
            e2e_p50,
            e2e_p90,
            throughput,
            quality,
            additional,
        ]

    rows = []
    rows.append(["TTS Benchmark Matrix"] + [""] * (len(header) - 1))
    rows.append(header)
    rows.append(["CosyVoice3 (0.5B)"] + [""] * (len(header) - 1))

    for r in text_rows:
        rows.append(make_row(
            "CosyVoice3",
            _safe_row(r, "text_len"),
            _safe_row(r, "concurrency", ""),
            _safe_row(r, "ok", ""),
            _format_num(r["rtf_mean"], 3),
            "",
            _format_num(r["ttfb_ms_p50"]),
            _format_num(r["ttfb_ms_p90"]),
            _format_num(r["itl_ms_p50"] if "itl_ms_p50" in r else ""),
            _format_num(r["itl_ms_p90"] if "itl_ms_p90" in r else ""),
            _format_num(r["e2e_ms_p50"]),
            _format_num(r["e2e_ms_p90"]),
            "",
            "",
            f"voice={r.get('voice', '')}; label={r.get('label', '')}; chunks_mean={_safe_row(r,'chunk_count_mean','')}; audio_dur_p50={_safe_row(r,'audio_dur_s_p50','')}; first_chunk={_safe_row(r,'first_chunk_s_mean','')}; second_after_first={_safe_row(r,'second_chunk_after_first_s_mean','')}",
        ))

    for r in mode_rows:
        rows.append(make_row(
            _safe_row(r, "mode", "CosyVoice3"),
            "",
            "",
            _safe_row(r, "ok", ""),
            _format_num(r["rtf_mean"], 3),
            "",
            _format_num(r["ttfb_ms_p50"]),
            _format_num(r["ttfb_ms_p90"]),
            _format_num(r["itl_ms_p50"] if "itl_ms_p50" in r else ""),
            _format_num(r["itl_ms_p90"] if "itl_ms_p90" in r else ""),
            _format_num(r["e2e_ms_p50"]),
            _format_num(r["e2e_ms_p90"]),
            "",
            "",
            "mode comparison",
        ))

    for r in conc_rows:
        rows.append(make_row(
            "CosyVoice3",
            _safe_row(r, "text_len", ""),
            _safe_row(r, "concurrency", ""),
            _safe_row(r, "ok", ""),
            _format_num(r["rtf_mean"], 3),
            _format_num(r["actual_rps"], 2),
            _format_num(r["ttfb_ms_p50"]),
            _format_num(r["ttfb_ms_p90"]),
            _format_num(r["itl_ms_p50"] if "itl_ms_p50" in r else ""),
            _format_num(r["itl_ms_p90"] if "itl_ms_p90" in r else ""),
            _format_num(r["e2e_ms_p50"]),
            _format_num(r["e2e_ms_p90"]),
            _format_num(r["throughput_x_rt"], 3),
            "",
            f"text={r.get('text_label', '')}; calls={r.get('ok', '')}/{r.get('total_calls', '')}; conc={r.get('concurrency', '')}",
        ))

    return rows


def _write_excel(path, rows):
    if pd is not None:
        df = pd.DataFrame(rows[1:], columns=rows[0])
        df.to_excel(path, index=False)
        return
    if Workbook is None:
        raise ImportError("Excel export requires pandas or openpyxl")

    wb = Workbook()
    ws = wb.active
    for row in rows:
        ws.append(row)
    wb.save(path)


def save_csv(path, text_rows, mode_rows, conc_rows):
    rows = build_export_rows(text_rows, mode_rows, conc_rows)
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f, delimiter=";")
        w.writerows(rows)
    print(f"\nResults saved to {path}")


def save_excel(path, text_rows, mode_rows, conc_rows):
    rows = build_export_rows(text_rows, mode_rows, conc_rows)
    _write_excel(path, rows)
    print(f"Results saved to {path}")


def save_report(path, text_rows, mode_rows, conc_rows, *,
                host, port, repeats, voices):
    """Write a markdown benchmark report."""
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        f"# CosyVoice3 Benchmark Report",
        f"",
        f"**Date:** {ts}  ",
        f"**Server:** `{host}:{port}`  ",
        f"**Repeats per test:** {repeats}  ",
        f"**Voices:** {', '.join(voices)}  ",
        "",
    ]

    if text_rows:
        lines += [
            "## Text Length / Language / Voice",
            "",
            "| Voice | Label | Chars | Audio (s) | Server Avg (ms) | E2E Avg (ms) | E2E p50 (ms) | E2E p90 (ms)  | TTFB Avg (ms) | TTFB p50 (ms) | TTFB p90 (ms) | RTF Avg | RTF Min | RTF Max |",
            "|-------|-------|------:|----------:|----------------:|-------------:|-------------:|-------------:|--------------:|--------------:|--------------:|--------:|--------:|--------:|",
        ]
        for r in text_rows:
            lines.append(
                f"| {r['voice']} | {r['label']} | {r['text_len']} "
                f"| {r['audio_s_mean']:.2f} "
                f"| {r['server_ms_mean']:.0f} "
                f"| {r['e2e_ms_mean']:.0f} "
                f"| {r['e2e_ms_p50']:.0f} "
                f"| {r['e2e_ms_p90']:.0f} "
                f"| {r['ttfb_ms_mean']:.0f} "
                f"| {r['ttfb_ms_p50']:.0f} "
                f"| {r['ttfb_ms_p90']:.0f} "
                f"| {r['rtf_mean']:.3f} "
                f"| {r['rtf_min']:.3f} "
                f"| {r['rtf_max']:.3f} |"
            )
        lines.append("")

    if mode_rows:
        lines += [
            "## Mode Comparison",
            "",
            "| Mode | Audio (s) | Server Avg (ms) | E2E Avg (ms) | E2E p95 (ms)  | TTFB Avg (ms) | TTFB p50 (ms) | TTFB p90 (ms) | RTF Avg | RTF Min | RTF Max |",
            "|------|----------:|----------------:|-------------:|-------------:|--------------:|--------------:|--------------:|--------:|--------:|--------:|",
        ]
        for r in mode_rows:
            lines.append(
                f"| {r['mode']} "
                f"| {r['audio_s_mean']:.2f} "
                f"| {r['server_ms_mean']:.0f} "
                f"| {r['e2e_ms_mean']:.0f} "
                f"| {r['e2e_ms_p90']:.0f} "
                f"| {r['ttfb_ms_mean']:.0f} "
                f"| {r['ttfb_ms_p50']:.0f} "
                f"| {r['ttfb_ms_p90']:.0f} "
                f"| {r['rtf_mean']:.3f} "
                f"| {r['rtf_min']:.3f} "
                f"| {r['rtf_max']:.3f} |"
            )
        lines.append("")

    if conc_rows:
        lines += [
            "## Concurrency / Throughput",
            "",
            "| Concurrency | Calls | RPS | Duration (s) | E2E Avg (ms) | E2E p50 (ms) | E2E p95 (ms) |  TTFB Avg (ms) | TTFB p90 (ms) | RTF Avg | Throughput (x RT) |",
            "|------------:|------:|----:|----------:|-------------:|-------------:|-------------:|--------------:|--------------:|--------:|------------------:|",
        ]
        for r in conc_rows:
            lines.append(
                f"| {r['concurrency']} "
                f"| {r['ok']} "
                f"| {r['actual_rps']:.1f} "
                f"| {r['duration_s']:.1f} "
                f"| {r['e2e_ms_mean']:.0f} "
                f"| {r['e2e_ms_p50']:.0f} "
                f"| {r['e2e_ms_p90']:.0f} "
                f"| {r['ttfb_ms_mean']:.0f} "
                f"| {r['ttfb_ms_p90']:.0f} "
                f"| {r['rtf_mean']:.3f} "
                f"| {r['throughput_x_rt']:.2f} |"
            )
        lines.append("")

    with open(path, "w") as f:
        f.write("\n".join(lines))
    print(f"\nReport saved to {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    now = datetime.now().strftime("%m_%d_%Y__%H_%M_%S")
    parser = argparse.ArgumentParser(description="CosyVoice3 gRPC Benchmark")
    parser.add_argument("--config", type=str, default="",
                        help="YAML config file (see benchmark_config.yaml)")
    parser.add_argument("--csv", type=str, default=f"{now}.csv",
                        help="Save results to CSV file")
    parser.add_argument("--report", type=str, default="",
                        help="Save markdown report (e.g. report.md)")
    parser.add_argument("--host", type=str, default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--repeats", type=int, default=None)
    parser.add_argument("--max-duration", type=int, default=None,
                        help="Max benchmark duration in seconds (0=unlimited)")
    parser.add_argument("--save-audio", type=str, default=f"audios_{now}",
                        help="Directory to save synthesized wav files")
    parser.add_argument("--stream", action="store_true",
                        help="Kept for compatibility with older benchmark runs")
    args = parser.parse_args()

    # Load config
    if args.config:
        cfg = load_config(args.config)
    else:
        cfg = DEFAULT_CONFIG

    # CLI overrides
    srv = cfg.get("server", {})
    host = args.host or srv.get("host", "localhost")
    port = args.port or srv.get("port", 50051)
    repeats = args.repeats or cfg.get("repeats", 3)
    max_dur = args.max_duration if args.max_duration is not None else cfg.get("max_duration_s", 0)
    save_dir = args.save_audio if args.save_audio is not None else cfg.get("save_audio_dir", "")

    # Create save directory if needed
    if save_dir:
        Path(save_dir).mkdir(parents=True, exist_ok=True)

    # Load voices
    base_dir = Path(__file__).resolve().parent.parent.parent.parent
    voices = []
    for vc in cfg.get("voices", []):
        p = vc["path"]
        if not Path(p).is_absolute():
            p = str(base_dir / p)
        v = VoiceSample(name=vc["name"], path=p, prompt_text=vc.get("prompt_text", ""))
        v.load()
        voices.append(v)

    if not voices:
        print("ERROR: No voice samples configured.")
        sys.exit(1)

    # Normalize texts: accept both dict {"label": "text"} and list [{"label":..,"text":..}]
    raw_texts = cfg.get("texts", {})
    if isinstance(raw_texts, list):
        texts = [(t["label"], t["text"]) for t in raw_texts]
    elif isinstance(raw_texts, dict):
        texts = list(raw_texts.items())
    else:
        texts = []
    if texts:
        base_text = texts[0][1]
        labels = {label for label, _ in texts}
        for target in (20, 100, 200):
            if str(target) not in labels:
                texts.append((str(target), _stretch_text(base_text, target)))
        texts = [(label, text) for label, text in texts if label in {"20", "100", "200"}]
        texts.sort(key=lambda item: int(item[0]) if item[0].isdigit() else 10**9)
    modes_cfg = cfg.get("modes", [])
    conc_cfg = cfg.get("concurrency", {})
    skip = cfg.get("skip", {})

    stub = make_stub(host, port)

    print("CosyVoice3 gRPC Benchmark")
    print(f"Server: {host}:{port}")
    print(f"Repeats: {repeats}")
    print(f"Voices: {[v.name for v in voices]}")
    print(f"Texts: {len(texts)} sentences")
    conc_levels = conc_cfg.get("levels", [1])
    rps_limit = conc_cfg.get("rps_limit", 0)
    calls_per_level = conc_cfg.get("calls_per_level", 10)
    print(f"Concurrency: {conc_levels}  RPS limit: {rps_limit or 'none'}")
    print("Streaming: CosyVoice Inference RPC")
    print(f"Max duration: {max_dur}s" if max_dur else "Max duration: unlimited")
    if save_dir:
        print(f"Saving audio to: {save_dir}")
    print()

    # Compute deadline
    deadline = (time.monotonic() + max_dur) if max_dur > 0 else 0

    # Warmup
    wu = cfg.get("warmup", {})
    if wu.get("enabled", True):
        warmup(stub, voices[0], n=wu.get("calls", 2))

    text_rows, mode_rows, conc_rows = [], [], []

    if not skip.get("texts"):
        print("=== Text Length / Language / Voice ===", flush=True)
        text_rows = bench_texts(stub, voices, texts, repeats=repeats,
                                save_dir=save_dir, deadline=deadline)
        text_rows = [dict(r, text_len=int(r.get("text_len", 0))) for r in text_rows]
        for r in text_rows:
            r["text_len"] = int(r.get("text_len", 0))
            r["text_label"] = str(r.get("text_len", r.get("label", "")))


    if not skip.get("modes") and modes_cfg and not _check_deadline(deadline):
        print("\n=== Mode Comparison ===", flush=True)
        mode_text = texts[1][1] if len(texts) > 1 else texts[0][1]
        mode_rows = bench_modes(stub, voices[0], mode_text, modes_cfg,
                                repeats=repeats, deadline=deadline)

    if not skip.get("concurrency") and not _check_deadline(deadline) and texts:
        print("\n=== Concurrency / Throughput ===", flush=True)
        for label, conc_text in texts:
            if _check_deadline(deadline):
                break
            conc_rows.extend(bench_concurrency(
                host, port, voices[0], conc_text,
                levels=conc_levels, rps_limit=rps_limit,
                calls_per_level=calls_per_level, text_label=label,
                deadline=deadline))


    print_summary(text_rows, mode_rows, conc_rows)

    if args.csv:
        save_csv(args.csv, text_rows, mode_rows, conc_rows)
        save_excel(str(Path(args.csv).with_suffix(".xlsx")), text_rows, mode_rows, conc_rows)

    if args.report:
        save_report(args.report, text_rows, mode_rows, conc_rows,
                    host=host, port=port, repeats=repeats,
                    voices=[v.name for v in voices])


if __name__ == "__main__":
    main()
