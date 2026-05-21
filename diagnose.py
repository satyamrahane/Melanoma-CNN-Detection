#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MelanomaAI v2 — terminal diagnostic runner.

Thin CLI wrapper around shared backend modules (same stack as app.py).

Usage:
    python diagnose.py image.jpg
    python diagnose.py image.jpg --age 55
    python diagnose.py image.jpg --threshold 0.45
    python diagnose.py image.jpg --no-gradcam
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Windows: enable ANSI colors in console
if sys.platform == "win32":
    try:
        import ctypes

        _handle = ctypes.windll.kernel32.GetStdHandle(-11)
        _mode = ctypes.c_ulong()
        ctypes.windll.kernel32.GetConsoleMode(_handle, ctypes.byref(_mode))
        ctypes.windll.kernel32.SetConsoleMode(_handle, _mode.value | 0x0007)
    except Exception:
        pass

RESET = "\033[0m"
CYAN = "\033[96m"
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
DIM = "\033[2m"
BOLD = "\033[1m"

WIDTH = 48


def _box_chars() -> tuple[str, str]:
    enc = getattr(sys.stdout, "encoding", None) or "utf-8"
    try:
        "\u2550".encode(enc)
        return "\u2550", "\u2500"
    except Exception:
        return "=", "-"


DOUBLE, SINGLE = _box_chars()


def c(text: str, code: str) -> str:
    if not sys.stdout.isatty():
        return text
    return f"{code}{text}{RESET}"


def fmt_pct(value) -> str:
    if value is None:
        return "N/A"
    v = float(value)
    return f"{v * 100:.2f}%" if v <= 1.0 else f"{v:.2f}%"


def line_double() -> str:
    return DOUBLE * WIDTH


def section(title: str) -> None:
    bar = SINGLE * 14
    print()
    print(c(f"{bar} {title} {bar}", CYAN))


def row(label: str, value: str, value_color: str = "") -> None:
    label_w = 14
    val = c(value, value_color) if value_color else value
    print(f"{label:<{label_w}}{val}")


def gradcam_output_path(image_path: Path) -> Path:
    out_dir = ROOT / "outputs" / "gradcam"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"{image_path.stem}_gradcam.jpg"


def open_gradcam(path: Path) -> None:
    p = str(path.resolve())
    if sys.platform == "win32":
        os.startfile(p)
    elif sys.platform == "darwin":
        os.system(f'open "{p}"')
    else:
        os.system(f'xdg-open "{p}"')


def run(args: argparse.Namespace) -> int:
    image_path = Path(args.image).expanduser().resolve()
    if not image_path.is_file():
        print(f"Error: image not found: {image_path}", file=sys.stderr)
        return 1

    from backend.gradcam import generate_gradcam
    from backend.metrics import load_metrics
    from backend.model import predict_image
    from risk_engine import compute_risk_score

    import torch

    threshold = float(args.threshold)
    age = int(args.age)

    t0 = time.perf_counter()
    try:
        result = predict_image(str(image_path), threshold=threshold)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    elapsed = time.perf_counter() - t0

    prob = float(result["probability"])
    ita = float(result["ita"])
    verdict = result["label"]
    risk = compute_risk_score(prob, ita, threshold, age=age)
    abcd = result["abcd"]
    clahe = bool(result.get("clahe_applied", False))
    reliability = result.get("reliability", risk.get("reliability", 0))
    tone = result.get("tone", "Unknown")

    metrics = load_metrics() or {}
    auc = metrics.get("auc_roc", 0.9770)
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"

    verdict_color = RED if verdict == "MALIGNANT" else GREEN

    print()
    print(c(line_double(), CYAN))
    print(c("        MelanomaAI v2 — Clinical Diagnosis".center(WIDTH), BOLD + CYAN))
    print(c(line_double(), CYAN))
    row("Image:", image_path.name)
    row("Model:", f"EfficientNetB3 | AUC {float(auc):.4f}")

    section("PREDICTION")
    row("VERDICT:", verdict, verdict_color)
    row("Probability:", f"{prob * 100:.1f}%")
    row("Threshold:", f"{threshold:.2f}")
    row("Inference:", f"{elapsed:.2f}s | GPU: {gpu}")

    section("RISK SCORE")
    row("Level:", risk.get("level", "MODERATE"))
    row("Score:", f"{risk.get('score', 0):.0f} / 100")
    row("Action:", risk.get("action", ""))
    factors = risk.get("factors", {})
    age_pts = factors.get("age_adjustment", 0)
    if age > 40:
        row("Age factor:", f"+{age_pts:.1f} pts (patient age {age})")
    else:
        row("Age factor:", f"0 pts (applies only for age > 40)")

    section("SKIN TONE")
    row("ITA Angle:", f"{ita:.1f}\u00b0 \u2192 {tone}")
    clahe_txt = "ON (ITA \u2264 28\u00b0)" if clahe else "OFF (ITA > 28\u00b0)"
    row("CLAHE:", clahe_txt)
    rel_pct = float(reliability) * 100 if float(reliability) <= 1 else float(reliability)
    row("Reliability:", f"{rel_pct:.0f}%")

    section("ABCD BIOMARKERS")
    a = abcd.get("asymmetry", {})
    b = abcd.get("border", {})
    col = abcd.get("color", {})
    d = abcd.get("diameter", {})
    row("Asymmetry:", f"{a.get('score', 0):.3f}  [{a.get('label', '')}]")
    row("Border:", f"{b.get('score', 0):.3f}  [{b.get('label', '')}]")
    row("Color:", f"{col.get('score', 0):.3f}  [{col.get('label', '')}]")
    row("Diameter:", f"{d.get('mm', 0):.1f}mm [{d.get('label', '')}]")

    section("EXPLAINABILITY")
    if args.no_gradcam:
        row("Grad-CAM:", "skipped (--no-gradcam)")
    else:
        gpath = gradcam_output_path(image_path)
        try:
            out = generate_gradcam(str(image_path), output_path=str(gpath))
            if out and Path(out).exists():
                row("Grad-CAM:", str(Path(out).relative_to(ROOT)))
                print(c(f"{'':14}Saved and opened", DIM))
                try:
                    open_gradcam(Path(out))
                except Exception as exc:
                    print(f"{'':14}{YELLOW}Could not auto-open: {exc}{RESET}")
            else:
                row("Grad-CAM:", c("generation failed", YELLOW))
        except Exception as exc:
            row("Grad-CAM:", c(str(exc), YELLOW))

    section("SYSTEM METRICS")
    row("Accuracy:", fmt_pct(metrics.get("accuracy")))
    row("AUC-ROC:", f"{float(metrics.get('auc_roc', auc)):.4f}")
    row("Sensitivity:", fmt_pct(metrics.get("sensitivity")))
    row("Specificity:", fmt_pct(metrics.get("specificity")))

    print()
    print(c(line_double(), CYAN))
    print()
    return 0


def build_parser() -> argparse.ArgumentParser:
    from backend.metrics import get_threshold

    default_thr = get_threshold()
    p = argparse.ArgumentParser(
        description="MelanomaAI v2 — CLI clinical diagnosis (shared backend)",
    )
    p.add_argument("image", help="Path to dermoscopy image (jpg/png)")
    p.add_argument("--age", type=int, default=40, help="Patient age (default: 40)")
    p.add_argument(
        "--threshold",
        type=float,
        default=default_thr,
        help=f"Classification threshold (default: {default_thr})",
    )
    p.add_argument(
        "--no-gradcam",
        action="store_true",
        help="Skip Grad-CAM generation and auto-open",
    )
    return p


def main() -> int:
    return run(build_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
