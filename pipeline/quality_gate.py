"""
quality_gate.py
────────────────
QC gate that quarantines low-quality clips instead of silently letting them
into the dataset.

Runs quality_check.py's scoring on a video, and if score < REJECT_THRESHOLD,
moves the real .mp4 out of assets/videos/ into assets/videos/rejected/ —
nothing is ever deleted — with a copy of its quality report saved alongside
so the reason for rejection travels with the clip. The original report also
stays in assets/processed/quality/ as the full historical record of every
clip ever scored. Clips that pass the gate are left exactly where they are.

Once a clip is in assets/videos/rejected/, it's no longer picked up by the
rest of the pipeline — every other module's directory listing of
assets/videos/ is non-recursive, so the rejected/ subfolder is naturally
skipped without needing an explicit exclude list.

Usage:
  Single video : python pipeline/quality_gate.py assets/videos/WashingCup.mp4
  All videos   : python pipeline/quality_gate.py assets/videos/
  Dry run      : python pipeline/quality_gate.py assets/videos/ --dry-run
"""

import sys
import shutil
from pathlib import Path

import quality_check

REJECTED_DIR      = Path("assets/videos/rejected")
REJECT_THRESHOLD  = 60


def gate_video(video_path: Path, dry_run: bool = False) -> dict:
    """Score one video; move it to rejected/ if below threshold. Returns the quality report.

    dry_run=True scores and prints the verdict but never moves any files.
    """
    if video_path.resolve().parent == REJECTED_DIR.resolve():
        print(f"  ⏭️  SKIPPED  — {video_path.name} is already in {REJECTED_DIR}/")
        report_path = quality_check.QUALITY_ROOT / f"{video_path.stem}_quality.json"
        if report_path.exists():
            import json
            with open(report_path) as f:
                return json.load(f)
        return quality_check.analyse_video(video_path)

    report = quality_check.analyse_video(video_path)

    if report["score"] < REJECT_THRESHOLD:
        if dry_run:
            print(f"  🔍 DRY-RUN  — {video_path.name} (score {report['score']}) would be REJECTED")
        else:
            REJECTED_DIR.mkdir(parents=True, exist_ok=True)

            dest_video = REJECTED_DIR / video_path.name
            shutil.move(str(video_path), str(dest_video))

            src_report  = quality_check.QUALITY_ROOT / f"{video_path.stem}_quality.json"
            dest_report = REJECTED_DIR / f"{video_path.stem}_quality.json"
            if src_report.exists():
                shutil.copy(str(src_report), str(dest_report))

            print(f"  🚫 REJECTED — {video_path.name} (score {report['score']}) → {dest_video}")
    else:
        print(f"  ✅ PASSED   — {video_path.name} (score {report['score']})")

    return report


def print_summary(reports: list):
    rejected = sum(1 for r in reports if r["score"] < REJECT_THRESHOLD)
    passed   = len(reports) - rejected
    print(f"\n  {len(reports)} video(s) checked — {passed} passed, {rejected} rejected → {REJECTED_DIR}/\n")


if __name__ == "__main__":
    dry_run  = "--dry-run" in sys.argv
    clean_argv = [a for a in sys.argv if a != "--dry-run"]

    if len(clean_argv) != 2:
        print("Usage:")
        print("  Single video : python pipeline/quality_gate.py assets/videos/WashingCup.mp4")
        print("  All videos   : python pipeline/quality_gate.py assets/videos/")
        print("  Dry run      : python pipeline/quality_gate.py assets/videos/ --dry-run")
        sys.exit(1)

    target  = Path(clean_argv[1])
    reports = []

    if dry_run:
        print(f"\n  DRY-RUN mode — no files will be moved\n")

    if target.is_dir():
        videos = sorted(target.glob("*.mp4"))    # non-recursive — never descends into rejected/
        if not videos:
            print(f"[ERROR] No .mp4 files found in {target}")
            sys.exit(1)
        print(f"\n  Found {len(videos)} video(s) in {target}")
        for v in videos:
            reports.append(gate_video(v, dry_run=dry_run))

    elif target.is_file():
        if target.suffix.lower() != ".mp4":
            print(f"[ERROR] Not an MP4 file: {target}")
            sys.exit(1)
        reports.append(gate_video(target, dry_run=dry_run))

    else:
        print(f"[ERROR] Path not found: {target}")
        sys.exit(1)

    if len(reports) > 1:
        print_summary(reports)
