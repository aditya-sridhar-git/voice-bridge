"""
Evaluation Pipeline Runner
============================
Runs evaluation metrics over a test set and produces a summary report.

Usage:
    python evaluation/eval_pipeline.py --test-dir tests/fixtures/ --output eval_report.json
"""

import argparse
import json
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)


def run_eval_suite(
    test_dir: str,
    output_report: str,
    device: str = "cpu",
    accent_pair: str = "indian_american",
):
    """
    Run the full pipeline + evaluation on every WAV in test_dir.

    Expected test_dir structure:
        test_dir/
          speaker1.wav
          speaker2.wav
          ...

    Output: JSON report with per-file and aggregate metrics.
    """
    from pipeline.run_pipeline import run_pipeline
    from evaluation.metrics import evaluate

    test_files = sorted(Path(test_dir).glob("*.wav"))
    if not test_files:
        logger.warning(f"No WAV files found in {test_dir}")
        return

    logger.info(f"Found {len(test_files)} test files in {test_dir}")
    results = []

    for audio_path in test_files:
        logger.info(f"\n{'─'*50}")
        logger.info(f"Processing: {audio_path.name}")

        try:
            # Run pipeline
            pr = run_pipeline(
                input_audio=str(audio_path),
                accent_pair=accent_pair,
                device=device,
                verbose=False,
            )

            # Load intermediate data for evaluation
            features_path = Path(pr.run_dir) / "stage2_features.json"
            with open(features_path) as f:
                feat = json.load(f)

            # Evaluate
            eval_result = evaluate(
                source_audio=str(audio_path),
                output_audio=pr.output_audio,
                reference_transcript=pr.transcript,
                source_f0=feat["prosody"]["f0_hz"],
                source_emotion_label=pr.emotion_label,
                frame_shift_ms=feat["prosody"].get("frame_shift_ms", 10),
                device=device,
            )

            results.append({
                "file": audio_path.name,
                "run_id": pr.run_id,
                "emotion_label": pr.emotion_label,
                "rewrites": pr.rewrites_count,
                "pipeline_duration_s": pr.duration_s,
                "metrics": eval_result.to_dict(),
                "passes": eval_result.passes_thresholds(),
            })

        except Exception as e:
            logger.error(f"Failed on {audio_path.name}: {e}")
            results.append({"file": audio_path.name, "error": str(e)})

    # Aggregate
    valid = [r for r in results if "error" not in r]
    agg = {}
    for metric in ["wer", "speaker_cosine", "f0_correlation"]:
        vals = [r["metrics"][metric] for r in valid if r["metrics"].get(metric) is not None]
        if vals:
            agg[f"mean_{metric}"] = round(sum(vals) / len(vals), 4)

    emo_matches = [r["metrics"]["emotion_match"] for r in valid if r["metrics"].get("emotion_match") is not None]
    if emo_matches:
        agg["emotion_accuracy"] = round(sum(emo_matches) / len(emo_matches), 4)

    agg["total_files"] = len(test_files)
    agg["passed_files"] = sum(1 for r in valid if r.get("passes"))

    report = {"aggregate": agg, "per_file": results}

    with open(output_report, "w") as f:
        json.dump(report, f, indent=2)

    logger.info(f"\n{'='*50}")
    logger.info(f"Evaluation complete. Report saved to {output_report}")
    logger.info(f"Aggregate: {json.dumps(agg, indent=2)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run evaluation suite")
    parser.add_argument("--test-dir", required=True, help="Directory of test WAV files")
    parser.add_argument("--output", default="eval_report.json")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--accent-pair", default="indian_american")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    run_eval_suite(args.test_dir, args.output, args.device, args.accent_pair)
