"""Score existing JSONL assistant SVGs with student_kit.reward.

This is a sanity check for reward.py before model generation is wired up. It
does not replace the official self-eval that compares base vs fine-tuned model.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from reward import score_svg


def get_message(messages: list[dict], role: str) -> str:
    for item in messages:
        if item.get("role") == role:
            return item.get("content", "")
    return ""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("jsonl", type=Path)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    rows = []
    for idx, line in enumerate(args.jsonl.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        record = json.loads(line)
        messages = record["messages"]
        prompt = get_message(messages, "user")
        svg = get_message(messages, "assistant")
        breakdown = score_svg(svg, prompt)
        rows.append(
            {
                "idx": idx,
                "score": breakdown.score,
                "validity": breakdown.validity,
                "structure": breakdown.structure,
                "geometry": breakdown.geometry,
                "palette": breakdown.palette,
                "prompt_alignment": breakdown.prompt_alignment,
                "penalties": breakdown.penalties,
                "notes": breakdown.notes,
            }
        )

    summary = {
        "file": str(args.jsonl),
        "count": len(rows),
        "mean_score": sum(r["score"] for r in rows) / len(rows) if rows else 0.0,
        "min_score": min((r["score"] for r in rows), default=0.0),
        "max_score": max((r["score"] for r in rows), default=0.0),
        "rows": rows,
    }

    text = json.dumps(summary, ensure_ascii=False, indent=2)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
