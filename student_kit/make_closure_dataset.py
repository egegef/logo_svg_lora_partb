"""Build a short valid-SVG curriculum dataset.

The original Sonnet SVGs are often long. This helper creates compact, valid SVG
targets from each prompt so the small model can learn the hard requirement:
produce one complete, closed SVG document and stop.
"""

from __future__ import annotations

import argparse
import html
import json
import re
from pathlib import Path


HEX_RE = re.compile(r"#[0-9a-fA-F]{6}\b")


def get_message(messages: list[dict], role: str) -> str:
    for item in messages:
        if item.get("role") == role:
            return item.get("content", "")
    return ""


def colors_from(prompt: str) -> list[str]:
    colors = []
    for color in HEX_RE.findall(prompt):
        color = color.upper()
        if color not in colors:
            colors.append(color)
    defaults = ["#1B3A5C", "#5DA88E", "#F2A93B", "#FFFFFF"]
    for color in defaults:
        if color not in colors:
            colors.append(color)
    return colors[:4]


def simple_svg(prompt: str) -> str:
    lower = prompt.lower()
    c = colors_from(prompt)
    bg, accent, warm, light = c
    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 256 256">',
        f'<circle cx="128" cy="128" r="108" fill="{bg}"/>',
        f'<circle cx="128" cy="128" r="92" fill="none" stroke="{light}" stroke-width="4"/>',
    ]

    if any(word in lower for word in ["leaf", "sprout", "plant", "growth", "green"]):
        parts.append(f'<path d="M70 162 C98 122 137 118 184 142 C145 170 104 184 70 162 Z" fill="{accent}"/>')
        parts.append(f'<path d="M128 176 C130 140 140 112 162 88" fill="none" stroke="{warm}" stroke-width="8" stroke-linecap="round"/>')
    elif any(word in lower for word in ["star", "spark", "bright"]):
        parts.append(f'<polygon points="128,54 145,105 199,105 155,136 172,190 128,158 84,190 101,136 57,105 111,105" fill="{warm}"/>')
    elif any(word in lower for word in ["mountain", "peak", "adventure"]):
        parts.append(f'<polygon points="58,176 112,82 146,138 172,104 204,176" fill="{accent}"/>')
        parts.append(f'<path d="M112 82 L130 112 L100 112 Z" fill="{light}" opacity="0.9"/>')
    elif any(word in lower for word in ["wave", "water", "river", "ocean"]):
        parts.append(f'<path d="M54 150 C82 124 106 174 134 148 C160 124 178 152 204 134 L204 180 L54 180 Z" fill="{accent}"/>')
    elif any(word in lower for word in ["shield", "badge", "crest"]):
        parts.append(f'<path d="M128 54 L190 82 L178 160 C166 194 142 212 128 218 C114 212 90 194 78 160 L66 82 Z" fill="{accent}"/>')
    else:
        parts.append(f'<rect x="76" y="82" width="104" height="104" rx="24" fill="{accent}"/>')
        parts.append(f'<circle cx="128" cy="128" r="34" fill="{warm}"/>')

    if any(word in lower for word in ["music", "note", "sound", "song"]):
        parts.append(f'<path d="M142 74 L142 156" stroke="{warm}" stroke-width="8" stroke-linecap="round"/>')
        parts.append(f'<ellipse cx="122" cy="164" rx="24" ry="16" fill="{warm}" transform="rotate(-18 122 164)"/>')
    if any(word in lower for word in ["line", "staff", "stripe", "tick"]):
        parts.append(f'<line x1="72" y1="196" x2="184" y2="196" stroke="{light}" stroke-width="5" stroke-linecap="round"/>')

    parts.append("</svg>")
    return "".join(parts)


def convert(src: Path, dst: Path) -> None:
    rows = []
    for line in src.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        messages = record["messages"]
        system = get_message(messages, "system")
        prompt = get_message(messages, "user")
        rows.append(
            {
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": simple_svg(prompt)},
                ]
            }
        )
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-in", default="./logo-detailed-prompt/train.jsonl")
    parser.add_argument("--valid-in", default="./logo-detailed-prompt/valid.jsonl")
    parser.add_argument("--out-dir", default="./closure_data")
    args = parser.parse_args()
    out = Path(args.out_dir)
    convert(Path(args.train_in), out / "train_closure.jsonl")
    convert(Path(args.valid_in), out / "valid_closure.jsonl")


if __name__ == "__main__":
    main()
