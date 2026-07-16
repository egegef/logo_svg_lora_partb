"""Programmatic reward for detailed-prompt to SVG-logo generation.

The score is intentionally a training proxy, not a perfect visual judge. It
rewards outputs that are valid, bounded, logo-like SVGs and lightly checks
whether obvious prompt concepts are represented by colors or element choices.
"""

from __future__ import annotations

import math
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Iterable


ALLOWED_TAGS = {
    "svg",
    "defs",
    "g",
    "path",
    "circle",
    "ellipse",
    "rect",
    "polygon",
    "line",
    "linearGradient",
    "radialGradient",
    "stop",
    "clipPath",
    "filter",
    "feGaussianBlur",
}

DISALLOWED_TAGS = {"script", "image", "foreignObject", "iframe", "audio", "video"}
SHAPE_TAGS = {"path", "circle", "ellipse", "rect", "polygon", "line"}
COLOR_RE = re.compile(r"#[0-9a-fA-F]{3}(?:[0-9a-fA-F]{3})?\b|rgb\([^)]+\)|\b(?:red|blue|green|white|black|yellow|orange|purple|pink|gray|grey|navy|teal|gold|brown)\b")
NUMBER_RE = re.compile(r"[-+]?(?:\d*\.\d+|\d+)")
HEX_RE = re.compile(r"#[0-9a-fA-F]{6}\b")


@dataclass
class RewardBreakdown:
    score: float
    validity: float
    structure: float
    geometry: float
    palette: float
    prompt_alignment: float
    penalties: float
    notes: list[str]


def local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def extract_svg(text: str) -> str:
    text = (text or "").strip()
    start = text.lower().find("<svg")
    end = text.lower().rfind("</svg>")
    if start == -1 or end == -1:
        return text
    return text[start : end + len("</svg>")]


def safe_float(value: str | None) -> float | None:
    if value is None:
        return None
    match = NUMBER_RE.search(str(value))
    if not match:
        return None
    try:
        return float(match.group(0))
    except ValueError:
        return None


def iter_numbers(value: str | None) -> Iterable[float]:
    if not value:
        return []
    out = []
    for match in NUMBER_RE.finditer(str(value)):
        try:
            out.append(float(match.group(0)))
        except ValueError:
            pass
    return out


def score_svg(svg_text: str, prompt: str = "") -> RewardBreakdown:
    notes: list[str] = []
    raw = svg_text or ""
    svg = extract_svg(raw)

    validity = 0.0
    structure = 0.0
    geometry = 0.0
    palette = 0.0
    prompt_alignment = 0.0
    penalties = 0.0

    if not svg.strip():
        return RewardBreakdown(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, ["empty output"])

    if raw.strip() != svg.strip():
        penalties += 0.06
        notes.append("extra text outside svg")

    try:
        root = ET.fromstring(svg)
    except ET.ParseError:
        lower = svg.lower()
        shape_mentions = sum(len(re.findall(fr"<{tag}\b", lower)) for tag in SHAPE_TAGS)
        colors = COLOR_RE.findall(svg)
        partial = 0.0
        partial += 0.06 if "<svg" in lower else 0.0
        partial += 0.04 if lower.lstrip().startswith("<svg") else 0.0
        partial += 0.06 if "http://www.w3.org/2000/svg" in lower else 0.0
        partial += 0.05 if 'viewbox="0 0 256 256"' in lower else 0.0
        partial += 0.08 * clamp(shape_mentions / 6.0)
        partial += 0.04 if colors else 0.0
        partial += 0.04 if "</svg>" in lower else 0.0
        partial -= 0.08 if "```" in lower else 0.0
        partial -= 0.08 if "script" in lower or "foreignobject" in lower else 0.0
        partial = clamp(partial, 0.02, 0.35)
        return RewardBreakdown(partial, 0.0, 0.0, 0.0, 0.0, 0.0, 0.75, ["xml parse error", "partial malformed-svg credit"])

    if local_name(root.tag) != "svg":
        return RewardBreakdown(0.03, 0.0, 0.0, 0.0, 0.0, 0.0, 0.9, ["root is not svg"])

    tags = [local_name(el.tag) for el in root.iter()]
    attrs = " ".join(f"{k}={v}" for el in root.iter() for k, v in el.attrib.items())
    text_lower = svg.lower()
    prompt_lower = (prompt or "").lower()

    validity_parts = [
        1.0,
        1.0 if "xmlns" in svg[:160].lower() else 0.0,
        1.0 if root.attrib.get("viewBox") == "0 0 256 256" else 0.4 if "viewBox" in root.attrib else 0.0,
        1.0 if not (set(tags) & DISALLOWED_TAGS) else 0.0,
        1.0 if all(tag in ALLOWED_TAGS for tag in tags) else 0.4,
    ]
    validity = sum(validity_parts) / len(validity_parts)
    if validity < 1.0:
        notes.append("validity issues")

    shape_count = sum(1 for tag in tags if tag in SHAPE_TAGS)
    path_count = tags.count("path")
    group_count = tags.count("g")
    has_defs_if_needed = ("url(#" not in svg) or ("defs" in tags)

    structure_parts = [
        clamp(shape_count / 6.0),
        1.0 if 3 <= shape_count <= 45 else 0.55 if shape_count > 0 else 0.0,
        1.0 if path_count <= 24 else 0.45,
        1.0 if group_count <= 12 else 0.65,
        1.0 if has_defs_if_needed else 0.0,
        1.0 if 250 <= len(svg) <= 9000 else 0.6 if 120 <= len(svg) <= 14000 else 0.2,
    ]
    structure = sum(structure_parts) / len(structure_parts)
    if shape_count == 0:
        notes.append("no vector shapes")

    numbers = []
    for el in root.iter():
        for key, value in el.attrib.items():
            if key in {"d", "points", "x", "y", "x1", "y1", "x2", "y2", "cx", "cy", "r", "rx", "ry", "width", "height", "stroke-width", "transform"}:
                numbers.extend(iter_numbers(value))

    if numbers:
        finite_ratio = sum(math.isfinite(n) for n in numbers) / len(numbers)
        in_soft_bounds = sum(-64 <= n <= 320 for n in numbers) / len(numbers)
        in_hard_bounds = sum(-512 <= n <= 768 for n in numbers) / len(numbers)
        positive_sizes = []
        for el in root.iter():
            for key in ("r", "rx", "ry", "width", "height", "stroke-width"):
                val = safe_float(el.attrib.get(key))
                if val is not None:
                    positive_sizes.append(1.0 if val >= 0 else 0.0)
        size_score = sum(positive_sizes) / len(positive_sizes) if positive_sizes else 0.8
        geometry = 0.35 * finite_ratio + 0.4 * in_soft_bounds + 0.15 * in_hard_bounds + 0.1 * size_score
    else:
        geometry = 0.1
        notes.append("no numeric geometry")

    colors = COLOR_RE.findall(attrs)
    hex_colors = HEX_RE.findall(attrs)
    unique_colors = {c.lower() for c in colors}
    fill_or_stroke = ("fill" in attrs.lower()) or ("stroke" in attrs.lower())
    palette_parts = [
        1.0 if fill_or_stroke else 0.0,
        1.0 if 2 <= len(unique_colors) <= 8 else 0.65 if len(unique_colors) == 1 or 9 <= len(unique_colors) <= 12 else 0.25,
        clamp(len(hex_colors) / 3.0),
        1.0 if "none" in attrs.lower() or len(unique_colors) >= 2 else 0.5,
    ]
    palette = sum(palette_parts) / len(palette_parts)

    prompt_hits = 0
    prompt_checks = 0
    concept_map = {
        "circle": ["circle", "badge", "round", "coin", "seal", "circular"],
        "rect": ["square", "rectangle", "block", "frame"],
        "line": ["line", "staff", "ray", "stripe", "tick"],
        "path": ["curve", "leaf", "wave", "flame", "sprout", "ribbon", "mountain"],
        "polygon": ["star", "triangle", "diamond", "hexagon", "shield"],
    }
    for tag, words in concept_map.items():
        if any(word in prompt_lower for word in words):
            prompt_checks += 1
            prompt_hits += 1 if tag in tags else 0

    for color in ("red", "blue", "green", "white", "black", "yellow", "orange", "purple", "pink", "gray", "grey", "navy", "teal", "gold", "brown"):
        if color in prompt_lower:
            prompt_checks += 1
            prompt_hits += 1 if color in text_lower or (color == "grey" and "gray" in text_lower) else 0

    if "#" in prompt:
        wanted_hex = {c.lower() for c in HEX_RE.findall(prompt)}
        if wanted_hex:
            prompt_checks += min(len(wanted_hex), 4)
            prompt_hits += sum(1 for c in list(wanted_hex)[:4] if c in text_lower)

    prompt_alignment = prompt_hits / prompt_checks if prompt_checks else 0.65

    if re.search(r"<svg[^>]*>\s*</svg>", svg, re.I | re.S):
        penalties += 0.3
        notes.append("empty svg")
    if len(set(svg)) < 18:
        penalties += 0.15
        notes.append("low character diversity")
    if max((len(m.group(0)) for m in re.finditer(r"(.)\1+", svg)), default=1) > 120:
        penalties += 0.12
        notes.append("repeated character run")
    external_links = [
        value
        for el in root.iter()
        for key, value in el.attrib.items()
        if key.lower().endswith("href") or key.lower() in {"src", "data"}
    ]
    if "script" in text_lower or any(value.startswith(("http://", "https://")) for value in external_links):
        penalties += 0.2
        notes.append("unsafe external/script content")

    score = (
        0.28 * validity
        + 0.22 * structure
        + 0.18 * geometry
        + 0.14 * palette
        + 0.18 * prompt_alignment
        - penalties
    )
    score = clamp(score)
    return RewardBreakdown(score, validity, structure, geometry, palette, prompt_alignment, penalties, notes)


def reward(svg_text: str, prompt: str = "") -> float:
    """Return only the scalar reward expected by training/eval code."""
    return score_svg(svg_text, prompt).score


if __name__ == "__main__":
    import argparse
    import json
    import sys

    parser = argparse.ArgumentParser()
    parser.add_argument("svg_file", nargs="?")
    parser.add_argument("--prompt", default="")
    args = parser.parse_args()

    if args.svg_file:
        content = open(args.svg_file, "r", encoding="utf-8").read()
    else:
        content = sys.stdin.read()
    print(json.dumps(score_svg(content, args.prompt).__dict__, ensure_ascii=False, indent=2))
