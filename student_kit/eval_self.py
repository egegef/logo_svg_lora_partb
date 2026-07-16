"""Generate and score base vs LoRA SVG outputs on valid.jsonl."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from reward import score_svg


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def get_message(messages: list[dict], role: str) -> str:
    for item in messages:
        if item.get("role") == role:
            return item.get("content", "")
    return ""


def extract_svg(text: str) -> str:
    start = text.lower().find("<svg")
    end = text.lower().rfind("</svg>")
    if start == -1:
        return text.strip()
    if end == -1:
        return text[start:].strip()
    return text[start : end + len("</svg>")].strip()


def load_model(model_path: str, adapter_path: str | None):
    tokenizer = AutoTokenizer.from_pretrained(adapter_path or model_path, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="cuda" if torch.cuda.is_available() else None,
        local_files_only=True,
    )
    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path, local_files_only=True)
    model.eval()
    return tokenizer, model


@torch.no_grad()
def generate_one(tokenizer, model, messages: list[dict], max_new_tokens: int) -> str:
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    eos_ids = [tokenizer.eos_token_id]
    end_turn_id = tokenizer.convert_tokens_to_ids("<end_of_turn>")
    if isinstance(end_turn_id, int) and end_turn_id >= 0:
        eos_ids.append(end_turn_id)
    output = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=None,
        top_p=None,
        top_k=None,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=eos_ids,
    )
    decoded = tokenizer.decode(output[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)
    return extract_svg(decoded)


def evaluate_name(name: str, tokenizer, model, rows: list[dict], max_new_tokens: int) -> dict:
    items = []
    for idx, row in enumerate(tqdm(rows, desc=name), 1):
        prompt = get_message(row["messages"], "user")
        target = get_message(row["messages"], "assistant")
        messages = [m for m in row["messages"] if m.get("role") != "assistant"]
        svg = generate_one(tokenizer, model, messages, max_new_tokens)
        b = score_svg(svg, prompt)
        items.append(
            {
                "idx": idx,
                "prompt": prompt,
                "generated_svg": svg,
                "target_svg": target,
                "score": b.score,
                "validity": b.validity,
                "structure": b.structure,
                "geometry": b.geometry,
                "palette": b.palette,
                "prompt_alignment": b.prompt_alignment,
                "penalties": b.penalties,
                "notes": b.notes,
            }
        )
    keys = ["score", "validity", "structure", "geometry", "palette", "prompt_alignment", "penalties"]
    summary = {f"mean_{k}": sum(item[k] for item in items) / len(items) for k in keys}
    return {"name": name, "summary": summary, "items": items}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="./gemma3-270m")
    parser.add_argument("--adapter", default="./adapter")
    parser.add_argument("--valid", default="./logo-detailed-prompt/valid.jsonl")
    parser.add_argument("--out", default="./results.json")
    parser.add_argument("--max-new-tokens", type=int, default=1536)
    args = parser.parse_args()

    rows = read_jsonl(Path(args.valid))
    tok, base = load_model(args.model, None)
    base_result = evaluate_name("base", tok, base, rows, args.max_new_tokens)
    del base
    torch.cuda.empty_cache()

    tok, tuned = load_model(args.model, args.adapter)
    tuned_result = evaluate_name("lora", tok, tuned, rows, args.max_new_tokens)

    result = {
        "model": args.model,
        "adapter": args.adapter,
        "valid": args.valid,
        "base": base_result,
        "lora": tuned_result,
        "delta_mean_score": tuned_result["summary"]["mean_score"] - base_result["summary"]["mean_score"],
    }
    Path(args.out).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: result[k] for k in ["model", "adapter", "delta_mean_score"]}, ensure_ascii=False, indent=2))
    print("base", base_result["summary"])
    print("lora", tuned_result["summary"])


if __name__ == "__main__":
    main()
