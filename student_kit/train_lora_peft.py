"""LoRA fine-tuning for Gemma 3 270M on detailed-prompt to SVG data."""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def split_messages(record: dict) -> tuple[list[dict], list[dict]]:
    messages = record["messages"]
    prompt_messages = [m for m in messages if m["role"] != "assistant"]
    return prompt_messages, messages


class SvgChatDataset(Dataset):
    def __init__(self, rows: list[dict], tokenizer, max_length: int):
        self.examples = []
        for row in rows:
            prompt_messages, full_messages = split_messages(row)
            prompt_text = tokenizer.apply_chat_template(
                prompt_messages, tokenize=False, add_generation_prompt=True
            )
            full_text = tokenizer.apply_chat_template(
                full_messages, tokenize=False, add_generation_prompt=False
            )
            prompt_ids = tokenizer(prompt_text, add_special_tokens=False).input_ids
            full_ids = tokenizer(full_text, add_special_tokens=False).input_ids
            if len(full_ids) > max_length:
                full_ids = full_ids[:max_length]
            labels = full_ids.copy()
            mask_len = min(len(prompt_ids), len(labels))
            labels[:mask_len] = [-100] * mask_len
            if all(x == -100 for x in labels):
                continue
            self.examples.append({"input_ids": full_ids, "labels": labels})

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict:
        return self.examples[idx]


def collate(batch: list[dict], pad_id: int) -> dict:
    max_len = max(len(x["input_ids"]) for x in batch)
    input_ids, labels, attention_mask = [], [], []
    for item in batch:
        pad = max_len - len(item["input_ids"])
        input_ids.append(item["input_ids"] + [pad_id] * pad)
        labels.append(item["labels"] + [-100] * pad)
        attention_mask.append([1] * len(item["input_ids"]) + [0] * pad)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
    }


@torch.no_grad()
def evaluate(model, loader, device: str) -> float:
    model.eval()
    losses = []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        loss = model(**batch).loss
        losses.append(float(loss.detach().cpu()))
    model.train()
    return sum(losses) / max(1, len(losses))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="./gemma3-270m")
    parser.add_argument("--train", default="./logo-detailed-prompt/train.jsonl")
    parser.add_argument("--valid", default="./logo-detailed-prompt/valid.jsonl")
    parser.add_argument("--output", default="./adapter")
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_rows = read_jsonl(Path(args.train))
    valid_rows = read_jsonl(Path(args.valid))
    train_data = SvgChatDataset(train_rows, tokenizer, args.max_length)
    valid_data = SvgChatDataset(valid_rows, tokenizer, args.max_length)
    print(f"train examples={len(train_data)} valid examples={len(valid_data)}")

    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda b: collate(b, tokenizer.pad_token_id),
    )
    valid_loader = DataLoader(
        valid_data,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda b: collate(b, tokenizer.pad_token_id),
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        local_files_only=True,
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model = get_peft_model(
        model,
        LoraConfig(
            r=args.rank,
            lora_alpha=args.alpha,
            lora_dropout=args.dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        ),
    )
    model.to(device)
    model.print_trainable_parameters()

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr)
    total_updates = math.ceil(len(train_loader) * args.epochs / args.grad_accum)
    scheduler = get_cosine_schedule_with_warmup(
        optim, num_warmup_steps=max(1, total_updates // 10), num_training_steps=total_updates
    )

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    best_loss = float("inf")
    global_step = 0
    history = []

    model.train()
    for epoch in range(1, args.epochs + 1):
        running = 0.0
        optim.zero_grad(set_to_none=True)
        pbar = tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs}")
        for step, batch in enumerate(pbar, 1):
            batch = {k: v.to(device) for k, v in batch.items()}
            loss = model(**batch).loss / args.grad_accum
            loss.backward()
            running += float(loss.detach().cpu()) * args.grad_accum
            if step % args.grad_accum == 0 or step == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optim.step()
                scheduler.step()
                optim.zero_grad(set_to_none=True)
                global_step += 1
            pbar.set_postfix(loss=f"{running / step:.4f}")

        valid_loss = evaluate(model, valid_loader, device)
        item = {"epoch": epoch, "train_loss": running / len(train_loader), "valid_loss": valid_loss}
        history.append(item)
        print(json.dumps(item, ensure_ascii=False))
        if valid_loss < best_loss:
            best_loss = valid_loss
            model.save_pretrained(output)
            tokenizer.save_pretrained(output)
            print(f"saved best adapter to {output} valid_loss={best_loss:.4f}")

    (output / "training_history.json").write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
