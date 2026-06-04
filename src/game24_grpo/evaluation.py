from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from game24_grpo.data import load_jsonl_dataset
from game24_grpo.verifier import verify_completion


@dataclass
class EvalMetrics:
    total: int
    solve_rate: float
    format_pass_rate: float
    valid_expression_rate: float
    hallucination_rate: float


def _infer_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _build_generation_prompt(prompt: str, tokenizer: AutoTokenizer) -> str:
    messages = [{"role": "user", "content": prompt}]
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception:  # noqa: BLE001
        return prompt


def _prepare_model_inputs(prompt: str, tokenizer: AutoTokenizer, device: str) -> dict[str, torch.Tensor]:
    rendered_prompt = _build_generation_prompt(prompt, tokenizer)
    inputs = tokenizer(rendered_prompt, return_tensors="pt")
    return {key: value.to(device) for key, value in inputs.items()}


def evaluate_model(
    model_name: str,
    dataset_path: str,
    prompt_template: str,
    max_new_tokens: int = 192,
    limit: int | None = None,
    output_path: str | None = None,
) -> EvalMetrics:
    print(f"[evaluate] loading dataset: {dataset_path}")
    dataset = load_jsonl_dataset(dataset_path, prompt_template)
    if limit is not None:
        dataset = dataset.select(range(min(limit, len(dataset))))
    print(f"[evaluate] dataset ready: {len(dataset)} rows")

    print(f"[evaluate] loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    device = _infer_device()
    print(f"[evaluate] loading model on {device}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        trust_remote_code=True,
    ).to(device)
    model.eval()
    print("[evaluate] starting generation loop")

    solved = 0
    format_pass = 0
    valid_expression = 0
    hallucinations = 0
    rows: list[dict[str, Any]] = []
    for index, row in enumerate(dataset, start=1):
        inputs = _prepare_model_inputs(row["prompt"], tokenizer, device)
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        completion = tokenizer.decode(output[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)
        verification = verify_completion(completion, row["numbers"])
        solved += int(verification.is_correct)
        format_pass += int(verification.has_answer_tag)
        valid_expression += int(verification.is_valid_expression)
        hallucinations += int((not row["solvable"]) and verification.is_valid_expression)
        rows.append(
            {
                "numbers": row["numbers"],
                "solvable": row["solvable"],
                "completion": completion,
                "answer_text": verification.answer_text,
                "is_correct": verification.is_correct,
                "has_answer_tag": verification.has_answer_tag,
                "is_valid_expression": verification.is_valid_expression,
                "used_numbers_match": verification.used_numbers_match,
                "evaluates_to_24": verification.evaluates_to_24,
                "error": verification.error,
            }
        )
        if index % 10 == 0 or index == len(dataset):
            print(f"[evaluate] processed {index}/{len(dataset)} examples")

    total = len(dataset)
    metrics = EvalMetrics(
        total=total,
        solve_rate=solved / total if total else 0.0,
        format_pass_rate=format_pass / total if total else 0.0,
        valid_expression_rate=valid_expression / total if total else 0.0,
        hallucination_rate=hallucinations / total if total else 0.0,
    )
    if output_path is not None:
        payload = {
            "model": model_name,
            "dataset_path": dataset_path,
            "device": device,
            "metrics": metrics.__dict__,
            "rows": rows,
        }
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with output_file.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=True)
            handle.write("\n")
        print(f"[evaluate] saved results to {output_file}")
    return metrics
