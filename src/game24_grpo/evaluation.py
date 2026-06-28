from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
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
    number_use_valid_rate: float
    value_correct_rate: float
    unsolvable_false_positive_rate: float

    @property
    def hallucination_rate(self) -> float:
        return self.unsolvable_false_positive_rate


def _infer_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _resolve_torch_dtype(dtype: str, device: str) -> torch.dtype:
    normalized = dtype.lower()
    if normalized == "auto":
        return torch.float32 if device == "cpu" else torch.float32
    if normalized in {"float32", "fp32"}:
        return torch.float32
    if normalized in {"float16", "fp16"}:
        return torch.float16
    if normalized in {"bfloat16", "bf16"}:
        return torch.bfloat16
    raise ValueError(f"unsupported dtype: {dtype}")


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


def _prepare_model_inputs(
    prompt: str,
    tokenizer: AutoTokenizer,
    device: str,
) -> dict[str, torch.Tensor]:
    rendered_prompt = _build_generation_prompt(prompt, tokenizer)
    inputs = tokenizer(rendered_prompt, return_tensors="pt")
    return {key: value.to(device) for key, value in inputs.items()}


def evaluate_model(
    model_name: str,
    dataset_path: str,
    prompt_template: str,
    split: str,
    max_new_tokens: int = 192,
    limit: int | None = None,
    output_path: str | None = None,
    experiment_id: str = "eval",
    dtype: str = "auto",
    temperature: float | None = None,
    top_p: float | None = None,
    num_samples: int = 1,
    baseline_type: str = "untrained_qwen",
    trained: bool = False,
    checkpoint: str | None = None,
    overlap_filter: str = "",
    difficulty_filter: str = "",
    notes: str = "",
) -> EvalMetrics:
    if num_samples < 1:
        raise ValueError("num_samples must be at least 1")
    if num_samples > 1 and (temperature is None or temperature <= 0):
        raise ValueError("num_samples > 1 requires sampling with --temperature > 0")

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
    torch_dtype = _resolve_torch_dtype(dtype, device)
    print(f"[evaluate] loading model on {device} with dtype={torch_dtype}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
    ).to(device)
    model.eval()
    print("[evaluate] starting generation loop")

    solved = 0
    format_pass = 0
    valid_expression = 0
    valid_number_use = 0
    value_correct = 0
    unsolvable_false_positives = 0
    rows: list[dict[str, Any]] = []
    for index, row in enumerate(dataset, start=1):
        inputs = _prepare_model_inputs(row["prompt"], tokenizer, device)
        do_sample = temperature is not None and temperature > 0
        generation_kwargs: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "num_return_sequences": num_samples,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }
        if do_sample:
            generation_kwargs["temperature"] = temperature
            if top_p is not None:
                generation_kwargs["top_p"] = top_p
        with torch.no_grad():
            output = model.generate(
                **inputs,
                **generation_kwargs,
            )
        completions = [
            tokenizer.decode(sequence[inputs["input_ids"].shape[1] :], skip_special_tokens=True)
            for sequence in output
        ]
        verifications = [
            verify_completion(completion, row["numbers"])
            for completion in completions
        ]
        selected_index = next((i for i, item in enumerate(verifications) if item.is_correct), 0)
        completion = completions[selected_index]
        verification = verifications[selected_index]

        solved += int(verification.is_correct)
        format_pass += int(verification.has_answer_tag)
        valid_expression += int(verification.is_valid_expression)
        valid_number_use += int(verification.used_numbers_match)
        value_correct += int(verification.evaluates_to_24)
        unsolvable_false_positives += int(
            (not row["solvable"]) and verification.is_valid_expression
        )
        rows.append(
            {
                "index": index - 1,
                "numbers": row["numbers"],
                "solvable": row["solvable"],
                "target": row["target"],
                "hard": split == "eval",
                "selected_sample": selected_index,
                "raw_model_output": completion,
                "extracted_answer": verification.answer_text,
                "verifier_status": {
                    "is_correct": verification.is_correct,
                    "has_answer_tag": verification.has_answer_tag,
                    "is_valid_expression": verification.is_valid_expression,
                    "used_numbers_match": verification.used_numbers_match,
                    "evaluates_to_24": verification.evaluates_to_24,
                    "value": verification.value,
                    "failure_reason": verification.error,
                },
                "samples": [
                    {
                        "raw_model_output": sample_completion,
                        "extracted_answer": sample_verification.answer_text,
                        "verifier_status": {
                            "is_correct": sample_verification.is_correct,
                            "has_answer_tag": sample_verification.has_answer_tag,
                            "is_valid_expression": sample_verification.is_valid_expression,
                            "used_numbers_match": sample_verification.used_numbers_match,
                            "evaluates_to_24": sample_verification.evaluates_to_24,
                            "value": sample_verification.value,
                            "failure_reason": sample_verification.error,
                        },
                    }
                    for sample_completion, sample_verification in zip(
                        completions,
                        verifications,
                        strict=True,
                    )
                ],
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
        number_use_valid_rate=valid_number_use / total if total else 0.0,
        value_correct_rate=value_correct / total if total else 0.0,
        unsolvable_false_positive_rate=unsolvable_false_positives / total if total else 0.0,
    )
    if output_path is not None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        payload = {
            "experiment_id": experiment_id,
            "timestamp": timestamp,
            "git_commit_or_diff_summary": "",
            "remote_host": "nlp",
            "model": model_name,
            "checkpoint": checkpoint or model_name,
            "baseline_type": baseline_type,
            "trained": trained,
            "dataset": dataset_path,
            "split": split,
            "overlap_filter": overlap_filter,
            "difficulty_filter": difficulty_filter,
            "num_examples": total,
            "prompt_variant": "configs/data prompt_template",
            "reward_variant": "none",
            "generation_config": {
                "temperature": temperature,
                "top_p": top_p,
                "num_samples": num_samples,
                "max_new_tokens": max_new_tokens,
                "do_sample": temperature is not None and temperature > 0,
                "dtype": str(torch_dtype).replace("torch.", ""),
            },
            "device": device,
            "metrics": {
                "solved_rate": metrics.solve_rate,
                "format_valid_rate": metrics.format_pass_rate,
                "expression_valid_rate": metrics.valid_expression_rate,
                "number_use_valid_rate": metrics.number_use_valid_rate,
                "value_correct_rate": metrics.value_correct_rate,
                "unsolvable_false_positive_rate": metrics.unsolvable_false_positive_rate,
            },
            "per_example": rows,
            "notes": notes,
        }
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with output_file.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=True)
            handle.write("\n")
        print(f"[evaluate] saved results to {output_file}")
    return metrics
