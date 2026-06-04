from __future__ import annotations

import json
from pathlib import Path

from datasets import Dataset

from game24_grpo.prompting import build_prompt


def load_jsonl_dataset(path: str | Path, prompt_template: str) -> Dataset:
    records = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            raw = json.loads(line)
            numbers = raw["numbers"]
            target = raw.get("target", 24)
            records.append(
                {
                    "prompt": build_prompt(numbers=numbers, target=target, prompt_template=prompt_template),
                    "numbers": numbers,
                    "target": target,
                    "solvable": raw.get("solvable", True),
                }
            )
    return Dataset.from_list(records)
