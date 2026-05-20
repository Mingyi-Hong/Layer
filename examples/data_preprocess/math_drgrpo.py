"""Preprocess MATH for Dr. GRPO training.

Two modes (paper: arXiv:2503.20783):
  --mode official_12k   1.5B recipe: official math_12k Arrow dataset from
                        github.com/sail-sg/understand-r1-zero (MATH train+test,
                        all levels, 12000 questions). Default.
  --mode lvl3to5        7B recipe: DigitalLearningGmbH/MATH-lighteval train split
                        filtered to Level 3-5 (~5586 questions).

Both modes apply the Qwen-Math system prompt
("Please reason step by step, and put your final answer within \\boxed{}.").

Test split is always MATH-500 (HuggingFaceH4/MATH-500).
"""

import argparse
import json
import os

import datasets

from verl.utils.reward_score.math_reward import last_boxed_only_string, remove_boxed


SYSTEM_PROMPT = (
    "Please reason step by step, and put your final answer within \\boxed{}."
)

LEVELS_KEEP_LVL3TO5 = {"Level 3", "Level 4", "Level 5"}


def extract_solution(solution_str: str) -> str:
    return remove_boxed(last_boxed_only_string(solution_str))


def make_row(question: str, answer: str, idx: int, split: str, level=None, subject=None) -> dict:
    return {
        "data_source": "math_drgrpo",
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ],
        "ability": "math",
        "reward_model": {"style": "rule", "ground_truth": str(answer)},
        "extra_info": {
            "split": split,
            "index": idx,
            "level": str(level) if level is not None else "Unknown",
            "subject": str(subject) if subject is not None else "Unknown",
        },
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["official_12k", "lvl3to5"], default="official_12k")
    parser.add_argument("--local_save_dir", default=None, help="Default: ~/data/math_drgrpo_<mode>")
    parser.add_argument(
        "--official_data_path",
        default="/tmp/understand-r1-zero/datasets/train/math_12k",
        help="Path to official math_12k Arrow dataset (mode=official_12k only)",
    )
    parser.add_argument(
        "--lvl3to5_source",
        default="DigitalLearningGmbH/MATH-lighteval",
        help="HF id for lvl3to5 mode source",
    )
    parser.add_argument(
        "--test_source",
        default="HuggingFaceH4/MATH-500",
        help="HF id for the test set",
    )
    args = parser.parse_args()

    save_dir = args.local_save_dir or f"~/data/math_drgrpo_{args.mode}"
    out_dir = os.path.expanduser(save_dir)
    os.makedirs(out_dir, exist_ok=True)

    if args.mode == "official_12k":
        # 1.5B paper recipe: use the official math_12k Arrow dataset as-is.
        # It has columns: problem, solution, answer, subject, level, unique_id.
        # `answer` is already extracted (no need to parse \boxed{} from solution).
        print(f"Loading official math_12k: {args.official_data_path}", flush=True)
        raw = datasets.load_from_disk(args.official_data_path)["train"]
        print(f"  total: {len(raw)}")
        train_ds = raw.map(
            lambda ex, idx: make_row(
                ex["problem"], ex["answer"], idx, "train",
                level=ex.get("level"), subject=ex.get("subject"),
            ),
            with_indices=True,
            remove_columns=raw.column_names,
        )
    else:  # lvl3to5
        print(f"Loading train: {args.lvl3to5_source}", flush=True)
        raw = datasets.load_dataset(args.lvl3to5_source, split="train")
        print(f"  total: {len(raw)}")
        raw = raw.filter(lambda ex: ex.get("level") in LEVELS_KEEP_LVL3TO5)
        print(f"  level 3-5 kept: {len(raw)}")
        train_ds = raw.map(
            lambda ex, idx: make_row(
                ex["problem"], extract_solution(ex["solution"]), idx, "train",
                level=ex.get("level"), subject=ex.get("type"),
            ),
            with_indices=True,
            remove_columns=raw.column_names,
        )

    # ---- test: MATH-500 (same for both modes) ----
    print(f"Loading test: {args.test_source}", flush=True)
    test_raw = datasets.load_dataset(args.test_source, split="test")
    print(f"  total: {len(test_raw)}")
    # MATH-500 has columns: problem, answer (already extracted), subject, level
    test_ds = test_raw.map(
        lambda ex, idx: make_row(
            ex["problem"], ex["answer"], idx, "test",
            level=ex.get("level"), subject=ex.get("subject"),
        ),
        with_indices=True,
        remove_columns=test_raw.column_names,
    )

    train_path = os.path.join(out_dir, "train.parquet")
    test_path = os.path.join(out_dir, "test.parquet")
    train_ds.to_parquet(train_path)
    test_ds.to_parquet(test_path)
    print(f"saved {train_path}: {len(train_ds)} rows")
    print(f"saved {test_path}: {len(test_ds)} rows")

    with open(os.path.join(out_dir, "train_example.json"), "w") as f:
        json.dump(train_ds[0], f, indent=2, ensure_ascii=False)
    with open(os.path.join(out_dir, "test_example.json"), "w") as f:
        json.dump(test_ds[0], f, indent=2, ensure_ascii=False)
    print("done.")
