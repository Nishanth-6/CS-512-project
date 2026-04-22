"""
build_dataset.py
----------------
Converts combined_output.json (flat, one record per query×attribute)
into full_dataset.json (one record per user profile), ready for ConversationEnv.

Input format (combined_output.json):
  List of records, each with:
    row_index                   -- unique user profile ID (0-499)
    query                       -- the user's question
    scenario                    -- scenario/domain label
    attribute_name              -- human-readable attr name (unused after grouping)
    source_column               -- snake_case attr key, e.g. "education_level"
    original_value_of_attribute -- ground truth value for this attribute
    generated_responses         -- dict with keys: direct, partial, evasive, off_topic

Output format (full_dataset.json):
  List of profile records, each with:
    query                       -- the user's question
    scenario                    -- scenario/domain label
    ground_truth_attributes     -- dict of {attr_key: value} for all 9 attributes
    responses                   -- dict of {ask_<attr>: {direct:.., partial:.., evasive:.., off_topic:..}}
"""

import json
import os
from collections import defaultdict

# ── Paths ─────────────────────────────────────────────────────────────────────
INPUT_PATH  = "data/combined_output.json"   # flat file from data generation
OUTPUT_PATH = "data/full_dataset.json" # consumed by DataLoader / ConversationEnv

# Values that indicate an attribute was not actually provided
INVALID_VALUES = {"not given", "unknown", "not specified", "", None}

# ── Build ──────────────────────────────────────────────────────────────────────
def build(input_path: str = INPUT_PATH, output_path: str = OUTPUT_PATH) -> list:
    with open(input_path, encoding="utf-8") as f:
        raw = json.load(f)

    # Group flat records by row_index (= unique user profile)
    profiles: dict = defaultdict(lambda: {
        "query": None,
        "scenario": None,
        "ground_truth_attributes": {},
        "responses": {},
    })

    for record in raw:
        idx     = record["row_index"]
        col     = record["source_column"]          # e.g. "education_level"
        action  = f"ask_{col}"                     # e.g. "ask_education_level"
        value   = record["original_value_of_attribute"]
        replies = record["generated_responses"]    # {direct, partial, evasive, off_topic}

        # Set query and scenario from the first record for this profile
        if profiles[idx]["query"] is None:
            profiles[idx]["query"]    = record["query"]
            profiles[idx]["scenario"] = record["scenario"]

        # Ground truth: only store non-missing values
        if value not in INVALID_VALUES:
            profiles[idx]["ground_truth_attributes"][col] = value

        # Responses: merge all four reply types under the action key
        if action not in profiles[idx]["responses"]:
            profiles[idx]["responses"][action] = {}
        profiles[idx]["responses"][action].update(replies)

    dataset = list(profiles.values())

    # ── Validation ────────────────────────────────────────────────────────────
    missing_query    = sum(1 for s in dataset if not s["query"])
    missing_gt       = sum(1 for s in dataset if len(s["ground_truth_attributes"]) == 0)
    missing_resp     = sum(1 for s in dataset if len(s["responses"]) == 0)
    gt_counts        = [len(s["ground_truth_attributes"]) for s in dataset]
    resp_counts      = [len(s["responses"]) for s in dataset]

    print(f"Total profiles   : {len(dataset)}")
    print(f"Missing query    : {missing_query}")
    print(f"Missing GT attrs : {missing_gt}")
    print(f"Missing responses: {missing_resp}")
    print(f"GT attr counts   : min={min(gt_counts)}, max={max(gt_counts)}, avg={sum(gt_counts)/len(gt_counts):.1f}")
    print(f"Response counts  : min={min(resp_counts)}, max={max(resp_counts)}, avg={sum(resp_counts)/len(resp_counts):.1f}")

    # ── Write ─────────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)

    print(f"\nSaved {len(dataset)} profiles → {output_path}")
    return dataset


if __name__ == "__main__":
    build()