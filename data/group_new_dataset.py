import json
from collections import defaultdict

INPUT_FILE = "data/merged_raw_new.json"
OUTPUT_FILE = "data/merged_9attr_dataset.json"

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

grouped = defaultdict(list)

for row in data:
    grouped[row["row_index"]].append(row)

final_data = []

for row_index, rows in grouped.items():
    first = rows[0]

    conversation = {
        "row_index": row_index,
        "query": first["query"],
        "scenario": first.get("scenario", "unknown"),
        "ground_truth_attributes": {},
        "responses": {}
    }

    for r in rows:
        attr = r["source_column"]
        action = f"ask_{attr}"

        conversation["ground_truth_attributes"][attr] = r["original_value_of_attribute"]
        conversation["responses"][action] = r["generated_responses"]

    final_data.append(conversation)

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(final_data, f, indent=2, ensure_ascii=False)

print(f"Saved {len(final_data)} grouped conversations to {OUTPUT_FILE}")