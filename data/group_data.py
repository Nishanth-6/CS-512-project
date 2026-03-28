import json

INPUT_FILE = "generated_user_responses.json"
OUTPUT_FILE = "grouped_conversations.json"

# load raw data
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

grouped = {}

for row in data:
    # skip bad rows
    if row.get("generation_status") != "success":
        continue

    row_index = row["row_index"]

    # if new conversation, initialize
    if row_index not in grouped:
        grouped[row_index] = {
            "row_index": row_index,
            "query": row["query"],
            "scenario": row["scenario"],
            "ground_truth_attributes": {},
            "responses": {}
        }

    attr_name = row["source_column"]  # snake_case already
    attr_value = row["original_value_of_attribute"]

    # store attribute value
    grouped[row_index]["ground_truth_attributes"][attr_name] = attr_value

    # store responses
    grouped[row_index]["responses"][f"ask_{attr_name}"] = row["generated_responses"]

# convert dict → list
final_data = list(grouped.values())

# save output
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(final_data, f, indent=2, ensure_ascii=False)

print(f"Grouped {len(final_data)} conversations and saved to {OUTPUT_FILE}")