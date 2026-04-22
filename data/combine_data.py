import json

input_files = [
    "generated_user_responses.json",
    "generated_user_responses1.json",
    "generated_user_responses2.json",
    "generated_user_responses3.json",
    "generated_user_responses4.json",
]

combined_data = []
current_offset = 0

for file_idx, file_name in enumerate(input_files):
    with open(file_name, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not data:
        continue

    # get max row_index from this file before shifting
    file_max_index = max(item["row_index"] for item in data)

    for item in data:
        new_item = item.copy()
        new_item["row_index"] = item["row_index"] + current_offset
        combined_data.append(new_item)

    # next file should start after this file's max row_index
    current_offset += file_max_index + 1

with open("combined_output.json", "w", encoding="utf-8") as f:
    json.dump(combined_data, f, indent=2, ensure_ascii=False)

print("Done. Combined file saved as combined_output.json")