import json

files = [
    "data/generated_user_responses.json",
    "data/generated_user_responses1.json",
    "data/generated_user_responses2.json",
    "data/generated_user_responses3.json",
    "data/generated_user_responses4.json"
]

all_data = []

for file_path in files:
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        all_data.extend(data)

with open("data/merged_raw_new.json", "w", encoding="utf-8") as f:
    json.dump(all_data, f, indent=2, ensure_ascii=False)

print(f"Merged {len(all_data)} rows into data/merged_raw_new.json")