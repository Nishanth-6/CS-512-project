import json
from collections import defaultdict

SIMULATED_FILES = [
    "simulated_dataset_0_10.json",
    "simulated_dataset_10_30.json",
    "simulated_dataset_30_40.json",
]

MERGED_9ATTR = "data/merged_9attr_dataset.json"
OUTPUT = "data/full_40_dataset.json"

with open(MERGED_9ATTR) as f:
    base_data = json.load(f)

base_lookup = {s["query"]: s for s in base_data}

all_flat = []
for path in SIMULATED_FILES:
    with open(path) as f:
        all_flat.extend(json.load(f))

sim_grouped = defaultdict(lambda: {"profile": {}, "responses": {}})
for s in all_flat:
    q = s["query"]
    sim_grouped[q]["query"] = q
    sim_grouped[q]["profile"].update(s["profile"])
    attr = s["attribute_asked"]
    action = f"ask_{attr}"
    reply_type = s["reply_type"].replace("-", "_")
    if action not in sim_grouped[q]["responses"]:
        sim_grouped[q]["responses"][action] = {}
    sim_grouped[q]["responses"][action][reply_type] = s["reply"]

output = []
for query, sim in sim_grouped.items():
    if query in base_lookup:
        base = base_lookup[query]
        merged_responses = dict(base["responses"])
        for action, replies in sim["responses"].items():
            if action not in merged_responses:
                merged_responses[action] = replies
            else:
                merged_responses[action].update(replies)
        record = {
            "query": query,
            "scenario": base.get("scenario", "education"),
            "ground_truth_attributes": base["ground_truth_attributes"],
            "responses": merged_responses,
        }
    else:
        invalid_values = {"unknown", "not specified", "", None}
        gt = {k: v for k, v in sim["profile"].items() if v not in invalid_values}
        record = {
            "query": query,
            "scenario": "education",
            "ground_truth_attributes": gt,
            "responses": sim["responses"],
        }
    output.append(record)

with open(OUTPUT, "w") as f:
    json.dump(output, f, indent=2)

print(f"Saved {len(output)} samples to {OUTPUT}")
print(f"Sample keys: {list(output[0].keys())}")
print(f"Sample gt attrs: {list(output[0]['ground_truth_attributes'].keys())}")
print(f"Sample response keys: {list(output[0]['responses'].keys())}")
