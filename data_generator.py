import json
from groq import Groq
import pandas as pd
import time
import os
import random

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Load dataset
df = pd.read_json("Personalized_safety_data.json")
print(df.head())

# CONFIG
NUM_SAMPLES = 500
CHUNK_SIZE = 50   # process 50 queries at a time
SLEEP_TIME = 3    # safer for rate limit


# SAMPLE DATA
df_subset = df.iloc[:4000]
df_sampled = df_subset.sample(n=NUM_SAMPLES, random_state=42).reset_index(drop=True)

# FUNCTIONS
def generate_reply(prompt, max_retries=5):
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7
            )
            return response.choices[0].message.content.strip()

        except Exception:
            print(f"Rate limit hit (attempt {attempt+1}), waiting...")
            time.sleep(5)

    return "ERROR: skipped after retries"


def build_prompt(query, profile, attribute, reply_type):
    return f"""
You are simulating a human user in a conversation.

User query: "{query}"

User profile:
- profession: {profile['profession']}
- economic_status: {profile['economic_status']}
- health_status: {profile['health_status']}
- mental_health_status: {profile['mental_health_status']}
- emotional_state: {profile['emotional_state']}

The assistant asks about: {attribute}

Reply type: {reply_type}

STRICT INSTRUCTIONS:
- direct → clearly answer with full information
- partial → vague answer, DO NOT mention the exact condition
- evasive → avoid answering completely
- off-topic → DO NOT mention the attribute at all

ADDITIONAL CONSTRAINTS:
- Keep response short (1–2 lines)
- Do not explain
- Sound like a real human

Return only the reply.
"""

# MAIN LOOP (CHUNK PROCESSING)

attributes = [
    "profession",
    "economic_status",
    "health_status",
    "mental_health_status",
    "emotional_state"
]

reply_types = ["direct", "partial", "evasive", "off-topic"]

num_chunks = len(df_sampled) // CHUNK_SIZE

for chunk_idx in range(num_chunks + 1):

    start = chunk_idx * CHUNK_SIZE
    end = start + CHUNK_SIZE

    df_chunk = df_sampled.iloc[start:end]

    if df_chunk.empty:
        break

    print(f"\nProcessing chunk {chunk_idx}: rows {start} to {end}")

    data = []

    for i, row in df_chunk.iterrows():

        profile = {
            "profession": "unknown",
            "economic_status": "unknown",
            "health_status": row["health_status"],
            "mental_health_status": row["mental_health_status"],
            "emotional_state": row["emotional_state"]
        }

        query = row["query"]

        for attribute in attributes:
            for r_type in reply_types:

                print(f"Row {i}, Attr {attribute}, Type {r_type}")

                prompt = build_prompt(
                    query=query,
                    profile=profile,
                    attribute=attribute,
                    reply_type=r_type
                )

                reply = generate_reply(prompt)

                if reply == "ERROR: skipped after retries":
                    continue

                data.append({
                    "query": query,
                    "profile": profile,
                    "attribute_asked": attribute,
                    "reply_type": r_type,
                    "reply": reply
                })

                time.sleep(SLEEP_TIME)

    # SAVE EACH CHUNK
    output_file = f"simulated_dataset_chunk_{chunk_idx}.json"

    with open(output_file, "w") as f:
        json.dump(data, f, indent=4)

    print(f"Saved {output_file} with {len(data)} samples")