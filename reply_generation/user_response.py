import pandas as pd
data = pd.read_csv('data.csv')

import os
import json
import time
import random
from typing import Dict, Any, List, Optional

from groq import Groq

os.environ['GROQ_API_KEY'] = 'your key'
# =========================================================
# CONFIG
# =========================================================
INPUT_JSON = "input_dataset.json"
OUTPUT_JSON = "generated_user_responses.json"

MODEL_NAME = "llama-3.1-8b-instant"

MAX_RETRIES = 3
BASE_SLEEP = 2

TEMPERATURE = 0.7
MAX_TOKENS = 250


ATTRIBUTE_COLUMN_MAP = {
    "education_level": "Education Level",
    "age": "Age",
    "gender": "Gender",
    "marital_status": "Marital Status",
    "profession": "Profession", 
    "economic_status": "Economic Status", 
    "health_status": "Health Status",
    "mental_health_status": "Mental Health Status",
    "emotional_state" : "Emotional State"
}


# =========================================================
# PROMPT
# =========================================================
SYSTEM_PROMPT = """
You are simulating a real user in a multi-turn conversation with an AI assistant.

You are given:
1. the user's original query
2. the scenario or context of the query
3. the attribute the assistant is asking about
4. the user's true value for that attribute

Your task:
Generate 4 possible user replies for the asked attribute:
- direct
- partial
- evasive
- off_topic

Definitions:
- direct: clearly reveals the true attribute value
- partial: gives only broad, incomplete, or less precise information related to the true value
- evasive: avoids answering the question, resists answering, or asks why the information is needed
- off_topic: says something natural in the conversation but does not answer the asked attribute

Rules:
- Keep replies short, natural, and realistic
- Stay consistent with the user's original query and scenario
- Do not sound robotic
- Do not include labels in the reply text
- The direct reply must reflect the true attribute value
- The partial reply must be weaker or less precise than direct
- The evasive reply must not reveal the attribute value
- The off_topic reply must not reveal the attribute value
- Return valid JSON only

Return exactly this JSON schema:
{
  "direct": "...",
  "partial": "...",
  "evasive": "...",
  "off_topic": "..."
}
""".strip()


def build_user_prompt(
    query: str,
    scenario: str,
    attribute_name: str,
    original_value_of_attribute: str
) -> str:
    return f"""
Query: {query}
Scenario: {scenario}
Asked attribute: {attribute_name}
True attribute value: {original_value_of_attribute}

Generate the JSON now.
""".strip()


# =========================================================
# CLIENT
# =========================================================
def get_groq_client() -> Groq:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError(
            "GROQ_API_KEY environment variable is not set. "
            "Please set it before running the script."
        )
    return Groq(api_key=api_key)



def save_output_json(path: str, data: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)



def safe_json_extract(text: str) -> Dict[str, str]:
    text = text.strip()

    if text.startswith("```"):
        lines = text.splitlines()
        if len(lines) >= 3:
            text = "\n".join(lines[1:-1]).strip()

    first_brace = text.find("{")
    last_brace = text.rfind("}")
    if first_brace != -1 and last_brace != -1:
        text = text[first_brace:last_brace + 1]

    data = json.loads(text)

    required_keys = ["direct", "partial", "evasive", "off_topic"]
    for key in required_keys:
        if key not in data:
            raise ValueError(f"Missing key in model output: {key}")
        if not isinstance(data[key], str):
            raise ValueError(f"Value for '{key}' must be a string")

    return {
        "direct": data["direct"].strip(),
        "partial": data["partial"].strip(),
        "evasive": data["evasive"].strip(),
        "off_topic": data["off_topic"].strip(),
    }


# =========================================================
# MODEL CALL
# =========================================================
def generate_user_replies(
    client: Groq,
    query: str,
    scenario: str,
    attribute_name: str,
    original_value_of_attribute: str,
    model_name: str = MODEL_NAME,
    temperature: float = TEMPERATURE,
    max_tokens: int = MAX_TOKENS,
) -> Dict[str, str]:
    prompt = build_user_prompt(
        query=query,
        scenario=scenario,
        attribute_name=attribute_name,
        original_value_of_attribute=original_value_of_attribute,
    )

    last_error: Optional[Exception] = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            completion = client.chat.completions.create(
                model=model_name,
                temperature=temperature,
                max_completion_tokens=max_tokens,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
            )

            content = completion.choices[0].message.content
            if not content:
                raise ValueError("Empty response from model")

            return safe_json_extract(content)

        except Exception as e:
            last_error = e
            sleep_time = BASE_SLEEP * attempt + random.uniform(0, 1)
            print(f"[Retry {attempt}/{MAX_RETRIES}] Error: {e}")
            time.sleep(sleep_time)

    raise RuntimeError(f"Generation failed after retries. Last error: {last_error}")




# =========================================================
# PROCESS DATASET
def process_dataframe_multi_attributes(
    test_data: pd.DataFrame,
    output_json: str = OUTPUT_JSON,
    attribute_column_map: Dict[str, str] = ATTRIBUTE_COLUMN_MAP,
) -> List[Dict[str, Any]]:
    client = get_groq_client()
    
    if test_data is None or test_data.empty:
        raise ValueError("test_data is empty")

    required_base_cols = ["query", "scenario"]
    missing_base_cols = [col for col in required_base_cols if col not in test_data.columns]
    if missing_base_cols:
        raise ValueError(f"Missing required dataframe columns: {missing_base_cols}")

    print("Columns in dataframe:", list(test_data.columns))
    print(f"Loaded {len(test_data)} rows from dataframe")

    output_records: List[Dict[str, Any]] = []
    total_tasks = 0

    for col in attribute_column_map:
        if col in test_data.columns:
            total_tasks += test_data[col].notna().sum()

    task_counter = 0

    for row_idx, (_, row) in enumerate(test_data.iterrows()):
        query = str(row["query"]).strip()
        scenario = str(row["scenario"]).strip()

        for attr_col, attr_name in attribute_column_map.items():
            if attr_col not in test_data.columns:
                continue

            raw_value = row[attr_col]

            original_value_of_attribute = str(raw_value).strip()
            task_counter += 1

            print(
                f"Processing task {task_counter}/{total_tasks} | "
                f"row={row_idx + 1} | attribute={attr_name}"
            )

            try:
                replies = generate_user_replies(
                    client=client,
                    query=query,
                    scenario=scenario,
                    attribute_name=attr_name,
                    original_value_of_attribute=original_value_of_attribute,
                )

                output_record = {
                    "row_index": row_idx,
                    "query": query,
                    "scenario": scenario,
                    "attribute_name": attr_name,
                    "source_column": attr_col,
                    "original_value_of_attribute": original_value_of_attribute,
                    "generated_responses": {
                        "direct": replies["direct"],
                        "partial": replies["partial"],
                        "evasive": replies["evasive"],
                        "off_topic": replies["off_topic"],
                    },
                }

            except Exception as e:
                output_record = {
                    "row_index": row_idx,
                    "query": query,
                    "scenario": scenario,
                    "attribute_name": attr_name,
                    "source_column": attr_col,
                    "original_value_of_attribute": original_value_of_attribute,
                    "generated_responses": {
                        "direct": "",
                        "partial": "",
                        "evasive": "",
                        "off_topic": "",
                    }, 
                }
                print(f"Failed row {row_idx + 1}, attribute {attr_name}: {e}")

            output_records.append(output_record)

            if task_counter % 10 == 0:
                save_output_json(output_json, output_records)
                print(f"Intermediate save written to {output_json}")

    save_output_json(output_json, output_records)
    print(f"Done. Final output saved to {output_json}")

    return output_records



# =========================================================
# ENTRY POINT
# =========================================================
if __name__ == "__main__":
    # Uncomment to test just one example
    # test_single_example()
    process_dataframe_multi_attributes(test_data, OUTPUT_JSON, ATTRIBUTE_COLUMN_MAP)
