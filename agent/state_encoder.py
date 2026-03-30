import numpy as np

KNOWN_ATTRS = ["education_level", "age", "gender", "marital_status"]
REPLY_TYPES = ["direct", "partial", "evasive", "off_topic", None, "meta"]
MAX_TURNS = 6
STATE_DIM = len(KNOWN_ATTRS) + 1 + len(REPLY_TYPES)

def encode(state: dict) -> np.ndarray:
    known = state.get("known_attributes", {})
    known_vec = [1.0 if attr in known else 0.0 for attr in KNOWN_ATTRS]

    turn_norm = state.get("turn_count", 0) / MAX_TURNS

    reply = state.get("last_reply_type", None)
    reply_vec = [1.0 if reply == rt else 0.0 for rt in REPLY_TYPES]

    return np.array(known_vec + [turn_norm] + reply_vec, dtype=np.float32)
