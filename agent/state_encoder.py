import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

KNOWN_ATTRS = [
    "education_level",
    "age",
    "gender",
    "marital_status",
    "profession",
    "economic_status",
    "health_status",
    "mental_health_status",
    "emotional_state",
]
REPLY_TYPES = ["direct", "partial", "evasive", "off_topic", None, "meta"]
MAX_TURNS = 9
QUERY_DIM = 20
STATE_DIM = len(KNOWN_ATTRS) + 1 + len(REPLY_TYPES) + QUERY_DIM


def fit_vectorizer(queries: list) -> TfidfVectorizer:
    vec = TfidfVectorizer(max_features=QUERY_DIM)
    vec.fit(queries)
    return vec


def encode(state: dict, vectorizer: TfidfVectorizer = None) -> np.ndarray:
    known = state.get("known_attributes", {})
    known_vec = [1.0 if attr in known else 0.0 for attr in KNOWN_ATTRS]

    turn_norm = state.get("turn_count", 0) / MAX_TURNS

    reply = state.get("last_reply_type", None)
    reply_vec = [1.0 if reply == rt else 0.0 for rt in REPLY_TYPES]

    if vectorizer is not None:
        query = state.get("query", "")
        query_vec = vectorizer.transform([query]).toarray()[0].tolist()
    else:
        query_vec = [0.0] * QUERY_DIM

    return np.array(known_vec + [turn_norm] + reply_vec + query_vec, dtype=np.float32)
