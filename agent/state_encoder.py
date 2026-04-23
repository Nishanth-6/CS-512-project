"""
state_encoder.py - MiniLM sentence embedding version
STATE_DIM = 417
  [  0:  9]  known_attrs   binary presence flags
  [  9: 12]  mental_risk   one-hot severity: low/med/high
  [ 12: 15]  econ_risk     one-hot severity: stable/moderate/struggling
  [ 15: 18]  health_risk   one-hot severity: good/moderate/poor
  [ 18: 19]  turn_norm     turn_count / MAX_TURNS
  [ 19: 23]  last_reply    one-hot: direct/partial/evasive/off_topic
  [ 23: 33]  last_action   one-hot: 9 ask_* + stop
  [ 33:417]  query_embed   MiniLM-L6-v2 384-dim sentence embedding

Changes from previous version:
  1. TF-IDF (50-dim) replaced with MiniLM-L6-v2 (384-dim semantic embeddings)
  2. Scenario risk profile (8-dim) REMOVED - used ground-truth label not
     available at inference time. MiniLM captures the same signal from query text.
  3. Singleton encoder + lru_cache so each query is embedded only once.
  4. fit_vectorizer() kept as backwards-compat shim.

Install: pip install sentence-transformers
"""

import numpy as np
from functools import lru_cache

KNOWN_ATTRS = [
    "education_level", "age", "gender", "marital_status", "profession",
    "economic_status", "health_status", "mental_health_status", "emotional_state",
]

ACTIONS = [
    "ask_education_level", "ask_age", "ask_gender", "ask_marital_status",
    "ask_profession", "ask_economic_status", "ask_health_status",
    "ask_mental_health_status", "ask_emotional_state", "stop",
]

REPLY_TYPES = ["direct", "partial", "evasive", "off_topic"]

MENTAL_HIGH = {
    "severe depression", "depression", "chronic ptsd", "ptsd", "severe anxiety",
    "anxiety disorder", "generalized anxiety disorder", "panic", "self-harm",
    "suicidal", "trauma", "grief", "severe stress", "burnout",
    "adjustment disorder", "mild ptsd",
}
MENTAL_MED = {
    "mild depression", "moderate depression", "moderate anxiety", "anxiety",
    "mild anxiety", "high stress", "high stress levels", "stress",
    "social anxiety disorder", "loneliness", "low self-esteem", "confusion",
    "content with occasional sadness", "mild stress",
}
MENTAL_LOW = {"high spirits", "content", "acceptance", "stable", "resilient"}

ECON_STRUGGLING = {"struggling", "difficult", "low", "limited", "dependent"}
ECON_MODERATE   = {"moderate", "variable"}
ECON_STABLE     = {"stable", "comfortable", "secure", "high", "affluent", "wealthy"}

HEALTH_POOR = {"poor", "very poor", "chronic illness", "arthritic", "overweight"}
HEALTH_MOD  = {"fair", "average", "moderate"}
HEALTH_GOOD = {"good", "excellent", "improving"}

MAX_TURNS = 6
EMBED_DIM = 384  # all-MiniLM-L6-v2 output dimension

STATE_DIM = (
    len(KNOWN_ATTRS) +  #   9  presence flags
    3 +                 #   3  mental risk
    3 +                 #   3  econ risk
    3 +                 #   3  health risk
    1 +                 #   1  turn norm
    len(REPLY_TYPES) +  #   4  last reply
    len(ACTIONS) +      #  10  last action
    EMBED_DIM           # 384  MiniLM query embedding
)
# STATE_DIM = 417

# ── Singleton encoder ──────────────────────────────────────────────────────────
_encoder = None

def load_encoder():
    """Load MiniLM encoder once. Downloads ~90MB on first call."""
    global _encoder
    if _encoder is None:
        from sentence_transformers import SentenceTransformer
        _encoder = SentenceTransformer("all-MiniLM-L6-v2")
    return _encoder


@lru_cache(maxsize=1000)
def _embed_query(query: str) -> np.ndarray:
    """Embed query string. Result cached by string value."""
    enc = load_encoder()
    return enc.encode([query], convert_to_numpy=True)[0].astype(np.float32)


# ── Risk bucket helpers ────────────────────────────────────────────────────────

def _mental_risk_bucket(value: str) -> list:
    v = value.lower().strip()
    if any(kw in v for kw in MENTAL_HIGH): return [0.0, 0.0, 1.0]
    if any(kw in v for kw in MENTAL_MED):  return [0.0, 1.0, 0.0]
    if any(kw in v for kw in MENTAL_LOW):  return [1.0, 0.0, 0.0]
    return [0.0, 1.0, 0.0]

def _econ_risk_bucket(value: str) -> list:
    v = value.lower().strip()
    if any(kw in v for kw in ECON_STRUGGLING): return [0.0, 0.0, 1.0]
    if any(kw in v for kw in ECON_MODERATE):   return [0.0, 1.0, 0.0]
    if any(kw in v for kw in ECON_STABLE):     return [1.0, 0.0, 0.0]
    return [0.0, 1.0, 0.0]

def _health_risk_bucket(value: str) -> list:
    v = value.lower().strip()
    if any(kw in v for kw in HEALTH_POOR): return [0.0, 0.0, 1.0]
    if any(kw in v for kw in HEALTH_MOD):  return [0.0, 1.0, 0.0]
    if any(kw in v for kw in HEALTH_GOOD): return [1.0, 0.0, 0.0]
    return [0.0, 1.0, 0.0]


# ── Main encode function ───────────────────────────────────────────────────────

def encode(state: dict, vectorizer=None) -> np.ndarray:
    """
    Encode a ConversationEnv state dict into float32 array of shape (417,).
    vectorizer argument is ignored (backwards-compat only).
    """
    known = state.get("known_attributes", {})

    known_vec  = [1.0 if attr in known else 0.0 for attr in KNOWN_ATTRS]

    mental_vec = _mental_risk_bucket(known["mental_health_status"]) \
                 if "mental_health_status" in known else [0.0, 0.0, 0.0]

    econ_vec   = _econ_risk_bucket(known["economic_status"]) \
                 if "economic_status" in known else [0.0, 0.0, 0.0]

    health_vec = _health_risk_bucket(known["health_status"]) \
                 if "health_status" in known else [0.0, 0.0, 0.0]

    turn_norm  = [state.get("turn_count", 0) / MAX_TURNS]

    reply      = state.get("last_reply_type", None)
    reply_vec  = [1.0 if reply == rt else 0.0 for rt in REPLY_TYPES]

    last_action = state.get("last_action", None)
    action_vec  = [1.0 if last_action == a else 0.0 for a in ACTIONS]

    query     = state.get("query", "")
    query_vec = _embed_query(query).tolist() if query else [0.0] * EMBED_DIM

    vec = (known_vec + mental_vec + econ_vec + health_vec +
           turn_norm + reply_vec + action_vec + query_vec)
    return np.array(vec, dtype=np.float32)


# ── Backwards-compat shim ─────────────────────────────────────────────────────

def fit_vectorizer(queries: list):
    """
    Shim for train.py / evaluate.py compatibility.
    Pre-warms the MiniLM encoder and caches embeddings for all training queries.
    Returns None — encode() ignores the vectorizer argument.
    """
    print(f"Loading MiniLM-L6-v2 and caching {len(queries)} query embeddings...")
    load_encoder()
    unique = list(set(queries))
    for q in unique:
        _embed_query(q)
    print(f"  Cached {len(unique)} unique queries. STATE_DIM={STATE_DIM}")
    return None