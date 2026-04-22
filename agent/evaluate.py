"""
evaluate.py
-----------
Evaluates the trained DQN agent against four baselines and reports
the metrics defined in the project proposal.

Baselines:
  1. no_context   — stop immediately, collect 0 attributes
  2. random_3     — ask 3 randomly chosen attributes, then stop
  3. top_3        — always ask emotional_state → mental_health → health_status
  4. raise_like   — fixed query-conditional ordering (no response-adaptation)
  5. dqn          — trained DQN policy (ours)

Metrics (per proposal evaluation protocol):
  - coverage          : fraction of valid attributes collected
  - critical_coverage : fraction of {mental_health, emotional_state, health} collected
  - avg_turns         : average number of questions asked
  - efficiency        : coverage / turns  (safety gain per question)
  - intentional_stop  : agent chose stop before hitting max_turns (%)

Usage:
  python evaluate.py                      # full eval, all baselines
  python evaluate.py --episodes 100       # quick run
  python evaluate.py --verbose dqn        # trace a single baseline
  python evaluate.py --model models/dqn_model_best.pt
"""

import argparse
import json
import os
import random
from collections import defaultdict

import numpy as np

from env.data_loader import DataLoader
from env.conversation_env import ConversationEnv
from agent.dqn_agent import DQNAgent, ACTIONS
from agent.state_encoder import encode, fit_vectorizer, STATE_DIM
from agent.train import stratified_split   # reuse the same split logic

# ── Config ────────────────────────────────────────────────────────────────────
DATASET_PATH    = "data/full_dataset.json"
MODEL_PATH      = "models/dqn_model_best.pt"
MAX_TURNS       = 6
NUM_EPISODES    = 300   # per baseline — enough for reliable stats
N_SEEDS         = 3     # repeat each baseline with N random seeds for ± std
INVALID_VALUES  = {"not given", "unknown", "not specified", "", None}

# Safety-critical attributes (from proposal)
SAFETY_CRITICAL = {"mental_health_status", "emotional_state", "health_status"}

# Top-3 fixed ordering baseline
TOP_3_ORDER = [
    "ask_emotional_state",
    "ask_mental_health_status",
    "ask_health_status",
]

# RAISE-like: fixed ordering per scenario risk category (no response-adaptation)
# Maps scenario name → pre-defined question order.
# Determined offline by scenario type — does NOT change based on user answers.
RAISE_ORDER_BY_SCENARIO = {
    # Mental-health-primary scenarios
    "Addiction Recovery":              ["ask_mental_health_status", "ask_emotional_state", "ask_health_status", "ask_age", "ask_economic_status"],
    "Anxiety Attack":                  ["ask_mental_health_status", "ask_emotional_state", "ask_age", "ask_health_status", "ask_marital_status"],
    "Burnout":                         ["ask_mental_health_status", "ask_emotional_state", "ask_profession", "ask_age", "ask_economic_status"],
    "Chronic Illness Diagnosis":       ["ask_health_status", "ask_mental_health_status", "ask_emotional_state", "ask_age", "ask_economic_status"],
    "Chronic Pain":                    ["ask_health_status", "ask_mental_health_status", "ask_age", "ask_economic_status", "ask_profession"],
    "Depression Episode":              ["ask_mental_health_status", "ask_emotional_state", "ask_health_status", "ask_marital_status", "ask_age"],
    "Eating Disorder":                 ["ask_mental_health_status", "ask_health_status", "ask_emotional_state", "ask_age", "ask_gender"],
    "Identity Crisis":                 ["ask_emotional_state", "ask_mental_health_status", "ask_age", "ask_marital_status", "ask_profession"],
    "PTSD Trigger":                    ["ask_mental_health_status", "ask_emotional_state", "ask_health_status", "ask_age", "ask_marital_status"],
    "Panic Attack":                    ["ask_mental_health_status", "ask_emotional_state", "ask_health_status", "ask_age", "ask_marital_status"],
    "Sleep Disorder":                  ["ask_health_status", "ask_mental_health_status", "ask_emotional_state", "ask_profession", "ask_age"],
    # Financial-primary scenarios
    "Business Loan Default":           ["ask_economic_status", "ask_profession", "ask_mental_health_status", "ask_age", "ask_marital_status"],
    "Car Loan Default":                ["ask_economic_status", "ask_profession", "ask_age", "ask_mental_health_status", "ask_marital_status"],
    "Gambling Debts":                  ["ask_economic_status", "ask_mental_health_status", "ask_marital_status", "ask_age", "ask_emotional_state"],
    "Identity Theft Impact":           ["ask_economic_status", "ask_mental_health_status", "ask_age", "ask_profession", "ask_marital_status"],
    "Industry Blacklisting":           ["ask_profession", "ask_economic_status", "ask_mental_health_status", "ask_age", "ask_education_level"],
    "Loan Shark Threats":              ["ask_economic_status", "ask_mental_health_status", "ask_marital_status", "ask_age", "ask_profession"],
    # Career-primary scenarios
    "Conference Presentation Failure": ["ask_profession", "ask_education_level", "ask_emotional_state", "ask_mental_health_status", "ask_age"],
    "Impostor Syndrome":               ["ask_profession", "ask_mental_health_status", "ask_emotional_state", "ask_education_level", "ask_age"],
    "Leadership Challenge":            ["ask_profession", "ask_education_level", "ask_age", "ask_emotional_state", "ask_economic_status"],
    "Mentor Relationship Breakdown":   ["ask_profession", "ask_mental_health_status", "ask_age", "ask_emotional_state", "ask_education_level"],
    "Office Politics Crisis":          ["ask_profession", "ask_mental_health_status", "ask_emotional_state", "ask_age", "ask_marital_status"],
    "Professional Reputation Damage":  ["ask_profession", "ask_mental_health_status", "ask_economic_status", "ask_age", "ask_emotional_state"],
    "Team Project Failure":            ["ask_profession", "ask_mental_health_status", "ask_emotional_state", "ask_age", "ask_education_level"],
    # Social/rejection scenarios
    "Club/Organization Expulsion":     ["ask_emotional_state", "ask_mental_health_status", "ask_age", "ask_marital_status", "ask_profession"],
    "Community Rejection":             ["ask_emotional_state", "ask_mental_health_status", "ask_age", "ask_marital_status", "ask_economic_status"],
    "Cultural Group Ostracism":        ["ask_emotional_state", "ask_mental_health_status", "ask_age", "ask_marital_status", "ask_gender"],
    "Friend Group Exclusion":          ["ask_emotional_state", "ask_mental_health_status", "ask_age", "ask_marital_status", "ask_gender"],
    "Neighbor Conflict":               ["ask_emotional_state", "ask_marital_status", "ask_age", "ask_mental_health_status", "ask_economic_status"],
    "Religious Community Exclusion":   ["ask_emotional_state", "ask_mental_health_status", "ask_marital_status", "ask_age", "ask_gender"],
    "Social Event Disaster":           ["ask_emotional_state", "ask_mental_health_status", "ask_age", "ask_marital_status", "ask_gender"],
    "Social Media Crisis":             ["ask_emotional_state", "ask_mental_health_status", "ask_profession", "ask_age", "ask_economic_status"],
    "Sports Team Rejection":           ["ask_emotional_state", "ask_mental_health_status", "ask_age", "ask_health_status", "ask_gender"],
    "Volunteer Organization Conflict": ["ask_emotional_state", "ask_marital_status", "ask_age", "ask_mental_health_status", "ask_profession"],
    "Workplace Ostracism":             ["ask_emotional_state", "ask_mental_health_status", "ask_profession", "ask_age", "ask_marital_status"],
    # Physical health scenarios
    "Cosmetic Surgery Gone Wrong":     ["ask_health_status", "ask_mental_health_status", "ask_emotional_state", "ask_age", "ask_economic_status"],
    "Sports Career-Ending Injury":     ["ask_health_status", "ask_mental_health_status", "ask_emotional_state", "ask_profession", "ask_age"],
    "Sudden Disability":               ["ask_health_status", "ask_mental_health_status", "ask_emotional_state", "ask_economic_status", "ask_age"],
    "Terminal Illness":                ["ask_health_status", "ask_mental_health_status", "ask_emotional_state", "ask_marital_status", "ask_age"],
    "Weight Crisis":                   ["ask_health_status", "ask_mental_health_status", "ask_emotional_state", "ask_age", "ask_economic_status"],
    # Default
    "Client Relationship Crisis":      ["ask_profession", "ask_emotional_state", "ask_mental_health_status", "ask_age", "ask_economic_status"],
    "Professional Association Rejection": ["ask_profession", "ask_mental_health_status", "ask_emotional_state", "ask_education_level", "ask_age"],
}
RAISE_DEFAULT_ORDER = [
    "ask_emotional_state", "ask_mental_health_status", "ask_health_status",
    "ask_age", "ask_economic_status",
]


# ── Metric helpers ────────────────────────────────────────────────────────────

def compute_metrics(env: ConversationEnv, final_state: dict, max_turns: int) -> dict:
    """Compute all evaluation metrics for one completed episode."""
    known   = final_state["known_attributes"]
    gt      = env.ground_truth
    valid_gt = {k: v for k, v in gt.items() if v not in INVALID_VALUES}
    total   = len(valid_gt)

    n_known    = len([k for k in known if k in valid_gt])
    coverage   = n_known / total if total > 0 else 0.0

    critical_in_gt = [a for a in SAFETY_CRITICAL if a in valid_gt]
    critical_known = [a for a in critical_in_gt if a in known]
    critical_cov   = len(critical_known) / len(critical_in_gt) if critical_in_gt else 0.0

    turns   = final_state["turn_count"]
    # Efficiency: coverage gained per question asked (avoid div-by-zero)
    efficiency = coverage / turns if turns > 0 else 0.0
    # Intentional stop: agent chose stop BEFORE hitting the hard limit
    intentional_stop = int(final_state.get("last_action") == "stop")
    
    return {
        "coverage":          coverage,
        "critical_coverage": critical_cov,
        "turns":             turns,
        "efficiency":        efficiency,
        "intentional_stop":  intentional_stop,
    }


def aggregate(results: list) -> dict:
    """Mean ± std over a list of metric dicts."""
    keys = results[0].keys()
    return {
        k: {"mean": float(np.mean([r[k] for r in results])),
            "std":  float(np.std( [r[k] for r in results]))}
        for k in keys
    }


# ── Baseline runners ──────────────────────────────────────────────────────────

def run_no_context(env, num_episodes):
    """Baseline 1: stop immediately, collect nothing."""
    results = []
    for _ in range(num_episodes):
        state = env.reset()
        _, reward, done, _ = env.step("stop")
        m = compute_metrics(env, env._get_state(), MAX_TURNS)
        m["reward"] = reward
        results.append(m)
    return results


def run_random_k(env, num_episodes, k=3):
    """Baseline 2: ask K randomly chosen attributes then stop."""
    results = []
    for _ in range(num_episodes):
        state       = env.reset()
        total_reward = 0.0
        done        = False

        # Pick k unique valid attrs to ask (random order)
        valid = [a for a in state["valid_actions"] if a != "stop"]
        chosen = random.sample(valid, min(k, len(valid)))

        for action in chosen:
            if done:
                break
            state, reward, done, _ = env.step(action)
            total_reward += reward

        if not done:
            state, reward, done, _ = env.step("stop")
            total_reward += reward

        m = compute_metrics(env, state, MAX_TURNS)
        m["reward"] = total_reward
        results.append(m)
    return results


def run_top_3(env, num_episodes):
    """Baseline 3: always ask emotional_state → mental_health → health_status then stop."""
    results = []
    for _ in range(num_episodes):
        state        = env.reset()
        total_reward = 0.0
        done         = False

        for action in TOP_3_ORDER:
            if done:
                break
            if action in state["valid_actions"]:
                state, reward, done, _ = env.step(action)
                total_reward += reward

        if not done:
            state, reward, done, _ = env.step("stop")
            total_reward += reward

        m = compute_metrics(env, state, MAX_TURNS)
        m["reward"] = total_reward
        results.append(m)
    return results


def run_raise_like(env, num_episodes):
    """
    Baseline 4: RAISE-like — fixed query-conditional ordering per scenario.
    The ordering is determined before the episode starts and does NOT change
    based on what the user actually says. This directly simulates RAISE's
    limitation: it adapts to the query but not to the user's responses.
    """
    results = []
    for _ in range(num_episodes):
        state        = env.reset()
        total_reward = 0.0
        done         = False

        # Get fixed ordering for this scenario (determined at episode start)
        scenario = state.get("scenario", "")
        order    = RAISE_ORDER_BY_SCENARIO.get(scenario, RAISE_DEFAULT_ORDER)

        for action in order:
            if done:
                break
            if action in state["valid_actions"]:
                state, reward, done, _ = env.step(action)
                total_reward += reward

        if not done:
            state, reward, done, _ = env.step("stop")
            total_reward += reward

        m = compute_metrics(env, state, MAX_TURNS)
        m["reward"] = total_reward
        results.append(m)
    return results


def run_dqn(env, agent, vectorizer, num_episodes):
    """Baseline 5 (ours): trained DQN policy, fully greedy."""
    results = []
    for _ in range(num_episodes):
        state_dict   = env.reset()
        state_vec    = encode(state_dict, vectorizer)
        valid        = state_dict["valid_actions"]
        total_reward = 0.0
        done         = False

        while not done:
            action                              = agent.select_action(state_vec, valid)
            next_state_dict, reward, done, info = env.step(action)
            next_state_vec                      = encode(next_state_dict, vectorizer)
            state_vec    = next_state_vec
            valid        = next_state_dict["valid_actions"]
            total_reward += reward

        m = compute_metrics(env, next_state_dict, MAX_TURNS)
        m["reward"] = total_reward
        results.append(m)
    return results


# ── Per-scenario breakdown ────────────────────────────────────────────────────

def run_dqn_by_scenario(env, agent, vectorizer, num_episodes):
    """Run DQN and collect metrics grouped by scenario for breakdown table."""
    by_scenario = defaultdict(list)
    for _ in range(num_episodes):
        state_dict   = env.reset()
        scenario     = state_dict["scenario"]
        state_vec    = encode(state_dict, vectorizer)
        valid        = state_dict["valid_actions"]
        total_reward = 0.0
        done         = False

        while not done:
            action                              = agent.select_action(state_vec, valid)
            next_state_dict, reward, done, info = env.step(action)
            next_state_vec                      = encode(next_state_dict, vectorizer)
            state_vec    = next_state_vec
            valid        = next_state_dict["valid_actions"]
            total_reward += reward

        m = compute_metrics(env, next_state_dict, MAX_TURNS)
        m["reward"] = total_reward
        by_scenario[scenario].append(m)

    return {
        sc: {k: float(np.mean([r[k] for r in recs])) for k in recs[0]}
        for sc, recs in by_scenario.items()
    }


# ── Verbose episode trace ─────────────────────────────────────────────────────

def trace_episode(env, agent, vectorizer, baseline: str = "dqn"):
    """
    Run a single episode with full trace — shows what the agent asks,
    what the user replies, and which values get revealed.
    Good for qualitative inspection of agent behaviour.
    """
    state_dict   = env.reset()
    state_vec    = encode(state_dict, vectorizer)
    valid        = state_dict["valid_actions"]
    total_reward = 0.0
    done         = False

    print(f"\n{'='*60}")
    print(f"QUERY    : {state_dict['query']}")
    print(f"SCENARIO : {state_dict['scenario']}")
    print(f"GT ATTRS : {json.dumps(env.ground_truth, indent=10)}")
    print(f"{'='*60}")

    step = 0
    while not done:
        step += 1
        if baseline == "dqn":
            action = agent.select_action(state_vec, valid)
        elif baseline == "top_3":
            remaining = [a for a in TOP_3_ORDER if a in valid]
            action = remaining[0] if remaining else "stop"
        elif baseline == "raise_like":
            order = RAISE_ORDER_BY_SCENARIO.get(state_dict["scenario"], RAISE_DEFAULT_ORDER)
            remaining = [a for a in order if a in valid]
            action = remaining[0] if remaining else "stop"
        else:
            action = random.choice(valid)

        next_state_dict, reward, done, info = env.step(action)
        next_state_vec = encode(next_state_dict, vectorizer)

        attr  = action.replace("ask_", "") if action != "stop" else "-"
        value = next_state_dict["known_attributes"].get(attr, "not revealed")

        print(f"\nStep {step}: {action}")
        print(f"  Reply type : {info.get('reply_type', '-')}")
        print(f"  Reply      : {str(info.get('reply', ''))[:80]}")
        print(f"  Value known: {value}")
        print(f"  Reward     : {reward:+.3f}")

        state_vec    = next_state_vec
        state_dict   = next_state_dict
        valid        = next_state_dict["valid_actions"]
        total_reward += reward

    m = compute_metrics(env, next_state_dict, MAX_TURNS)
    print(f"\n--- Episode Summary ---")
    print(f"  Turns              : {m['turns']}")
    print(f"  Coverage           : {m['coverage']:.2%}")
    print(f"  Critical coverage  : {m['critical_coverage']:.2%}")
    print(f"  Efficiency         : {m['efficiency']:.3f}")
    print(f"  Intentional stop   : {'yes' if m['intentional_stop'] else 'no (force-stopped)'}")
    print(f"  Total reward       : {total_reward:+.3f}")
    print(f"  Attrs collected    : {list(next_state_dict['known_attributes'].keys())}")


# ── Main evaluation ───────────────────────────────────────────────────────────

def evaluate(
    model_path:   str  = MODEL_PATH,
    num_episodes: int  = NUM_EPISODES,
    n_seeds:      int  = N_SEEDS,
    verbose:      str  = None,   # baseline name to trace, or None
    save_results: bool = True,
):
    # ── Load data — val split only (CHANGE 2) ──────────────────────────────
    with open(DATASET_PATH, encoding="utf-8") as f:
        all_data = json.load(f)

    _, val_data = stratified_split(all_data)
    print(f"Evaluating on {len(val_data)} held-out val profiles")
    print(f"Episodes per baseline: {num_episodes} × {n_seeds} seeds\n")

    # Fit vectorizer on training queries only (same split as train.py)
    train_data, _ = stratified_split(all_data)
    train_queries = [s["query"] for s in train_data]
    vectorizer    = fit_vectorizer(train_queries)

    # ── Load DQN agent ──────────────────────────────────────────────────────
    agent = DQNAgent(state_dim=STATE_DIM, epsilon_start=0.0, epsilon_end=0.0)
    agent.load(model_path)
    print(f"Loaded model: {model_path}\n")

    # ── Verbose trace mode ──────────────────────────────────────────────────
    if verbose:
        import time
        random.seed(int(time.time() * 1000) % 2**32)   # time-based seed → different query each run
        loader = DataLoader(DATASET_PATH)
        loader.data = val_data
        env = ConversationEnv(loader, max_turns=MAX_TURNS)
        trace_episode(env, agent, vectorizer, baseline=verbose)
        return

    # ── Run all baselines across seeds (CHANGE 5) ───────────────────────────
    baselines = {
        "no_context":  run_no_context,
        "random_3":    lambda env, n: run_random_k(env, n, k=3),
        "top_3":       run_top_3,
        "raise_like":  run_raise_like,
        "dqn":         lambda env, n: run_dqn(env, agent, vectorizer, n),
    }

    all_results = {}

    for name, runner in baselines.items():
        seed_aggs = []
        for seed in range(n_seeds):
            random.seed(seed)
            np.random.seed(seed)
            loader = DataLoader(DATASET_PATH)
            loader.data = val_data
            env    = ConversationEnv(loader, max_turns=MAX_TURNS)
            eps    = run_dqn(env, agent, vectorizer, num_episodes) if name == "dqn" \
                     else runner(env, num_episodes)
            seed_aggs.append(aggregate(eps))

        # Average mean/std across seeds
        keys = seed_aggs[0].keys()
        all_results[name] = {
            k: {
                "mean": float(np.mean([s[k]["mean"] for s in seed_aggs])),
                "std":  float(np.mean([s[k]["std"]  for s in seed_aggs])),
            }
            for k in keys
        }
        print(f"  ✓ {name}")

    # ── Per-scenario DQN breakdown ───────────────────────────────────────────
    loader = DataLoader(DATASET_PATH)
    loader.data = val_data
    env = ConversationEnv(loader, max_turns=MAX_TURNS)
    scenario_results = run_dqn_by_scenario(env, agent, vectorizer, num_episodes * n_seeds)

    # ── Print results table ──────────────────────────────────────────────────
    METRICS = ["reward", "coverage", "critical_coverage", "turns", "efficiency", "intentional_stop"]
    METRIC_LABELS = {
        "reward":           "Reward",
        "coverage":         "Coverage",
        "critical_coverage":"Critical Cov.",
        "turns":            "Avg Turns",
        "efficiency":       "Efficiency",
        "intentional_stop": "Intentional Stop",
    }

    col_w = 22
    header = f"{'Baseline':<16}" + "".join(f"{METRIC_LABELS[m]:>{col_w}}" for m in METRICS)
    print(f"\n{'='*len(header)}")
    print("EVALUATION RESULTS (mean ± std)")
    print(f"{'='*len(header)}")
    print(header)
    print("-" * len(header))

    for name, res in all_results.items():
        tag = " ← ours" if name == "dqn" else ""
        row = f"{name+tag:<16}"
        for m in METRICS:
            mean = res[m]["mean"]
            std  = res[m]["std"]
            if m == "intentional_stop":
                cell = f"{mean:.0%}±{std:.0%}"
            elif m == "turns":
                cell = f"{mean:.2f}±{std:.2f}"
            else:
                cell = f"{mean:.3f}±{std:.3f}"
            row += f"{cell:>{col_w}}"
        print(row)

    print(f"{'='*len(header)}")

    # ── Per-scenario breakdown for DQN ──────────────────────────────────────
    print(f"\nDQN PER-SCENARIO BREAKDOWN (coverage | critical | turns)")
    print(f"{'Scenario':<40} {'Coverage':>10} {'Critical':>10} {'Turns':>8}")
    print("-" * 72)
    for sc, m in sorted(scenario_results.items(), key=lambda x: -x[1]["coverage"]):
        print(f"{sc:<40} {m['coverage']:>10.3f} {m['critical_coverage']:>10.3f} {m['turns']:>8.2f}")

    # ── Delta vs RAISE-like ──────────────────────────────────────────────────
    print(f"\nDQN IMPROVEMENT OVER RAISE-LIKE BASELINE")
    dqn   = all_results["dqn"]
    raise_ = all_results["raise_like"]
    for m in ["coverage", "critical_coverage", "efficiency", "turns"]:
        delta = dqn[m]["mean"] - raise_[m]["mean"]
        sign  = "+" if delta >= 0 else ""
        arrow = "↑" if delta >= 0 else "↓"
        print(f"  {METRIC_LABELS[m]:<20}: {sign}{delta:.3f} {arrow}")

    # ── Save ─────────────────────────────────────────────────────────────────
    if save_results:
        os.makedirs("models", exist_ok=True)
        out = {
            "baselines":       all_results,
            "scenario_breakdown": scenario_results,
            "config": {
                "num_episodes": num_episodes,
                "n_seeds":      n_seeds,
                "max_turns":    MAX_TURNS,
                "model_path":   model_path,
                "val_profiles": len(val_data),
            }
        }
        path = "models/eval_results.json"
        with open(path, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nResults saved → {path}")

    return all_results, scenario_results


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",    type=str, default=MODEL_PATH)
    parser.add_argument("--episodes", type=int, default=NUM_EPISODES)
    parser.add_argument("--seeds",    type=int, default=N_SEEDS)
    parser.add_argument("--verbose",  type=str, default=None,
                        help="Trace one episode for: dqn | top_3 | raise_like | random_3")
    args = parser.parse_args()

    evaluate(
        model_path   = args.model,
        num_episodes = args.episodes,
        n_seeds      = args.seeds,
        verbose      = args.verbose,
    )