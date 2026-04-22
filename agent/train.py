"""
train.py
--------
Training loop for the DQN attribute-acquisition agent.

Usage:
    python train.py                    # full training run
    python train.py --episodes 5000    # quick debug run
    python train.py --resume           # resume from latest checkpoint
"""

import argparse
import json
import os
import random
import time
from collections import defaultdict

import numpy as np
import torch

from env.data_loader import DataLoader
from env.conversation_env import ConversationEnv
from agent.dqn_agent import DQNAgent, ACTIONS
from agent.state_encoder import encode, fit_vectorizer, STATE_DIM

# ── Config ────────────────────────────────────────────────────────────────────
DATASET_PATH    = "data/full_dataset.json"
MODEL_DIR       = "models1"
MODEL_SAVE_PATH = os.path.join(MODEL_DIR, "dqn_model.pt")
LOG_PATH        = os.path.join(MODEL_DIR, "train_log.json")

NUM_EPISODES    = 15000
MAX_TURNS       = 6      # CHANGE 2: was 9 — forces agent to be efficient, not just exhaustive
PRINT_EVERY     = 250    # CHANGE 5: was 1000 — finer visibility especially early in training
CHECKPOINT_EVERY = 2500  # CHANGE 7: save intermediate checkpoints
VAL_EVERY       = 500    # how often to run validation episodes
NUM_VAL_EPISODES = 50    # how many val episodes per evaluation
VAL_SPLIT       = 0.2    # fraction of profiles held out for validation

# Safety-critical attributes — tracked separately in metrics
SAFETY_CRITICAL = {"mental_health_status", "emotional_state", "health_status"}


# ── Data splitting ────────────────────────────────────────────────────────────

def stratified_split(data: list, val_fraction: float = VAL_SPLIT, seed: int = 42):
    """
    CHANGE 6: Stratified 80/20 train/val split by scenario.
    Without this all 500 profiles go into training and we can't detect
    overfitting or measure generalisation across unseen user profiles.
    """
    random.seed(seed)
    by_scenario = defaultdict(list)
    for item in data:
        by_scenario[item["scenario"]].append(item)

    train_data, val_data = [], []
    for scenario, items in by_scenario.items():
        random.shuffle(items)
        n_val = max(1, int(len(items) * val_fraction))
        val_data.extend(items[:n_val])
        train_data.extend(items[n_val:])

    random.shuffle(train_data)
    random.shuffle(val_data)
    return train_data, val_data


# ── Metrics helpers ───────────────────────────────────────────────────────────

def episode_metrics(env: ConversationEnv, final_state: dict) -> dict:
    """
    CHANGE 4: Compute per-episode metrics beyond just total reward.
    These are what matter for evaluating the proposal's goals:
      - coverage: fraction of valid attributes collected
      - critical_coverage: fraction of safety-critical attrs collected
      - turns: number of questions asked
      - stopped_early: did the agent choose stop before max_turns?
    """
    known      = final_state["known_attributes"]
    gt         = env.ground_truth
    invalid    = env.invalid_values
    valid_gt   = {k: v for k, v in gt.items() if v not in invalid}
    total      = len(valid_gt)

    coverage   = len([k for k in known if k in valid_gt]) / total if total > 0 else 0.0

    critical_in_gt = [a for a in SAFETY_CRITICAL if a in valid_gt]
    critical_known = [a for a in critical_in_gt if a in known]
    critical_cov   = (
        len(critical_known) / len(critical_in_gt) if critical_in_gt else 0.0
    )

    return {
        "coverage":          coverage,
        "critical_coverage": critical_cov,
        "turns":             final_state["turn_count"],
        "stopped_early":     final_state["turn_count"] < MAX_TURNS,
        "n_known":           len(known),
    }


def run_validation(env, agent, num_episodes: int) -> dict:
    """
    Run the agent greedily (epsilon=0) on validation profiles and return
    aggregate metrics. Called every VAL_EVERY training episodes.
    """
    saved_eps = agent.epsilon
    agent.epsilon = 0.0   # full greedy for validation

    rewards, coverages, critical_covs, turns_list = [], [], [], []

    for _ in range(num_episodes):
        state_dict  = env.reset()
        state_vec   = encode(state_dict)
        valid       = state_dict.get("valid_actions", ACTIONS)
        total_reward = 0.0
        done        = False

        while not done:
            action                          = agent.select_action(state_vec, valid)
            next_state_dict, reward, done, _ = env.step(action)
            next_state_vec                  = encode(next_state_dict)
            state_vec = next_state_vec
            valid     = next_state_dict.get("valid_actions", ACTIONS)
            total_reward += reward

        m = episode_metrics(env, next_state_dict)
        rewards.append(total_reward)
        coverages.append(m["coverage"])
        critical_covs.append(m["critical_coverage"])
        turns_list.append(m["turns"])

    agent.epsilon = saved_eps  # restore

    return {
        "val_reward":   float(np.mean(rewards)),
        "val_coverage": float(np.mean(coverages)),
        "val_critical": float(np.mean(critical_covs)),
        "val_turns":    float(np.mean(turns_list)),
    }


# ── Training loop ─────────────────────────────────────────────────────────────

def train(num_episodes: int = NUM_EPISODES, resume: bool = False):
    # ── Setup ──────────────────────────────────────────────────────────────
    os.makedirs(MODEL_DIR, exist_ok=True)

    with open(DATASET_PATH, encoding="utf-8") as f:
        all_data = json.load(f)

    # CHANGE 6: stratified train/val split
    train_data, val_data = stratified_split(all_data)
    print(f"Dataset      : {len(all_data)} profiles")
    print(f"Train / Val  : {len(train_data)} / {len(val_data)}")

    # Pre-warm MiniLM encoder and cache all query embeddings (called once)
    # fit_vectorizer() loads the model + caches embeddings via lru_cache
    all_queries = [s["query"] for s in all_data]
    fit_vectorizer(all_queries)

    train_loader = DataLoader(DATASET_PATH)
    train_loader.data = train_data

    val_loader   = DataLoader(DATASET_PATH)
    val_loader.data = val_data

    train_env = ConversationEnv(train_loader, max_turns=MAX_TURNS)
    val_env   = ConversationEnv(val_loader,   max_turns=MAX_TURNS)

    agent = DQNAgent(state_dim=STATE_DIM)

    # CHANGE 7: optionally resume from checkpoint
    start_episode = 1
    if resume and os.path.exists(MODEL_SAVE_PATH):
        agent.load(MODEL_SAVE_PATH)
        print(f"Resumed from {MODEL_SAVE_PATH}")
    
    # ── Tracking ────────────────────────────────────────────────────────────
    log = {
        "train_rewards":   [],
        "train_coverage":  [],
        "train_critical":  [],
        "train_turns":     [],
        "train_losses":    [],
        "val_checkpoints": [],   # list of {episode, val_reward, val_coverage, ...}
    }

    window_rewards   = []
    window_coverages = []
    window_critical  = []
    window_turns     = []
    window_losses    = []

    best_val_reward = -np.inf
    t_start = time.time()

    print(f"\nStarting training — {num_episodes} episodes | STATE_DIM={STATE_DIM} | encoder=MiniLM-L6-v2")
    print(f"{'Episode':>8} | {'AvgReward':>9} | {'Coverage':>8} | {'Critical':>8} | {'AvgTurns':>8} | {'Loss':>8} | {'Epsilon':>7}")
    print("-" * 80)

    # ── Main loop ────────────────────────────────────────────────────────────
    for episode in range(start_episode, num_episodes + 1):

        state_dict  = train_env.reset()
        state_vec   = encode(state_dict)
        valid       = state_dict.get("valid_actions", ACTIONS)
        total_reward = 0.0
        ep_losses   = []
        done        = False

        while not done:
            action                              = agent.select_action(state_vec, valid)
            next_state_dict, reward, done, info = train_env.step(action)
            next_state_vec                      = encode(next_state_dict)
            next_valid                          = next_state_dict.get("valid_actions", ACTIONS)

            agent.store(state_vec, action, reward, next_state_vec, done)

            # CHANGE 3: capture and track loss
            loss = agent.learn()
            if loss is not None:
                ep_losses.append(loss)

            state_vec    = next_state_vec
            valid        = next_valid
            total_reward += reward

        # Per-episode metrics (CHANGE 4)
        m = episode_metrics(train_env, next_state_dict)

        window_rewards.append(total_reward)
        window_coverages.append(m["coverage"])
        window_critical.append(m["critical_coverage"])
        window_turns.append(m["turns"])
        if ep_losses:
            window_losses.append(np.mean(ep_losses))

        log["train_rewards"].append(total_reward)
        log["train_coverage"].append(m["coverage"])
        log["train_critical"].append(m["critical_coverage"])
        log["train_turns"].append(m["turns"])
        if ep_losses:
            log["train_losses"].append(np.mean(ep_losses))

        # ── Print progress every PRINT_EVERY episodes ───────────────────────
        if episode % PRINT_EVERY == 0:
            avg_r   = np.mean(window_rewards)
            avg_cov = np.mean(window_coverages)
            avg_crit= np.mean(window_critical)
            avg_t   = np.mean(window_turns)
            avg_l   = np.mean(window_losses) if window_losses else 0.0

            print(
                f"{episode:>8d} | "
                f"{avg_r:>9.3f} | "
                f"{avg_cov:>8.3f} | "
                f"{avg_crit:>8.3f} | "
                f"{avg_t:>8.2f} | "
                f"{avg_l:>8.5f} | "
                f"{agent.epsilon:>7.4f}"
            )

            # ── Attribute collection snapshot ─────────────────────────────
            # Shows what the last episode actually collected so you can see
            # whether the agent is prioritising safety-critical attrs.
            known   = next_state_dict["known_attributes"]
            gt      = train_env.ground_truth
            invalid = train_env.invalid_values
            # Mark each attr: ✓ collected | ✗ missed | - not in ground truth
            ATTR_ORDER = [
                "mental_health_status", "emotional_state", "health_status",  # critical first
                "economic_status", "age", "marital_status",                   # medium
                "education_level", "gender", "profession",                    # easy
            ]
            parts = []
            for attr in ATTR_ORDER:
                short = attr.replace("_status","").replace("_level","").replace("_state","")
                short = short[:6]  # truncate for compact display
                if attr not in gt or gt[attr] in invalid:
                    parts.append(f"{short}:-")
                elif attr in known:
                    parts.append(f"{short}:✓")
                else:
                    parts.append(f"{short}:✗")
            critical_collected = sum(1 for a in SAFETY_CRITICAL if a in known)
            print(f"          └─ last ep attrs [{'  '.join(parts)}] "
                  f"critical={critical_collected}/3 turns={next_state_dict['turn_count']}")

            # Reset windows
            window_rewards   = []
            window_coverages = []
            window_critical  = []
            window_turns     = []
            window_losses    = []

        # ── Validation checkpoint (CHANGE 6+7) ──────────────────────────────
        if episode % VAL_EVERY == 0:
            val_metrics = run_validation(val_env, agent, NUM_VAL_EPISODES)
            val_metrics["episode"] = episode
            log["val_checkpoints"].append(val_metrics)

            # Save best model
            if val_metrics["val_reward"] > best_val_reward:
                best_val_reward = val_metrics["val_reward"]
                agent.save(MODEL_SAVE_PATH.replace(".pt", "_best.pt"))

            print(
                f"  [VAL ep {episode}] "
                f"reward={val_metrics['val_reward']:.3f} | "
                f"coverage={val_metrics['val_coverage']:.3f} | "
                f"critical={val_metrics['val_critical']:.3f} | "
                f"turns={val_metrics['val_turns']:.2f}"
                f"  {'← best' if val_metrics['val_reward'] == best_val_reward else ''}"
            )

        # ── Periodic checkpoint (CHANGE 7) ──────────────────────────────────
        if episode % CHECKPOINT_EVERY == 0:
            ckpt_path = MODEL_SAVE_PATH.replace(".pt", f"_ep{episode}.pt")
            agent.save(ckpt_path)

    # ── Final save ───────────────────────────────────────────────────────────
    agent.save(MODEL_SAVE_PATH)
    with open(LOG_PATH, "w") as f:
        json.dump(log, f, indent=2)

    elapsed = time.time() - t_start
    print(f"\nTraining complete in {elapsed/60:.1f} min")
    print(f"Final model  → {MODEL_SAVE_PATH}")
    print(f"Best model   → {MODEL_SAVE_PATH.replace('.pt', '_best.pt')}")
    print(f"Training log → {LOG_PATH}")

    return agent, log


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=NUM_EPISODES,
                        help="Number of training episodes")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from existing model checkpoint")
    args = parser.parse_args()

    train(num_episodes=args.episodes, resume=args.resume)