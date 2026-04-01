import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
import torch
from env.data_loader import DataLoader
from env.conversation_env import ConversationEnv
from agent.dqn_agent import DQNAgent, ACTIONS
from agent.state_encoder import encode, fit_vectorizer, STATE_DIM

NUM_EPISODES = 15000
PRINT_EVERY = 1000
MODEL_SAVE_PATH = "agent/dqn_model.pt"
DATASET_PATH = "data/full_40_dataset.json"


def train():
    loader = DataLoader(DATASET_PATH)
    env = ConversationEnv(loader, max_turns=9)

    with open(DATASET_PATH) as f:
        all_queries = [s["query"] for s in json.load(f)]
    vectorizer = fit_vectorizer(all_queries)

    agent = DQNAgent(state_dim=STATE_DIM)

    episode_rewards = []
    episode_lengths = []

    for episode in range(1, NUM_EPISODES + 1):
        state_dict = env.reset()
        state_vec = encode(state_dict, vectorizer)
        valid_actions = [a for a in state_dict.get("valid_actions", ACTIONS) if a in ACTIONS]
        total_reward = 0
        steps = 0
        done = False

        while not done:
            action = agent.select_action(state_vec, valid_actions)
            next_state_dict, reward, done, info = env.step(action)
            next_state_vec = encode(next_state_dict, vectorizer)
            next_valid = [a for a in next_state_dict.get("valid_actions", ACTIONS) if a in ACTIONS]

            agent.store(state_vec, action, reward, next_state_vec, done)
            agent.learn()

            state_vec = next_state_vec
            valid_actions = next_valid
            total_reward += reward
            steps += 1

        episode_rewards.append(total_reward)
        episode_lengths.append(steps)

        if episode % PRINT_EVERY == 0:
            avg_reward = sum(episode_rewards[-PRINT_EVERY:]) / PRINT_EVERY
            avg_steps = sum(episode_lengths[-PRINT_EVERY:]) / PRINT_EVERY
            print(
                f"Episode {episode:5d} | "
                f"Avg Reward: {avg_reward:6.3f} | "
                f"Avg Steps: {avg_steps:5.2f} | "
                f"Epsilon: {agent.epsilon:.3f}"
            )

    torch.save(agent.policy_net.state_dict(), MODEL_SAVE_PATH)
    print(f"\nModel saved to {MODEL_SAVE_PATH}")
    return agent, episode_rewards, episode_lengths


if __name__ == "__main__":
    train()
