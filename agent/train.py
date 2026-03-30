import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from env.data_loader import DataLoader
from env.conversation_env import ConversationEnv
from agent.dqn_agent import DQNAgent
from agent.state_encoder import encode, STATE_DIM

NUM_EPISODES = 5000
PRINT_EVERY = 500
MODEL_SAVE_PATH = "agent/dqn_model.pt"


def train():
    loader = DataLoader("data/grouped_conversations.json")
    env = ConversationEnv(loader)
    agent = DQNAgent(state_dim=STATE_DIM)

    episode_rewards = []
    episode_lengths = []

    for episode in range(1, NUM_EPISODES + 1):
        state_dict = env.reset()
        state_vec = encode(state_dict)
        total_reward = 0
        steps = 0
        done = False

        while not done:
            action = agent.select_action(state_vec)
            next_state_dict, reward, done, info = env.step(action)
            next_state_vec = encode(next_state_dict)

            agent.store(state_vec, action, reward, next_state_vec, done)
            agent.learn()

            state_vec = next_state_vec
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
