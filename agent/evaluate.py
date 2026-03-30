import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from env.data_loader import DataLoader
from env.conversation_env import ConversationEnv
from agent.dqn_agent import DQNAgent, ACTIONS
from agent.state_encoder import encode, STATE_DIM

MODEL_PATH = "agent/dqn_model.pt"
NUM_EVAL_EPISODES = 200


def evaluate(verbose=False):
    loader = DataLoader("data/grouped_conversations.json")
    env = ConversationEnv(loader)

    agent = DQNAgent(state_dim=STATE_DIM, epsilon_start=0.0, epsilon_end=0.0)
    agent.policy_net.load_state_dict(torch.load(MODEL_PATH, map_location=agent.device))
    agent.policy_net.eval()

    total_rewards = []
    total_steps = []
    total_coverage = []
    stop_correct = 0

    for episode in range(NUM_EVAL_EPISODES):
        state_dict = env.reset()
        state_vec = encode(state_dict)
        total_reward = 0
        steps = 0
        done = False

        if verbose:
            print(f"\n=== Episode {episode + 1} ===")
            print(f"Query: {state_dict['query']}")

        while not done:
            with torch.no_grad():
                t = torch.FloatTensor(state_vec).unsqueeze(0).to(agent.device)
                q_vals = agent.policy_net(t)
                idx = q_vals.argmax(dim=1).item()
            action = ACTIONS[idx]

            next_state_dict, reward, done, info = env.step(action)
            next_state_vec = encode(next_state_dict)

            if verbose:
                print(f"  Action: {action} | Reply type: {info.get('reply_type')} | Reward: {reward:.2f}")

            state_vec = next_state_vec
            total_reward += reward
            steps += 1

        known = next_state_dict["known_attributes"]
        ground_truth = env.ground_truth
        valid_gt = {k: v for k, v in ground_truth.items() if v != "not given"}
        coverage = len([k for k in known if k in valid_gt]) / max(len(valid_gt), 1)

        if coverage >= 0.75:
            stop_correct += 1

        total_rewards.append(total_reward)
        total_steps.append(steps)
        total_coverage.append(coverage)

        if verbose:
            print(f"  Known: {list(known.keys())} | Coverage: {coverage:.2f} | Total reward: {total_reward:.2f}")

    print(f"\n{'='*40}")
    print(f"Eval over {NUM_EVAL_EPISODES} episodes")
    print(f"  Avg Reward   : {sum(total_rewards) / NUM_EVAL_EPISODES:.3f}")
    print(f"  Avg Steps    : {sum(total_steps) / NUM_EVAL_EPISODES:.2f}")
    print(f"  Avg Coverage : {sum(total_coverage) / NUM_EVAL_EPISODES:.2%}")
    print(f"  Success Rate : {stop_correct / NUM_EVAL_EPISODES:.2%}  (coverage >= 75%)")
    print(f"{'='*40}")


if __name__ == "__main__":
    evaluate(verbose=True)
