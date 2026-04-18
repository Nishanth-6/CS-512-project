import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from agent.replay_buffer import ReplayBuffer

ACTIONS = [
    "ask_education_level",
    "ask_age",
    "ask_gender",
    "ask_marital_status",
    "ask_profession",
    "ask_economic_status",
    "ask_health_status",
    "ask_mental_health_status",
    "ask_emotional_state",
    "stop",
]
ACTION_TO_IDX = {a: i for i, a in enumerate(ACTIONS)}
N_ACTIONS = len(ACTIONS)


class QNetwork(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, N_ACTIONS),
        )

    def forward(self, x):
        return self.net(x)


class DQNAgent:
    def __init__(
        self,
        state_dim,
        lr=1e-3,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay=500,
        batch_size=32,
        target_update_freq=50,
        buffer_capacity=10000,
    ):
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.steps_done = 0

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.policy_net = QNetwork(state_dim).to(self.device)
        self.target_net = QNetwork(state_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.buffer = ReplayBuffer(buffer_capacity)
        self.loss_fn = nn.MSELoss()

    def select_action(self, state_vec: np.ndarray, valid_actions: list = None) -> str:
        self.epsilon = self.epsilon_end + (self.epsilon - self.epsilon_end) * np.exp(
            -self.steps_done / self.epsilon_decay
        )
        self.steps_done += 1

        allowed = valid_actions if valid_actions else ACTIONS
        allowed = [a for a in allowed if a in ACTION_TO_IDX]

        if np.random.rand() < self.epsilon:
            return np.random.choice(allowed)

        with torch.no_grad():
            t = torch.FloatTensor(state_vec).unsqueeze(0).to(self.device)
            q_vals = self.policy_net(t).squeeze(0)

        allowed_indices = [ACTION_TO_IDX[a] for a in allowed]
        best_idx = max(allowed_indices, key=lambda i: q_vals[i].item())
        return ACTIONS[best_idx]

    def store(self, state, action, reward, next_state, done):
        self.buffer.push(state, ACTION_TO_IDX[action], reward, next_state, done)

    def learn(self):
        if len(self.buffer) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        q_values = self.policy_net(states).gather(1, actions)

        with torch.no_grad():
            max_next_q = self.target_net(next_states).max(dim=1, keepdim=True)[0]
            targets = rewards + self.gamma * max_next_q * (1 - dones)

        loss = self.loss_fn(q_values, targets)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.steps_done % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        return loss.item()
