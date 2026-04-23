import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from agent.replay_buffer import ReplayBuffer

# ── Action space (must stay in sync with ConversationEnv and state_encoder) ───
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
N_ACTIONS = len(ACTIONS)  # 10


# ── Q-Network ─────────────────────────────────────────────────────────────────

class QNetwork(nn.Module):
    """
    3-layer MLP mapping state vectors to Q-values over all actions.

    Architecture: 417 → 256 → 128 → 10

    Why 256 first layer (was 128):
      State dim grew from 91 → 417 after replacing TF-IDF with MiniLM-L6-v2.
      The input now has two structurally different parts:
        [0:33]   — 33 sparse structured dims (binary flags, one-hots, turn count)
        [33:417] — 384 dense MiniLM embedding dims (rich semantic signal)

      With 417 → 128, the first layer compresses 3.3x in one step — too aggressive.
      It forces the network to simultaneously project 384 MiniLM dims AND process
      the sparse structured features through the same 128 bottleneck, which discards
      signal before the network can learn what matters.

      417 → 256 is a gentler 1.6x compression, giving the network room to:
        - Project MiniLM semantic directions relevant to attribute prioritisation
        - Handle sparse structured features independently
      Then 256 → 128 → 10 refines and produces Q-values.

    Why Dropout(0.2):
      With only 406 training profiles and 141K parameters, dropout prevents
      the network from over-relying on any single MiniLM direction and
      regularises the projection of dense embedding dims.
      Dropout is disabled automatically during eval (agent.policy_net.eval()).

    LayerNorm retained:
      MiniLM embeddings (~unit norm, range [-0.3, 0.3]) and binary flags
      (exactly 0 or 1) live on different scales. LayerNorm normalises
      activations at each layer without needing manual feature scaling.
    """
    def __init__(self, state_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(128, N_ACTIONS),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ── DQN Agent ─────────────────────────────────────────────────────────────────

class DQNAgent:
    """
    Double DQN agent with experience replay and epsilon-greedy exploration.

    Changes from previous version:
      1. QNetwork: 417→256→128→10 with Dropout(0.2). First layer widened from
         128→256 to handle 384-dim MiniLM input without aggressive bottleneck.
      2. Epsilon decay: 5000→20000 steps. At ~6 steps/episode, decay=5000
         meant epsilon hit 0.05 by episode ~850 — the agent went fully greedy
         before learning anything useful. decay=20000 keeps meaningful
         exploration for the first ~3300 episodes (~22% of training).
      3. Double DQN target to reduce Q-value overestimation.
      4. Gradient clipping (max_norm=10) + Huber loss for reward range [-1,3].
      5. Target network updated every 300 steps for stable TD targets.
      6. Buffer capacity 50000, batch size 64.
    """

    def __init__(
        self,
        state_dim: int,
        lr: float               = 1e-3,
        gamma: float            = 0.99,
        epsilon_start: float    = 1.0,
        epsilon_end: float      = 0.05,
        epsilon_decay: int      = 20000,   # was 5000 — see class docstring
        batch_size: int         = 64,
        target_update_freq: int = 300,
        buffer_capacity: int    = 50000,
    ):
        self.gamma              = gamma
        self.epsilon_start      = epsilon_start   # fixed reference for decay formula
        self.epsilon            = epsilon_start
        self.epsilon_end        = epsilon_end
        self.epsilon_decay      = epsilon_decay
        self.batch_size         = batch_size
        self.target_update_freq = target_update_freq
        self.steps_done         = 0

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.policy_net = QNetwork(state_dim).to(self.device)
        self.target_net = QNetwork(state_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.buffer    = ReplayBuffer(buffer_capacity)
        self.loss_fn   = nn.SmoothL1Loss()   # Huber loss: less sensitive to outlier rewards than MSE

    # ── Action selection ──────────────────────────────────────────────────────

    def select_action(self, state_vec: np.ndarray, valid_actions: list = None) -> str:
        """
        Epsilon-greedy action selection, restricted to valid_actions.
        valid_actions comes from ConversationEnv._get_state() and excludes
        already-known attributes, so the agent never wastes turns re-asking.
        """
        # Decay epsilon — uses epsilon_START (fixed) not self.epsilon (current).
        # Bug in previous versions: using self.epsilon compounds multiplicatively
        # each step, reaching the floor in ~1000 steps regardless of decay value.
        # Correct formula: one-shot from fixed start based on total steps done.
        self.epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * np.exp(
            -self.steps_done / self.epsilon_decay
        )
        self.steps_done += 1

        allowed = valid_actions if valid_actions else ACTIONS
        allowed = [a for a in allowed if a in ACTION_TO_IDX]

        # Explore
        if np.random.rand() < self.epsilon:
            return np.random.choice(allowed)

        # Exploit: pick highest Q-value among valid actions only
        with torch.no_grad():
            t      = torch.FloatTensor(state_vec).unsqueeze(0).to(self.device)
            q_vals = self.policy_net(t).squeeze(0)

        allowed_indices = [ACTION_TO_IDX[a] for a in allowed]
        best_idx        = max(allowed_indices, key=lambda i: q_vals[i].item())
        return ACTIONS[best_idx]

    # ── Experience storage ────────────────────────────────────────────────────

    def store(self, state, action, reward, next_state, done):
        self.buffer.push(state, ACTION_TO_IDX[action], reward, next_state, done)

    # ── Learning step ─────────────────────────────────────────────────────────

    def learn(self):
        """
        One gradient step using a sampled mini-batch from the replay buffer.

        CHANGE 3 — Double DQN target:
          Vanilla DQN:  target = r + γ · max_a Q_target(s', a)
          Problem: using the same network to both SELECT and EVALUATE the best
          next action leads to systematic Q-value overestimation, causing the
          agent to be over-confident and stop asking too early.

          Double DQN:   best_a = argmax_a Q_policy(s', a)   ← policy net selects
                        target = r + γ · Q_target(s', best_a) ← target net evaluates
          The two networks act as a check on each other, keeping Q-values
          calibrated and making the stopping decision more reliable.

        CHANGE 4 — Gradient clipping:
          Rewards span [-1, 3]. Without clipping, large TD errors produce
          large gradients that destabilise the network early in training.
          Clipping to max_norm=10 prevents this while still allowing fast
          learning on informative transitions.
        """
        if len(self.buffer) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)

        states      = torch.FloatTensor(states).to(self.device)
        actions     = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards     = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones       = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        # Current Q-values for actions taken
        q_values = self.policy_net(states).gather(1, actions)

        # CHANGE 3: Double DQN target
        with torch.no_grad():
            # Policy net picks the best action in next state
            best_next_actions = self.policy_net(next_states).argmax(dim=1, keepdim=True)
            # Target net evaluates that action
            max_next_q = self.target_net(next_states).gather(1, best_next_actions)
            targets    = rewards + self.gamma * max_next_q * (1 - dones)

        loss = self.loss_fn(q_values, targets)

        self.optimizer.zero_grad()
        loss.backward()

        # CHANGE 4: Gradient clipping
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=10.0)

        self.optimizer.step()

        # CHANGE 5: Update target net less frequently for stability
        if self.steps_done % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        return loss.item()

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: str):
        torch.save(self.policy_net.state_dict(), path)

    def load(self, path: str):
        self.policy_net.load_state_dict(
            torch.load(path, map_location=self.device)
        )
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.policy_net.eval()
        self.target_net.eval()