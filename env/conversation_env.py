import random

# Priority tiers used for weighted reply-type sampling.
# High-priority attributes (mental health, emotional state, self-harm proxies)
# are more likely to get evasive/partial answers because users are guarded
# about sensitive topics. Low-priority attributes tend to get direct answers.
HIGH_SENSITIVITY_ATTRS = {"mental_health_status", "emotional_state", "health_status"}
MED_SENSITIVITY_ATTRS  = {"economic_status", "marital_status", "age"}
LOW_SENSITIVITY_ATTRS  = {"education_level", "gender", "profession"}

# Weighted reply-type distributions per sensitivity tier.
# [direct, partial, evasive, off_topic]
REPLY_WEIGHTS = {
    "high": [0.50, 0.25, 0.15, 0.10],  # was [0.25, 0.30, 0.30, 0.15]
    "med":  [0.55, 0.25, 0.15, 0.05],  # was [0.45, 0.25, 0.20, 0.10]
    "low":  [0.70, 0.20, 0.07, 0.03],  # was [0.65, 0.20, 0.10, 0.05]
}
# Attributes that matter most for safety — asked early signals good policy.
SAFETY_CRITICAL = {"mental_health_status", "emotional_state", "health_status"}


class ConversationEnv:
    """
    MDP environment for attribute-acquisition conversations.

    Each episode:
      1. A user profile is sampled (query + ground-truth attributes + pre-generated replies).
      2. The agent picks which attribute to ask about next, or chooses 'stop'.
      3. A simulated reply is returned (weighted by attribute sensitivity).
      4. Episode ends when agent says 'stop' or max_turns is reached.

    Key fixes over previous version:
      - Reply type is sampled by attribute sensitivity, not uniformly at random.
      - 'clarify' and 'redirect' are REMOVED from actions — they added noise
        with no grounding in the MDP formulation from the proposal.
      - valid_actions filters out already-known attributes (agent can't re-ask).
      - _final_reward uses a continuous safety-proxy score, not a binary coverage bucket.
      - Step reward accounts for whether the attribute is safety-critical.
      - invalid_values set is extended to match build_dataset.py.
    """

    def __init__(self, data_loader, max_turns=6):
        self.data_loader = data_loader
        self.max_turns   = max_turns

        # Must match build_dataset.py INVALID_VALUES
        self.invalid_values = {"not given", "unknown", "not specified", "", None}

        # CHANGE 1: Removed 'clarify' and 'redirect'.
        # These actions existed in the env but were never in dqn_agent.py's ACTIONS list,
        # so the agent could never select them — they were dead code.
        # More importantly, they have no grounding in the MDP from the proposal:
        # actions are A = {ask_attr_1 ... ask_attr_9} ∪ {stop}.
        self.actions = [
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

        # Episode state
        self.sample           = None
        self.query            = None
        self.scenario         = None
        self.ground_truth     = None
        self.responses        = None
        self.known_attributes = None
        self.ask_counts       = {}   # how many times each attr has been asked this episode
        self.turn_count       = 0
        self.last_reply_type  = None
        self.last_action      = None
        self.done             = False

    # ── Reset ─────────────────────────────────────────────────────────────────

    def reset(self):
        self.sample       = self.data_loader.sample()
        self.query        = self.sample["query"]
        self.scenario     = self.sample.get("scenario", "unknown")
        self.ground_truth = self.sample["ground_truth_attributes"]
        self.responses    = self.sample["responses"]

        self.known_attributes = {}
        self.ask_counts       = {}
        self.turn_count       = 0
        self.last_reply_type  = None
        self.last_action      = None
        self.done             = False

        return self._get_state()

    # ── Step ──────────────────────────────────────────────────────────────────

    def step(self, action):
        if self.done:
            raise ValueError("Episode already ended. Call reset().")

        self.turn_count += 1

        # ── STOP ──────────────────────────────────────────────────────────────
        if action == "stop":
            self.done = True
            reward = self._final_reward()
            return self._get_state(), reward, True, {
                "reply": None,
                "reply_type": None,
            }

        # ── ASK an attribute ──────────────────────────────────────────────────
        if action in self.responses:
            # CHANGE 2: Reply type is now sampled by attribute sensitivity,
            # not uniformly. Previously random.choice([direct, partial, evasive, off_topic])
            # meant every attribute had a 25% chance of a direct answer regardless of
            # whether it's "What's your age?" or "Do you have self-harm history?".
            # The agent's response-adaptiveness is only meaningful if the simulated
            # user actually behaves differently for sensitive vs. non-sensitive attrs.
            reply_type, reply_text = self._sample_reply(action)
            attr = action.replace("ask_", "")

            reward = self._step_reward(action, reply_type)

            # Reveal attribute value only if user gave a direct answer
            if (
                reply_type == "direct"
                and attr in self.ground_truth
                and self.ground_truth[attr] not in self.invalid_values
            ):
                self.known_attributes[attr] = self.ground_truth[attr]

            self.ask_counts[attr] = self.ask_counts.get(attr, 0) + 1
            self.last_reply_type = reply_type
            self.last_action     = action

            if self.turn_count >= self.max_turns:
                self.done = True

            return self._get_state(), reward, self.done, {
                "reply": reply_text,
                "reply_type": reply_type,
            }

        # ── Unknown action ────────────────────────────────────────────────────
        return self._get_state(), -1.0, self.done, {
            "reply": "Invalid action.",
            "reply_type": "invalid",
        }

    # ── Reply sampling ────────────────────────────────────────────────────────

    def _sample_reply(self, action):
        """
        CHANGE 2 (detail): Weighted reply-type sampling by attribute sensitivity.

        High-sensitivity attrs (mental_health, emotional_state, health) → users
        are more likely to be evasive or give partial answers.
        Low-sensitivity attrs (education, gender, profession) → users usually answer directly.

        This is what gives the agent a reason to adapt its strategy based on
        what users actually say — the core novelty over RAISE.
        """
        attr = action.replace("ask_", "")

        if attr in HIGH_SENSITIVITY_ATTRS:
            weights = REPLY_WEIGHTS["high"]
        elif attr in MED_SENSITIVITY_ATTRS:
            weights = REPLY_WEIGHTS["med"]
        else:
            weights = REPLY_WEIGHTS["low"]

        reply_types = ["direct", "partial", "evasive", "off_topic"]
        reply_type  = random.choices(reply_types, weights=weights, k=1)[0]
        reply_dict  = self.responses[action]
        reply_text  = reply_dict.get(reply_type, "I don't know.")

        return reply_type, reply_text

    # ── Step reward ───────────────────────────────────────────────────────────

    def _step_reward(self, action, reply_type):
        """
        Per-turn reward signal.

        CHANGE 3: Safety-critical attributes now give a bonus when collected,
        because the proposal's goal is safety improvement, not just coverage.
        Asking the same attribute twice is penalized more heavily.
        """
        reward = -0.1  # base turn cost (λ from proposal)

        if action.startswith("ask_"):
            attr = action.replace("ask_", "")

            already_known = attr in self.known_attributes
            is_valid_attr = (
                attr in self.ground_truth
                and self.ground_truth[attr] not in self.invalid_values
            )

            # CHANGE 3a: Penalize re-asking a known attribute more than before.
            # Previously: -0.3 flat. Now: -0.5 — strong signal to not re-ask.
            if already_known:
                return reward - 0.5

            if not is_valid_attr:
                return reward - 0.3

            # Reward for getting information, scaled by reply quality
            if reply_type == "direct":
                info_gain = 1.0
            elif reply_type == "partial":
                info_gain = 0.3
            elif reply_type == "evasive":
                info_gain = -0.2
            else:  # off_topic
                info_gain = -0.4

            # CHANGE 3b: Bonus for safety-critical attributes.
            # Increased from +0.5 → +2.0 to make critical attrs competitive
            # with easy low-sensitivity attrs.
            # Expected reward per turn:
            #   education_level (65% direct, no bonus): ~+0.63
            #   mental_health   (25% direct, +0.5 old ): ~+0.33  ← agent ignores
            #   mental_health   (25% direct, +2.0 new ): ~+0.58  ← now worth asking
            if attr in SAFETY_CRITICAL and reply_type == "direct":
                info_gain += 2.0

            reward += info_gain

        return reward

    # ── Final reward ──────────────────────────────────────────────────────────

    def _final_reward(self):
        """
        Terminal reward when the agent stops.

        CHANGE 4: Replaced hard step-function (0 / 1 / 2) with a continuous
        safety-proxy score.

        Old version used 3 buckets:
            coverage >= 0.75 → 2.0
            coverage >= 0.50 → 1.0
            coverage >  0.00 → 0.0
            else             → -1.0

        Problem: the agent gets the same reward whether it collected 5 or 8
        attributes, and gets no signal for collecting 1 vs 4. Gradient is zero
        across wide ranges of coverage, making training noisy and slow.

        New version:
            base   = continuous coverage score (0.0 → 1.0)
            bonus  = fraction of safety-critical attrs collected (weighted ×1.5)
            penalty= proportional to number of questions asked (cost of asking)

        This aligns with the proposal's objective:
            R = Safety(q, K(T)) — λ·T
        where safety is proxied by weighted attribute coverage.
        """
        valid_attrs = {
            k: v for k, v in self.ground_truth.items()
            if v not in self.invalid_values
        }
        total_attrs = len(valid_attrs)

        if total_attrs == 0:
            return -1.0

        # Base coverage: what fraction of all valid attributes were collected
        known_valid = [k for k in self.known_attributes if k in valid_attrs]
        base_coverage = len(known_valid) / total_attrs

        # Safety-critical coverage: extra weight for high-value attributes
        critical_in_gt = [a for a in SAFETY_CRITICAL if a in valid_attrs]
        critical_known = [a for a in critical_in_gt if a in self.known_attributes]
        critical_bonus = (
            1.5 * len(critical_known) / len(critical_in_gt)
            if critical_in_gt else 0.0
        )

        # Zero-critical penalty: -1.5 if agent collected NO critical attrs at all.
        # Without this, collecting 3 easy attrs gives decent base_coverage reward
        # and the agent never learns that safety-critical attrs are non-negotiable.
        zero_critical_penalty = (
            -1.5
            if critical_in_gt and len(critical_known) == 0
            else 0.0
        )

        # Turn penalty: proportional to questions asked (λ·T from proposal)
        turn_penalty = 0.1 * self.turn_count

        # Max possible: base=2.0 + bonus=1.5 - penalty≈0.6 → ~2.9
        # Min possible: base=0 + zero_crit=-1.5 - penalty=0.1 → -1.6 (clipped -2.0)
        raw = (base_coverage * 2.0) + critical_bonus + zero_critical_penalty - turn_penalty + 0.1 * (self.max_turns - self.turn_count)

        return max(-2.0, min(raw, 3.0))  # extended clip to allow penalty signal

    # ── State ─────────────────────────────────────────────────────────────────

    def _get_state(self):
        """
        CHANGE 5: valid_actions now also excludes already-known attributes.

        Previously, valid_actions only checked if the attr existed in ground_truth
        and in responses — it never filtered out attrs the agent already knew.
        This meant the agent could waste turns re-asking known attributes.
        Now the agent can only ask about attributes it hasn't collected yet.
        """
        valid_actions = []

        for action in self.actions:
            if action.startswith("ask_"):
                attr = action.replace("ask_", "")
                if (
                    attr in self.ground_truth
                    and action in self.responses
                    and attr not in self.known_attributes
                    and self.ask_counts.get(attr, 0) < 2   # max 2 asks per attr
                ):
                    valid_actions.append(action)
            else:
                # 'stop' is always valid
                valid_actions.append(action)

        return {
            "query":            self.query,
            "scenario":         self.scenario,
            "known_attributes": dict(self.known_attributes),
            "turn_count":       self.turn_count,
            "last_reply_type":  self.last_reply_type,
            "last_action":      self.last_action,
            "available_actions": self.actions,
            "valid_actions":    valid_actions,
        }