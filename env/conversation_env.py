import random

from env import reward


class ConversationEnv:
    def __init__(self, data_loader, max_turns=6):
        self.data_loader = data_loader
        self.max_turns = max_turns

        self.actions = [
            "ask_education_level",
            "ask_age",
            "ask_gender",
            "ask_marital_status",
            "clarify",
            "redirect",
            "stop"
        ]

        self.sample = None
        self.query = None
        self.scenario = None
        self.ground_truth = None
        self.responses = None
        self.known_attributes = None
        self.turn_count = 0
        self.last_reply_type = None
        self.last_action = None
        self.done = False

    def reset(self):
        self.sample = self.data_loader.sample()
        self.query = self.sample["query"]
        self.scenario = self.sample["scenario"]
        self.ground_truth = self.sample["ground_truth_attributes"]
        self.responses = self.sample["responses"]

        self.known_attributes = {}
        self.turn_count = 0
        self.last_reply_type = None
        self.last_action = None
        self.done = False

        return self._get_state()

    def step(self, action):
        if self.done:
            raise ValueError("Episode already ended. Call reset().")

        self.turn_count += 1

        if action == "stop":
            self.done = True
            reward = self._final_reward()
            return self._get_state(), reward, True, {
                "reply": None,
                "reply_type": None
            }

        if action in self.responses:
            reply_type, reply_text = self._sample_reply(action)
            attr = action.replace("ask_", "")

            # reveal only if usable direct answer
            if (
                reply_type == "direct"
                and attr in self.ground_truth
                and self.ground_truth[attr] != "not given"
            ):
                self.known_attributes[attr] = self.ground_truth[attr]

            reward = self._step_reward(action, reply_type)

            self.last_reply_type = reply_type
            self.last_action = action

            if self.turn_count >= self.max_turns:
                self.done = True

            return self._get_state(), reward, self.done, {
                "reply": reply_text,
                "reply_type": reply_type
            }

        if action == "clarify":
            reward = self._step_reward(action, None)
            self.last_action = action

            if self.turn_count >= self.max_turns:
                self.done = True

            return self._get_state(), reward, self.done, {
                "reply": "Can you clarify what you mean?",
                "reply_type": "meta"
            }

        if action == "redirect":
            reward = self._step_reward(action, None)
            self.last_action = action

            if self.turn_count >= self.max_turns:
                self.done = True

            return self._get_state(), reward, self.done, {
                "reply": "Let's come back to the earlier question for a moment.",
                "reply_type": "meta"
            }

        return self._get_state(), -1.0, self.done, {
            "reply": "Invalid action.",
            "reply_type": "invalid"
        }

    def _sample_reply(self, action):
        reply_dict = self.responses[action]
        reply_type = random.choice(["direct", "partial", "evasive", "off_topic"])
        reply_text = reply_dict.get(reply_type, "I don't know.")
        return reply_type, reply_text

    def _step_reward(self, action, reply_type):
        reward = -0.1  # turn cost

        if action.startswith("ask_"):
            attr = action.replace("ask_", "")

            is_valid_unknown = (
                attr in self.ground_truth
                and self.ground_truth[attr] != "not given"
                and attr not in self.known_attributes
            )

            if not is_valid_unknown:
                reward -= 0.3
            else:
                if reply_type == "direct":
                    reward += 1.0
                elif reply_type == "partial":
                    reward += 0.3
                elif reply_type == "evasive":
                    reward -= 0.2
                elif reply_type == "off_topic":
                    reward -= 0.4

        elif action == "clarify":
            if self.last_reply_type in ["partial", "evasive"]:
                reward += 0.4
            else:
                reward -= 0.2

        elif action == "redirect":
            if self.last_reply_type == "off_topic":
                reward += 0.6
            else:
                reward -= 0.2

        return reward

    def _final_reward(self):
        valid_attrs = {
            k: v for k, v in self.ground_truth.items()
            if v != "not given"
        }

        total_attrs = len(valid_attrs)
        known = len([k for k in self.known_attributes if k in valid_attrs])

        if total_attrs == 0:
            return -1.0

        coverage = known / total_attrs

        if coverage >= 0.75:
            return 2.0
        elif coverage >= 0.5:
            return 1.0
        elif coverage > 0:
            return 0.0
        else:
            return -1.0

    def _get_state(self):
        return {
            "query": self.query,
            "scenario": self.scenario,
            "known_attributes": dict(self.known_attributes),
            "turn_count": self.turn_count,
            "last_reply_type": self.last_reply_type,
            "last_action": self.last_action,
            "available_actions": self.actions
        }