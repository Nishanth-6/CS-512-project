import json
import random


class DataLoader:
    def __init__(self, path: str):
        with open(path, "r", encoding="utf-8") as f:
            self.data = json.load(f)

    def sample(self):
        return random.choice(self.data)

    def get_by_index(self, idx: int):
        return self.data[idx]

    def __len__(self):
        return len(self.data)