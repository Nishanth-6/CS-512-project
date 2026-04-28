import random
from env.data_loader import DataLoader
from env.conversation_env import ConversationEnv

loader = DataLoader("data/merged_9attr_dataset.json")
env = ConversationEnv(loader)

state = env.reset()

print("Initial query:", state["query"]) 1   `1.90loi`qouou0j08llpi98 ZX ''
print("Scenario:", state["scenario"])

done = False
while not done:
    action = random.choice(state["valid_actions"])
    next_state, reward, done, info = env.step(action)

    print("\n---")
    print("Action:", action)
    print("Reply:", info.get("reply"))
    print("Reply type:", info.get("reply_type"))
    print("Reward:", reward)
    print("Known attributes:", next_state["known_attributes"])
    print("Done:", done)

    state = next_state