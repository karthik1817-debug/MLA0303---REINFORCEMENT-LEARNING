# Experiment 16
# Policy Gradient for Autonomous Lane Keeping

import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical


# --------------------------------
# 1. Policy Network
# --------------------------------

class PolicyNetwork(nn.Module):

    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, 3),
            nn.Softmax(dim=-1)
        )

    def forward(self, state):
        return self.network(state)


policy = PolicyNetwork()

optimizer = optim.Adam(
    policy.parameters(),
    lr=0.01
)


# --------------------------------
# 2. Actions
# --------------------------------

# 0 = Steer Left
# 1 = Go Straight
# 2 = Steer Right

actions = [
    "Steer Left",
    "Go Straight",
    "Steer Right"
]


# --------------------------------
# 3. Environment
# --------------------------------

# Lane position:
# -2 = far left
# -1 = left
#  0 = center
#  1 = right
#  2 = far right


def move_car(position, action):

    if action == 0:
        position -= 1

    elif action == 2:
        position += 1

    # Keep position within lane limits
    position = max(-2, min(2, position))

    return position


# --------------------------------
# 4. Reward Function
# --------------------------------

def get_reward(position):

    if position == 0:
        return 10

    elif abs(position) == 1:
        return 2

    else:
        return -5


# --------------------------------
# 5. Training
# --------------------------------

episodes = 500
gamma = 0.9

print("Training Lane-Keeping Agent...")


for episode in range(episodes):

    # Start from random lane position
    position = random.choice([-2, -1, 1, 2])

    log_probs = []
    rewards = []

    for step in range(10):

        state = torch.tensor(
            [position / 2],
            dtype=torch.float32
        )

        probabilities = policy(state)

        distribution = Categorical(
            probabilities
        )

        action = distribution.sample()

        log_prob = distribution.log_prob(
            action
        )

        # Perform steering action
        position = move_car(
            position,
            action.item()
        )

        reward = get_reward(position)

        log_probs.append(log_prob)
        rewards.append(reward)


    # --------------------------------
    # Discounted Returns
    # --------------------------------

    returns = []

    G = 0

    for reward in reversed(rewards):

        G = reward + gamma * G

        returns.insert(0, G)


    returns = torch.tensor(
        returns,
        dtype=torch.float32
    )

    # Normalize returns
    if len(returns) > 1:

        returns = (
            returns - returns.mean()
        ) / (
            returns.std() + 1e-9
        )


    # --------------------------------
    # Policy Gradient Loss
    # --------------------------------

    loss = 0

    for log_prob, G in zip(
        log_probs,
        returns
    ):

        loss += -log_prob * G


    optimizer.zero_grad()

    loss.backward()

    optimizer.step()


    if (episode + 1) % 100 == 0:

        print(
            f"Episode {episode + 1}/{episodes} completed"
        )


print("\nTraining Completed!")


# --------------------------------
# 6. Test Learned Policy
# --------------------------------

print("\n" + "=" * 55)

print("POLICY GRADIENT - AUTONOMOUS LANE KEEPING")

print("=" * 55)


test_positions = [-2, -1, 0, 1, 2]


for position in test_positions:

    state = torch.tensor(
        [position / 2],
        dtype=torch.float32
    )

    with torch.no_grad():

        probabilities = policy(state)

        action = torch.argmax(
            probabilities
        ).item()


    new_position = move_car(
        position,
        action
    )


    print("\nInitial Position:", position)

    print(
        "Selected Action:",
        actions[action]
    )

    print(
        "New Position:",
        new_position
    )

    print(
        "Reward:",
        get_reward(new_position)
    )


print("\nLane Position Meaning:")

print("-2,-1 = Left side")
print(" 0    = Lane center")
print(" 1, 2 = Right side")

print("\nExperiment completed successfully.")