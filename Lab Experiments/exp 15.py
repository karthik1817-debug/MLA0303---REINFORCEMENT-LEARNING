# Experiment 15
# PPO for Humanoid Robot Walking and Balance

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import random


# --------------------------------
# 1. Policy Network
# --------------------------------

class PolicyNetwork(nn.Module):

    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(),
            nn.Linear(32, 3),
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

# 0 = Lean Left
# 1 = Stay Balanced
# 2 = Lean Right

actions = [
    "Lean Left",
    "Stay Balanced",
    "Lean Right"
]


# --------------------------------
# 3. Environment
# --------------------------------

def perform_action(balance, action):

    if action == 0:
        balance -= 1

    elif action == 2:
        balance += 1

    return balance


def get_reward(balance):

    # Best position is balance = 0

    if balance == 0:
        return 10

    elif abs(balance) == 1:
        return 2

    else:
        return -5


# --------------------------------
# 4. PPO Parameters
# --------------------------------

episodes = 500

clip_value = 0.2


print("Training Humanoid Robot using PPO...")


# --------------------------------
# 5. PPO Training
# --------------------------------

for episode in range(episodes):

    # Random initial imbalance
    balance = random.choice([-2, -1, 1, 2])

    # State:
    # balance + walking progress
    state = torch.tensor(
        [
            balance / 2,
            0.0
        ],
        dtype=torch.float32
    )

    # Old policy
    probabilities = policy(state)

    distribution = Categorical(
        probabilities
    )

    action = distribution.sample()

    old_log_probability = (
        distribution.log_prob(action).detach()
    )


    # Perform action
    new_balance = perform_action(
        balance,
        action.item()
    )

    reward = get_reward(
        new_balance
    )


    # --------------------------------
    # PPO Update
    # --------------------------------

    for update in range(4):

        new_probabilities = policy(
            state
        )

        new_distribution = Categorical(
            new_probabilities
        )

        new_log_probability = (
            new_distribution.log_prob(action)
        )


        # Probability ratio
        ratio = torch.exp(
            new_log_probability
            - old_log_probability
        )


        advantage = torch.tensor(
            float(reward)
        )


        # Normal PPO objective
        objective1 = (
            ratio * advantage
        )


        # Clipped PPO objective
        objective2 = (
            torch.clamp(
                ratio,
                1 - clip_value,
                1 + clip_value
            )
            * advantage
        )


        # PPO loss
        loss = -torch.min(
            objective1,
            objective2
        )


        optimizer.zero_grad()

        loss.backward()

        optimizer.step()


    if (episode + 1) % 100 == 0:

        print(
            f"Episode {episode + 1}/{episodes} completed"
        )


print("\nPPO Training Completed!")


# --------------------------------
# 6. Test Robot
# --------------------------------

print("\nTesting Humanoid Robot...")


test_balances = [
    -2,
    -1,
    1,
    2
]


# --------------------------------
# 7. Display Results
# --------------------------------

print("\n" + "=" * 55)

print("PPO - HUMANOID ROBOT BALANCE")

print("=" * 55)


for balance in test_balances:

    state = torch.tensor(
        [
            balance / 2,
            0.0
        ],
        dtype=torch.float32
    )


    with torch.no_grad():

        probabilities = policy(
            state
        )

        action = torch.argmax(
            probabilities
        ).item()


    new_balance = perform_action(
        balance,
        action
    )


    print(
        "\nInitial Balance:",
        balance
    )

    print(
        "Robot Action:",
        actions[action]
    )

    print(
        "New Balance:",
        new_balance
    )

    print(
        "Reward:",
        get_reward(new_balance)
    )


print("\n" + "=" * 55)

print("PPO AND TRPO COMPARISON")

print("=" * 55)

print(
    "PPO  : Uses clipping to prevent large policy updates."
)

print(
    "TRPO : Uses a trust region to restrict policy updates."
)

print(
    "Goal : Stable walking and balance."
)

print("\nExperiment completed successfully.")