# Experiment 11
# DQN vs Double DQN vs Dueling DQN with PER

import random
import numpy as np
import torch
import torch.nn as nn


# ---------------------------------
# Settings
# ---------------------------------

STATE_SIZE = 4
ACTION_SIZE = 2


# =================================
# 1. Standard DQN
# =================================

class DQN(nn.Module):

    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(STATE_SIZE, 16),
            nn.ReLU(),
            nn.Linear(16, ACTION_SIZE)
        )

    def forward(self, x):
        return self.network(x)


# =================================
# 2. Double DQN
# =================================

class DoubleDQN(nn.Module):

    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(STATE_SIZE, 16),
            nn.ReLU(),
            nn.Linear(16, ACTION_SIZE)
        )

    def forward(self, x):
        return self.network(x)


# =================================
# 3. Dueling DQN
# =================================

class DuelingDQN(nn.Module):

    def __init__(self):
        super().__init__()

        self.feature = nn.Sequential(
            nn.Linear(STATE_SIZE, 16),
            nn.ReLU()
        )

        # State value
        self.value = nn.Linear(16, 1)

        # Action advantages
        self.advantage = nn.Linear(
            16,
            ACTION_SIZE
        )

    def forward(self, x):

        features = self.feature(x)

        value = self.value(features)

        advantage = self.advantage(features)

        return (
            value
            + advantage
            - advantage.mean(
                dim=-1,
                keepdim=True
            )
        )


# =================================
# 4. Prioritized Experience Replay
# =================================

class PERMemory:

    def __init__(self):
        self.memory = []
        self.priorities = []

    def add(self, experience, priority):

        self.memory.append(experience)

        self.priorities.append(priority)

    def sample(self):

        probabilities = np.array(
            self.priorities,
            dtype=float
        )

        probabilities /= probabilities.sum()

        index = np.random.choice(
            len(self.memory),
            p=probabilities
        )

        return self.memory[index]


# =================================
# Create Networks
# =================================

dqn = DQN()

double_dqn = DoubleDQN()

dueling_dqn = DuelingDQN()


# Example state
state = torch.tensor(
    [1.0, 0.5, -0.2, 0.8],
    dtype=torch.float32
)


# Get Q-values
with torch.no_grad():

    dqn_values = dqn(state)

    double_values = double_dqn(state)

    dueling_values = dueling_dqn(state)


# =================================
# Double DQN Action Selection
# =================================

with torch.no_grad():

    # Online network selects action
    best_action = torch.argmax(
        double_dqn(state)
    ).item()

    # Target network would normally
    # evaluate this selected action
    double_selected_value = double_values[
        best_action
    ].item()


# =================================
# PER Example
# =================================

memory = PERMemory()

memory.add(
    ("State 1", "Action 0", 1),
    priority=1
)

memory.add(
    ("State 2", "Action 1", 5),
    priority=5
)

memory.add(
    ("State 3", "Action 0", 10),
    priority=10
)

sample = memory.sample()


# =================================
# Display Results
# =================================

print("=" * 60)

print("DQN, DOUBLE DQN, DUELING DQN AND PER")

print("=" * 60)


print("\nStandard DQN")

print(
    "Q-Values:",
    dqn_values.numpy().round(3)
)

print(
    "Selected Action:",
    torch.argmax(dqn_values).item()
)


print("\nDouble DQN")

print(
    "Q-Values:",
    double_values.numpy().round(3)
)

print(
    "Selected Action:",
    best_action
)

print(
    "Selected Q-Value:",
    round(double_selected_value, 3)
)


print("\nDueling DQN")

print(
    "Q-Values:",
    dueling_values.numpy().round(3)
)

print(
    "Selected Action:",
    torch.argmax(dueling_values).item()
)


print("\nPrioritized Experience Replay")

print(
    "Sampled Experience:",
    sample
)


print("\nComparison:")

print("DQN        : Basic Deep Q-Network")

print(
    "Double DQN : Reduces Q-value overestimation"
)

print(
    "Dueling DQN: Separates state value and action advantage"
)

print(
    "PER        : Gives important experiences higher sampling priority"
)

print("\nExperiment completed successfully.")