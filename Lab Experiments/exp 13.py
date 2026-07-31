# Experiment 12
# Policy-Based Reinforcement Learning for Robotic Arm

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
            nn.Linear(2, 16),
            nn.ReLU(),
            nn.Linear(16, 3),
            nn.Softmax(dim=-1)
        )

    def forward(self, state):
        return self.network(state)


# --------------------------------
# 2. Create Model
# --------------------------------

policy = PolicyNetwork()

optimizer = optim.Adam(
    policy.parameters(),
    lr=0.01
)


# --------------------------------
# 3. Environment
# --------------------------------

# Robotic arm position
# State = [x, y]

start_position = [0.0, 0.0]

# Target position
target_position = [2.0, 2.0]

# Actions:
# 0 = Move X
# 1 = Move Y
# 2 = Stay

actions = [
    "Move X",
    "Move Y",
    "Stay"
]


def perform_action(state, action):

    x, y = state

    if action == 0:
        x += 1

    elif action == 1:
        y += 1

    return [x, y]


# --------------------------------
# 4. Reward Function
# --------------------------------

def get_reward(state):

    if state == target_position:
        return 10

    # Penalize unnecessary movement
    return -1


# --------------------------------
# 5. Training
# --------------------------------

episodes = 500

print("Training Robotic Arm...")


for episode in range(episodes):

    state = start_position.copy()

    log_probs = []
    rewards = []

    for step in range(10):

        state_tensor = torch.tensor(
            state,
            dtype=torch.float32
        )

        # Get action probabilities
        probabilities = policy(state_tensor)

        distribution = Categorical(probabilities)

        action = distribution.sample()

        log_probs.append(
            distribution.log_prob(action)
        )

        # Perform selected action
        state = perform_action(
            state,
            action.item()
        )

        reward = get_reward(state)

        rewards.append(reward)

        # Stop when target reached
        if state == target_position:
            break


    # Total return
    total_reward = sum(rewards)

    # Policy gradient loss
    loss = 0

    for log_prob in log_probs:
        loss += -log_prob * total_reward


    optimizer.zero_grad()

    loss.backward()

    optimizer.step()


    if (episode + 1) % 100 == 0:

        print(
            "Episode",
            episode + 1,
            "completed"
        )


print("\nTraining Completed!")


# --------------------------------
# 6. Test Trained Arm
# --------------------------------

state = start_position.copy()

path = [state.copy()]


for step in range(10):

    state_tensor = torch.tensor(
        state,
        dtype=torch.float32
    )

    with torch.no_grad():

        probabilities = policy(
            state_tensor
        )

        action = torch.argmax(
            probabilities
        ).item()


    state = perform_action(
        state,
        action
    )

    path.append(state.copy())


    if state == target_position:
        break


# --------------------------------
# 7. Display Result
# --------------------------------

print("\n" + "=" * 55)

print("POLICY-BASED RL - ROBOTIC ARM")

print("=" * 55)


print("\nStart Position :", start_position)

print("Target Position:", target_position)


print("\nArm Movement:")

for position in path:

    print(position, end=" -> ")

print("END")


if state == target_position:

    print(
        "\nSUCCESS: Robotic arm reached the target!"
    )

else:

    print(
        "\nRobotic arm did not reach the target."
    )


print("\nLearned Action Probabilities:")

state_tensor = torch.tensor(
    start_position,
    dtype=torch.float32
)

with torch.no_grad():

    probabilities = policy(
        state_tensor
    )


for action, probability in zip(
    actions,
    probabilities
):

    print(
        action,
        ":",
        round(probability.item(), 3)
    )