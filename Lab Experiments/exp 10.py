# Experiment 10
# Deep Q-Network (DQN) for Drone Navigation

import random
import torch
import torch.nn as nn
import torch.optim as optim

# -----------------------------
# Environment
# -----------------------------

GRID_SIZE = 5
START = (0, 0)
GOAL = (4, 4)

# Actions:
# 0 = UP
# 1 = DOWN
# 2 = LEFT
# 3 = RIGHT

actions = ["UP", "DOWN", "LEFT", "RIGHT"]


def move(state, action):
    row, col = state

    if action == 0:
        row = max(0, row - 1)

    elif action == 1:
        row = min(GRID_SIZE - 1, row + 1)

    elif action == 2:
        col = max(0, col - 1)

    elif action == 3:
        col = min(GRID_SIZE - 1, col + 1)

    next_state = (row, col)

    if next_state == GOAL:
        reward = 20
        done = True
    else:
        reward = -1
        done = False

    return next_state, reward, done


# -----------------------------
# DQN Model
# -----------------------------

class DQN(nn.Module):

    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(2, 32)
        self.fc2 = nn.Linear(32, 4)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


model = DQN()

optimizer = optim.Adam(
    model.parameters(),
    lr=0.01
)

loss_function = nn.MSELoss()


# -----------------------------
# Convert State
# -----------------------------

def convert_state(state):

    return torch.tensor(
        [
            state[0] / 4,
            state[1] / 4
        ],
        dtype=torch.float32
    )


# -----------------------------
# Training Parameters
# -----------------------------

episodes = 300
gamma = 0.9

epsilon = 1.0
epsilon_min = 0.05
epsilon_decay = 0.98


print("Training DQN Drone...")


# -----------------------------
# Training
# -----------------------------

for episode in range(episodes):

    state = START

    for step in range(30):

        state_tensor = convert_state(state)

        # Exploration
        if random.random() < epsilon:

            action = random.randint(0, 3)

        # Exploitation
        else:

            with torch.no_grad():

                q_values = model(state_tensor)

                action = torch.argmax(
                    q_values
                ).item()


        next_state, reward, done = move(
            state,
            action
        )

        next_tensor = convert_state(
            next_state
        )


        # Current Q-values
        q_values = model(state_tensor)


        # Target Q-values
        target = q_values.detach().clone()


        if done:

            target[action] = reward

        else:

            with torch.no_grad():

                next_q = model(next_tensor)

                target[action] = (
                    reward
                    + gamma * torch.max(next_q)
                )


        # Calculate loss
        loss = loss_function(
            q_values,
            target
        )


        # Update neural network
        optimizer.zero_grad()

        loss.backward()

        optimizer.step()


        state = next_state


        if done:
            break


    # Reduce exploration
    epsilon = max(
        epsilon_min,
        epsilon * epsilon_decay
    )


    # Show progress
    if (episode + 1) % 50 == 0:

        print(
            "Episode",
            episode + 1,
            "completed"
        )


print("\nTraining Completed!")


# -----------------------------
# Test Drone
# -----------------------------

state = START

path = [state]


for step in range(20):

    state_tensor = convert_state(state)

    with torch.no_grad():

        q_values = model(state_tensor)

        action = torch.argmax(
            q_values
        ).item()


    next_state, reward, done = move(
        state,
        action
    )


    # Prevent getting stuck
    if next_state == state:
        break


    path.append(next_state)

    state = next_state


    if done:
        break


# -----------------------------
# Output
# -----------------------------

print("\n" + "=" * 50)

print("DQN DRONE NAVIGATION")

print("=" * 50)


print("\nStart Position:", START)

print("Goal Position :", GOAL)


print("\nDrone Path:")

print(
    " -> ".join(
        str(position)
        for position in path
    )
)


if state == GOAL:

    print(
        "\nSUCCESS: Drone reached the target!"
    )

else:

    print(
        "\nDrone did not reach the target."
    )