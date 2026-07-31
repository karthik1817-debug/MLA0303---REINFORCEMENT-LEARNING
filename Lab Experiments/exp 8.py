# Experiment 8
# Monte Carlo Method for Vacuum Cleaner Robot

import random

# Possible states
states = ["Room A", "Room B"]

# Store total rewards and number of visits
total_rewards = {
    "Room A": 0,
    "Room B": 0
}

visits = {
    "Room A": 0,
    "Room B": 0
}

# Number of episodes
episodes = 1000


# ----------------------------
# Generate Episodes
# ----------------------------

for episode in range(episodes):

    # Randomly select starting room
    current_state = random.choice(states)

    episode_data = []

    # Each episode has 5 steps
    for step in range(5):

        # Randomly decide whether room is dirty
        dirty = random.choice([True, False])

        if dirty:
            action = "Clean"
            reward = 10
        else:
            action = "Move"
            reward = -1

        episode_data.append(
            (current_state, action, reward)
        )

        # Move to other room
        if action == "Move":

            if current_state == "Room A":
                current_state = "Room B"
            else:
                current_state = "Room A"


    # ----------------------------
    # Monte Carlo Update
    # ----------------------------

    G = 0

    # Calculate return backwards
    for state, action, reward in reversed(episode_data):

        G = reward + G

        total_rewards[state] += G
        visits[state] += 1


# ----------------------------
# Calculate State Values
# ----------------------------

state_values = {}

for state in states:

    if visits[state] > 0:
        state_values[state] = (
            total_rewards[state] / visits[state]
        )
    else:
        state_values[state] = 0


# ----------------------------
# Display Results
# ----------------------------

print("=" * 55)
print("MONTE CARLO - VACUUM CLEANER ROBOT")
print("=" * 55)

print("\nEstimated State Values:")

for state in states:

    print(
        state,
        ":",
        round(state_values[state], 2)
    )

print("\nNumber of Visits:")

for state in states:

    print(
        state,
        ":",
        visits[state]
    )

print("\nMonte Carlo learning completed successfully.")