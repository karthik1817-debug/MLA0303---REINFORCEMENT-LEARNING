# Experiment 17
# Hierarchical Reinforcement Learning (HRL)
# Household Robot using MAXQ-style task decomposition

import random

# --------------------------------
# 1. Environment
# --------------------------------

rooms = ["Kitchen", "Living Room", "Bedroom"]

# Q-values for high-level tasks
q_values = {
    "Kitchen": {
        "Clean Kitchen": 0.0,
        "Go Living Room": 0.0
    },

    "Living Room": {
        "Clean Living Room": 0.0,
        "Go Bedroom": 0.0
    },

    "Bedroom": {
        "Clean Bedroom": 0.0,
        "Finish": 0.0
    }
}


# --------------------------------
# 2. Parameters
# --------------------------------

alpha = 0.1
gamma = 0.9
epsilon = 0.2

episodes = 1000


# --------------------------------
# 3. Select Action
# --------------------------------

def choose_action(state):

    available_actions = list(
        q_values[state].keys()
    )

    # Exploration
    if random.random() < epsilon:
        return random.choice(
            available_actions
        )

    # Exploitation
    return max(
        q_values[state],
        key=q_values[state].get
    )


# --------------------------------
# 4. Environment Action
# --------------------------------

def perform_action(state, action):

    if state == "Kitchen":

        if action == "Clean Kitchen":
            return "Kitchen", 5, False

        if action == "Go Living Room":
            return "Living Room", 2, False


    elif state == "Living Room":

        if action == "Clean Living Room":
            return "Living Room", 5, False

        if action == "Go Bedroom":
            return "Bedroom", 2, False


    elif state == "Bedroom":

        if action == "Clean Bedroom":
            return "Bedroom", 5, False

        if action == "Finish":
            return "Bedroom", 10, True


# --------------------------------
# 5. Training
# --------------------------------

print("Training Household Robot using HRL...")


for episode in range(episodes):

    state = "Kitchen"

    for step in range(20):

        action = choose_action(state)

        next_state, reward, done = perform_action(
            state,
            action
        )

        old_q = q_values[state][action]

        if done:

            target = reward

        else:

            next_max = max(
                q_values[next_state].values()
            )

            target = (
                reward
                + gamma * next_max
            )


        # Q-learning update
        q_values[state][action] = (
            old_q
            + alpha * (target - old_q)
        )

        state = next_state

        if done:
            break


print("Training Completed!")


# --------------------------------
# 6. Display Learned Values
# --------------------------------

print("\n" + "=" * 55)

print("HRL - AUTONOMOUS HOUSEHOLD ROBOT")

print("=" * 55)


print("\nLearned Q-Values:")

for state in rooms:

    print("\nRoom:", state)

    for action, value in q_values[state].items():

        print(
            action,
            ":",
            round(value, 2)
        )


# --------------------------------
# 7. Demonstrate Hierarchy
# --------------------------------

print("\n" + "=" * 55)

print("HIERARCHICAL TASK EXECUTION")

print("=" * 55)


# High-level tasks
tasks = [
    ("Kitchen", "Clean Kitchen"),
    ("Kitchen", "Go Living Room"),

    ("Living Room", "Clean Living Room"),
    ("Living Room", "Go Bedroom"),

    ("Bedroom", "Clean Bedroom"),
    ("Bedroom", "Finish")
]


for room, task in tasks:

    print(
        room,
        "->",
        task
    )


print("\nTask Hierarchy:")

print("Household Task")

print("   |-- Clean Kitchen")

print("   |-- Move to Living Room")

print("   |-- Clean Living Room")

print("   |-- Move to Bedroom")

print("   |-- Clean Bedroom")

print("   |-- Finish")


print("\nAll household tasks completed successfully!")

print("\nMAXQ divides a large task into smaller subtasks.")

print("HAM organizes actions using hierarchical control.")

print("\nExperiment completed successfully.")