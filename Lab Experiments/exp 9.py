# Experiment 9
# TD(0), SARSA and Q-Learning

import random

# Simple environment
# States: 0 -> 1 -> 2 -> 3 -> 4
# State 4 is the goal

num_states = 5
num_actions = 2

LEFT = 0
RIGHT = 1

alpha = 0.1
gamma = 0.9
epsilon = 0.2
episodes = 1000


# --------------------------------
# Environment
# --------------------------------

def move(state, action):

    if action == RIGHT:
        next_state = min(state + 1, 4)
    else:
        next_state = max(state - 1, 0)

    # Reward for reaching goal
    reward = 10 if next_state == 4 else -1

    return next_state, reward


# --------------------------------
# Epsilon-Greedy Action
# --------------------------------

def choose_action(q_table, state):

    if random.random() < epsilon:
        return random.choice([LEFT, RIGHT])

    if q_table[state][RIGHT] >= q_table[state][LEFT]:
        return RIGHT

    return LEFT


# ==================================
# 1. TD(0)
# ==================================

values = [0.0] * num_states

for episode in range(episodes):

    state = 0

    while state != 4:

        # Random policy
        action = random.choice([LEFT, RIGHT])

        next_state, reward = move(state, action)

        # TD(0) update
        values[state] += alpha * (
            reward
            + gamma * values[next_state]
            - values[state]
        )

        state = next_state


# ==================================
# 2. SARSA
# ==================================

sarsa_q = [
    [0.0, 0.0]
    for _ in range(num_states)
]

for episode in range(episodes):

    state = 0

    action = choose_action(sarsa_q, state)

    while state != 4:

        next_state, reward = move(state, action)

        if next_state == 4:

            sarsa_q[state][action] += alpha * (
                reward - sarsa_q[state][action]
            )

            break

        next_action = choose_action(
            sarsa_q,
            next_state
        )

        # SARSA update
        sarsa_q[state][action] += alpha * (
            reward
            + gamma * sarsa_q[next_state][next_action]
            - sarsa_q[state][action]
        )

        state = next_state
        action = next_action


# ==================================
# 3. Q-Learning
# ==================================

q_learning = [
    [0.0, 0.0]
    for _ in range(num_states)
]

for episode in range(episodes):

    state = 0

    while state != 4:

        action = choose_action(
            q_learning,
            state
        )

        next_state, reward = move(
            state,
            action
        )

        # Best future Q-value
        if next_state == 4:
            next_max = 0
        else:
            next_max = max(
                q_learning[next_state]
            )

        # Q-Learning update
        q_learning[state][action] += alpha * (
            reward
            + gamma * next_max
            - q_learning[state][action]
        )

        state = next_state


# ==================================
# Results
# ==================================

print("=" * 55)
print("TD(0), SARSA AND Q-LEARNING")
print("=" * 55)


print("\nTD(0) State Values:")

for state in range(num_states):

    print(
        "State",
        state,
        ":",
        round(values[state], 2)
    )


print("\nSARSA Q-Table:")

print("State\tLEFT\tRIGHT")

for state in range(num_states):

    print(
        state,
        "\t",
        round(sarsa_q[state][LEFT], 2),
        "\t",
        round(sarsa_q[state][RIGHT], 2)
    )


print("\nQ-Learning Q-Table:")

print("State\tLEFT\tRIGHT")

for state in range(num_states):

    print(
        state,
        "\t",
        round(q_learning[state][LEFT], 2),
        "\t",
        round(q_learning[state][RIGHT], 2)
    )


# --------------------------------
# Best Path using Q-Learning
# --------------------------------

state = 0
path = [state]

for step in range(10):

    if state == 4:
        break

    action = max(
        range(num_actions),
        key=lambda a: q_learning[state][a]
    )

    state, reward = move(state, action)

    path.append(state)


print("\nBest Path learned by Q-Learning:")

print(" -> ".join(map(str, path)))

if state == 4:
    print("\nAgent successfully reached the goal!")
else:
    print("\nAgent failed to reach the goal.")