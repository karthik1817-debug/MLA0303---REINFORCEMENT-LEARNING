# Experiment 6
# Robot Navigation using Q-Learning with Gymnasium

import gymnasium as gym
import numpy as np
import random

# Create FrozenLake environment
# is_slippery=False makes movement deterministic
env = gym.make("FrozenLake-v1", is_slippery=False)

num_states = env.observation_space.n
num_actions = env.action_space.n

# Initialize Q-table
q_table = np.zeros((num_states, num_actions))

# Parameters
learning_rate = 0.8
discount_factor = 0.95

episodes = 10000

# Start with high exploration
epsilon = 1.0
epsilon_decay = 0.999
min_epsilon = 0.05


# -------------------------
# Training
# -------------------------

for episode in range(episodes):

    state, _ = env.reset()

    terminated = False
    truncated = False

    while not (terminated or truncated):

        # Epsilon-greedy strategy
        if random.random() < epsilon:
            # Explore
            action = env.action_space.sample()

        else:
            # Exploit
            action = np.argmax(q_table[state])

        # Perform action
        next_state, reward, terminated, truncated, _ = env.step(action)

        # Q-learning formula
        old_value = q_table[state, action]

        next_max = np.max(q_table[next_state])

        q_table[state, action] = old_value + learning_rate * (
            reward
            + discount_factor * next_max
            - old_value
        )

        state = next_state

    # Reduce exploration gradually
    epsilon = max(min_epsilon, epsilon * epsilon_decay)


# -------------------------
# Display Q-table
# -------------------------

print("=" * 55)
print("ROBOT NAVIGATION USING Q-LEARNING")
print("=" * 55)

print("\nLearned Q-Table:")

print(np.round(q_table, 2))


# -------------------------
# Test Robot
# -------------------------

state, _ = env.reset()

terminated = False
truncated = False

path = [state]

# Prevent infinite movement
max_steps = 30

for step in range(max_steps):

    # Choose best learned action
    action = np.argmax(q_table[state])

    state, reward, terminated, truncated, _ = env.step(action)

    path.append(state)

    if terminated or truncated:
        break


# -------------------------
# Result
# -------------------------

print("\nRobot Path:")

print(" -> ".join(map(str, path)))

if reward == 1:

    print("\nRobot successfully reached the goal!")

else:

    print("\nRobot failed to reach the goal.")


env.close()