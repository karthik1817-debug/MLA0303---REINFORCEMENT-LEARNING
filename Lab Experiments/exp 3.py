# Experiment 3
# Markov Decision Process for Warehouse Robot Navigation

# Warehouse locations
states = ["Start", "A", "B", "C", "Goal"]

# Possible robot movements
actions = {
    "Start": ["A", "B"],
    "A": ["B", "C"],
    "B": ["A", "C"],
    "C": ["B", "Goal"]
}

# Rewards for reaching each location
rewards = {
    "Start": -1,
    "A": -1,
    "B": -1,
    "C": -1,
    "Goal": 10
}

# Discount factor
gamma = 0.9

# Initial state values
values = {
    state: 0
    for state in states
}

# Value Iteration
for iteration in range(20):

    new_values = values.copy()

    for state in actions:

        possible_values = []

        for next_state in actions[state]:

            value = rewards[next_state] + gamma * values[next_state]

            possible_values.append(value)

        new_values[state] = max(possible_values)

    values = new_values


# Display results
print("=" * 50)
print("WAREHOUSE ROBOT NAVIGATION USING MDP")
print("=" * 50)

print("\nState Values:")

for state in states:
    print(state, ":", round(values[state], 2))


print("\nOptimal Robot Path:")

current_state = "Start"

print(current_state, end="")

while current_state != "Goal":

    next_state = max(
        actions[current_state],
        key=lambda state:
        rewards[state] + gamma * values[state]
    )

    print(" ->", next_state, end="")

    current_state = next_state

print("\n\nRobot successfully reached the Goal.")