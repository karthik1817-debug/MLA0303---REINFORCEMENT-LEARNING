# Experiment 4
# Bellman Equation for Delivery Robot

# States
states = ["Warehouse", "A", "B", "Destination"]

# Rewards
rewards = {
    "Warehouse": -1,
    "A": -1,
    "B": -1,
    "Destination": 10
}

# Possible next states
transitions = {
    "Warehouse": ["A", "B"],
    "A": ["B", "Destination"],
    "B": ["A", "Destination"]
}

# Discount factor
gamma = 0.9

# Initial state values
values = {
    state: 0
    for state in states
}

# Destination is terminal
values["Destination"] = 10


# Apply Bellman Equation
for iteration in range(10):

    new_values = values.copy()

    for state in transitions:

        possible_values = []

        for next_state in transitions[state]:

            value = rewards[next_state] + gamma * values[next_state]

            possible_values.append(value)

        new_values[state] = max(possible_values)

    values = new_values


# Display results
print("=" * 50)
print("BELLMAN EQUATION - DELIVERY ROBOT")
print("=" * 50)

print("\nFinal State Values:")

for state in states:
    print(state, ":", round(values[state], 2))


# Find best path
print("\nBest Delivery Path:")

current_state = "Warehouse"

print(current_state, end="")

while current_state != "Destination":

    next_state = max(
        transitions[current_state],
        key=lambda s: rewards[s] + gamma * values[s]
    )

    print(" ->", next_state, end="")

    current_state = next_state

print("\n\nDelivery completed successfully.")