import random

# Rooms in the smart home
rooms = ["Living Room", "Kitchen", "Bedroom", "Charging Station"]

# Q-table
q_table = {
    room: {"Move": 0, "Clean": 0, "Charge": 0}
    for room in rooms
}

# Parameters
learning_rate = 0.1
discount_factor = 0.9
epsilon = 0.2
episodes = 100

# Reward function
def get_reward(room, action):
    if action == "Clean" and room != "Charging Station":
        return 10

    if action == "Charge" and room == "Charging Station":
        return 20

    if action == "Move":
        return -1

    return -5


# Training
for episode in range(episodes):

    current_room = random.choice(rooms)

    for step in range(10):

        # Epsilon-greedy action selection
        if random.random() < epsilon:
            action = random.choice(list(q_table[current_room].keys()))
        else:
            action = max(
                q_table[current_room],
                key=q_table[current_room].get
            )

        reward = get_reward(current_room, action)

        # Move to another room
        if action == "Move":
            next_room = random.choice(rooms)
        else:
            next_room = current_room

        # Q-learning update
        old_value = q_table[current_room][action]

        next_max = max(q_table[next_room].values())

        new_value = old_value + learning_rate * (
            reward
            + discount_factor * next_max
            - old_value
        )

        q_table[current_room][action] = new_value

        current_room = next_room


# Display learned Q-values
print("=" * 55)
print("SMART HOME ROBOT - REINFORCEMENT LEARNING")
print("=" * 55)

for room in rooms:

    print("\nRoom:", room)

    for action, value in q_table[room].items():
        print(action, ":", round(value, 2))

    best_action = max(
        q_table[room],
        key=q_table[room].get
    )

    print("Best Action:", best_action)

print("\nTraining Completed Successfully")