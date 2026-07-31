# Experiment 19
# Multi-Agent Reinforcement Learning (MARL)
# Warehouse Robot Coordination

import random

# --------------------------------
# 1. Warehouse Environment
# --------------------------------

GRID_SIZE = 5

# Two robots
robots = ["Robot 1", "Robot 2"]

# Starting positions
start_positions = {
    "Robot 1": (0, 0),
    "Robot 2": (0, 4)
}

# Target positions
targets = {
    "Robot 1": (4, 4),
    "Robot 2": (4, 0)
}

# Actions
actions = ["UP", "DOWN", "LEFT", "RIGHT"]


# --------------------------------
# 2. Move Robot
# --------------------------------

def move(position, action):

    row, col = position

    if action == "UP":
        row = max(0, row - 1)

    elif action == "DOWN":
        row = min(GRID_SIZE - 1, row + 1)

    elif action == "LEFT":
        col = max(0, col - 1)

    elif action == "RIGHT":
        col = min(GRID_SIZE - 1, col + 1)

    return row, col


# --------------------------------
# 3. Distance to Target
# --------------------------------

def distance(position, target):

    return (
        abs(position[0] - target[0])
        + abs(position[1] - target[1])
    )


# --------------------------------
# 4. Select Best Action
# --------------------------------

def choose_action(position, target):

    # Exploration
    if random.random() < 0.1:
        return random.choice(actions)

    best_action = None
    best_distance = float("inf")

    for action in actions:

        new_position = move(
            position,
            action
        )

        new_distance = distance(
            new_position,
            target
        )

        if new_distance < best_distance:

            best_distance = new_distance
            best_action = action

    return best_action


# --------------------------------
# 5. Start Simulation
# --------------------------------

positions = start_positions.copy()

paths = {
    robot: [positions[robot]]
    for robot in robots
}

print("=" * 60)
print("MULTI-AGENT RL - WAREHOUSE ROBOT COORDINATION")
print("=" * 60)

print("\nInitial Positions:")

for robot in robots:

    print(
        robot,
        ":",
        positions[robot],
        "Target:",
        targets[robot]
    )


# --------------------------------
# 6. Multi-Agent Coordination
# --------------------------------

for step in range(30):

    print(f"\nStep {step + 1}")

    new_positions = positions.copy()

    for robot in robots:

        # Robot already reached target
        if positions[robot] == targets[robot]:

            print(
                robot,
                "already reached target."
            )

            continue

        action = choose_action(
            positions[robot],
            targets[robot]
        )

        proposed_position = move(
            positions[robot],
            action
        )

        # Check collision with other robot
        other_robot = (
            "Robot 2"
            if robot == "Robot 1"
            else "Robot 1"
        )

        if proposed_position == new_positions[other_robot]:

            print(
                robot,
                "avoided collision at",
                proposed_position
            )

            proposed_position = positions[robot]

        else:

            print(
                robot,
                "Action:",
                action,
                "->",
                proposed_position
            )

        new_positions[robot] = proposed_position

        paths[robot].append(
            proposed_position
        )

    positions = new_positions


    # Check if all robots reached targets
    if all(
        positions[robot] == targets[robot]
        for robot in robots
    ):

        break


# --------------------------------
# 7. Results
# --------------------------------

print("\n" + "=" * 60)

print("FINAL RESULTS")

print("=" * 60)


for robot in robots:

    print(
        "\n",
        robot,
        "Path:",
        sep=""
    )

    print(
        " -> ".join(
            str(position)
            for position in paths[robot]
        )
    )

    if positions[robot] == targets[robot]:

        print(
            robot,
            "successfully reached its target!"
        )

    else:

        print(
            robot,
            "did not reach its target."
        )


print(
    "\nMulti-agent coordination completed successfully."
)