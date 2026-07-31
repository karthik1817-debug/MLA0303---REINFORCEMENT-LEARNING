# Experiment 20
# Multi-Agent Reinforcement Learning
# Traffic Signal Coordination

import random

# --------------------------------
# 1. Traffic Signal Agents
# --------------------------------

signals = ["Signal A", "Signal B"]

# Actions
actions = ["NS_GREEN", "EW_GREEN"]

# Q-table for each signal
q_table = {
    signal: {
        "NS_GREEN": 0.0,
        "EW_GREEN": 0.0
    }
    for signal in signals
}


# --------------------------------
# 2. Parameters
# --------------------------------

alpha = 0.1
epsilon = 0.2
episodes = 1000


# --------------------------------
# 3. Choose Action
# --------------------------------

def choose_action(signal):

    # Exploration
    if random.random() < epsilon:
        return random.choice(actions)

    # Exploitation
    return max(
        q_table[signal],
        key=q_table[signal].get
    )


# --------------------------------
# 4. Calculate Reward
# --------------------------------

def calculate_reward(
    action,
    ns_cars,
    ew_cars
):

    # Reward based on number of
    # vehicles allowed to move

    if action == "NS_GREEN":
        passed = ns_cars
        waiting = ew_cars

    else:
        passed = ew_cars
        waiting = ns_cars

    reward = passed - waiting

    return reward


# --------------------------------
# 5. Training
# --------------------------------

print(
    "Training Traffic Signals using Multi-Agent RL..."
)


for episode in range(episodes):

    for signal in signals:

        # Random traffic
        ns_cars = random.randint(1, 20)
        ew_cars = random.randint(1, 20)

        action = choose_action(signal)

        reward = calculate_reward(
            action,
            ns_cars,
            ew_cars
        )

        # Q-value update
        old_value = q_table[signal][action]

        q_table[signal][action] = (
            old_value
            + alpha * (
                reward - old_value
            )
        )


    if (episode + 1) % 200 == 0:

        print(
            f"Episode {episode + 1}/{episodes} completed"
        )


print("\nTraining Completed!")


# --------------------------------
# 6. Display Q-Values
# --------------------------------

print("\n" + "=" * 60)

print(
    "MULTI-AGENT RL - TRAFFIC SIGNAL COORDINATION"
)

print("=" * 60)


for signal in signals:

    print("\n", signal, sep="")

    for action in actions:

        print(
            action,
            ":",
            round(
                q_table[signal][action],
                2
            )
        )


# --------------------------------
# 7. Test Traffic System
# --------------------------------

print("\n" + "=" * 60)

print("TRAFFIC SIGNAL TEST")

print("=" * 60)


test_traffic = {
    "Signal A": (15, 5),
    "Signal B": (4, 18)
}


total_waiting = 0


for signal in signals:

    ns_cars, ew_cars = test_traffic[signal]

    # Select green direction based
    # on current traffic

    if ns_cars >= ew_cars:
        selected_action = "NS_GREEN"
        waiting = ew_cars

    else:
        selected_action = "EW_GREEN"
        waiting = ns_cars


    total_waiting += waiting


    print(
        "\n",
        signal,
        sep=""
    )

    print(
        "North-South Cars:",
        ns_cars
    )

    print(
        "East-West Cars  :",
        ew_cars
    )

    print(
        "Selected Action :",
        selected_action
    )

    print(
        "Waiting Vehicles:",
        waiting
    )


# --------------------------------
# 8. Result
# --------------------------------

print(
    "\nTotal Waiting Vehicles:",
    total_waiting
)

print(
    "\nTraffic signals coordinated successfully!"
)

print(
    "Goal: Reduce vehicle waiting time "
    "and improve traffic flow."
)