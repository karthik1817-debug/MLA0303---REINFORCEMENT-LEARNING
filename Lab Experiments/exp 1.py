# EXPERIMENT 1
# Markov Decision Process (MDP) for Simplified Chess

# 1. Define the states
states = ["Start", "Attack", "Defend", "Win", "Lose"]

# 2. Define possible actions
actions = {
    "Start": ["Attack", "Defend"],
    "Attack": ["Strong Attack", "Safe Move"],
    "Defend": ["Counter Attack", "Continue Defend"]
}

# 3. Define transition probabilities
transitions = {
    ("Start", "Attack"): {
        "Attack": 0.8,
        "Lose": 0.2
    },

    ("Start", "Defend"): {
        "Defend": 0.9,
        "Lose": 0.1
    },

    ("Attack", "Strong Attack"): {
        "Win": 0.7,
        "Lose": 0.3
    },

    ("Attack", "Safe Move"): {
        "Win": 0.6,
        "Lose": 0.4
    },

    ("Defend", "Counter Attack"): {
        "Win": 0.65,
        "Lose": 0.35
    },

    ("Defend", "Continue Defend"): {
        "Win": 0.5,
        "Lose": 0.5
    }
}

# 4. Define rewards
rewards = {
    "Start": 0,
    "Attack": 2,
    "Defend": 1,
    "Win": 10,
    "Lose": -10
}


# 5. Calculate expected reward
def calculate_reward(state, action):

    total_reward = 0

    for next_state, probability in transitions[(state, action)].items():

        reward = rewards[next_state]

        total_reward += probability * reward

    return total_reward


# 6. Find the best action
def find_best_action(state):

    best_action = None
    best_reward = float("-inf")

    print("\nState:", state)

    for action in actions[state]:

        reward = calculate_reward(state, action)

        print(
            "Action:",
            action,
            "| Expected Reward:",
            round(reward, 2)
        )

        if reward > best_reward:
            best_reward = reward
            best_action = action

    return best_action, best_reward


# 7. Main program
print("=" * 50)
print("MDP - SIMPLIFIED CHESS GAME")
print("=" * 50)

current_state = "Start"

print("\nInitial State:", current_state)

# Find first optimal action
best_action, reward = find_best_action(current_state)

print("\nBest Action:", best_action)
print("Expected Reward:", round(reward, 2))


# Change state according to selected action
if best_action == "Attack":
    current_state = "Attack"

elif best_action == "Defend":
    current_state = "Defend"


# Find second optimal action
best_action, reward = find_best_action(current_state)

print("\nBest Action:", best_action)
print("Expected Reward:", round(reward, 2))

print("\nOptimal Move Sequence Completed")
