# Experiment 5
# Epsilon-Greedy Multi-Armed Bandit for Advertisement Selection

import random

# Advertisements
ads = ["Ad A", "Ad B", "Ad C"]

# Actual click probabilities
click_probabilities = [0.2, 0.5, 0.7]

# Estimated reward of each ad
estimated_rewards = [0.0, 0.0, 0.0]

# Number of times each ad is selected
counts = [0, 0, 0]

# Epsilon value
epsilon = 0.1

# Number of trials
trials = 1000

total_reward = 0


for trial in range(trials):

    # Exploration
    if random.random() < epsilon:
        selected_ad = random.randint(0, len(ads) - 1)

    # Exploitation
    else:
        selected_ad = estimated_rewards.index(
            max(estimated_rewards)
        )

    # Simulate user click
    if random.random() < click_probabilities[selected_ad]:
        reward = 1
    else:
        reward = 0

    total_reward += reward

    # Update selection count
    counts[selected_ad] += 1

    # Update estimated reward
    n = counts[selected_ad]

    estimated_rewards[selected_ad] += (
        reward - estimated_rewards[selected_ad]
    ) / n


# Display results
print("=" * 50)
print("EPSILON-GREEDY AD SELECTION")
print("=" * 50)

for i in range(len(ads)):

    print("\nAdvertisement:", ads[i])
    print("Selected:", counts[i], "times")
    print(
        "Estimated Reward:",
        round(estimated_rewards[i], 3)
    )


best_ad = estimated_rewards.index(
    max(estimated_rewards)
)

print("\nBest Advertisement:", ads[best_ad])
print("Total Reward:", total_reward)
print("\nExperiment completed successfully.")