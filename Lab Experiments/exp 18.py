# Experiment 18
# Meta-Reinforcement Learning for Industrial Robot

import random

# --------------------------------
# 1. Manufacturing Tasks
# --------------------------------

tasks = {
    "Assembly": {
        "Pick": 5,
        "Place": 10,
        "Inspect": 2
    },

    "Packaging": {
        "Pick": 5,
        "Place": 4,
        "Inspect": 10
    },

    "Sorting": {
        "Pick": 10,
        "Place": 3,
        "Inspect": 5
    }
}

actions = ["Pick", "Place", "Inspect"]


# --------------------------------
# 2. Meta Knowledge
# --------------------------------

# Knowledge shared between tasks

meta_q = {
    action: 0.0
    for action in actions
}


# --------------------------------
# 3. Parameters
# --------------------------------

alpha = 0.2
epsilon = 0.2

episodes = 500


# --------------------------------
# 4. Select Action
# --------------------------------

def choose_action(q_values):

    # Exploration
    if random.random() < epsilon:

        return random.choice(actions)

    # Exploitation
    return max(
        q_values,
        key=q_values.get
    )


# --------------------------------
# 5. Meta Training
# --------------------------------

print("Training Industrial Robot using Meta-RL...")


for episode in range(episodes):

    # Select random manufacturing task
    task_name = random.choice(
        list(tasks.keys())
    )

    rewards = tasks[task_name]

    # Start with shared meta knowledge
    task_q = meta_q.copy()


    # Learn current task
    for step in range(10):

        action = choose_action(task_q)

        reward = rewards[action]

        # Update task knowledge
        task_q[action] += alpha * (
            reward - task_q[action]
        )


    # Update shared meta knowledge
    for action in actions:

        meta_q[action] += 0.05 * (
            task_q[action] - meta_q[action]
        )


    if (episode + 1) % 100 == 0:

        print(
            f"Episode {episode + 1}/{episodes} completed"
        )


print("\nMeta Training Completed!")


# --------------------------------
# 6. Display Meta Knowledge
# --------------------------------

print("\n" + "=" * 55)

print("META-RL - INDUSTRIAL ROBOT")

print("=" * 55)


print("\nShared Meta Knowledge:")

for action, value in meta_q.items():

    print(
        action,
        ":",
        round(value, 2)
    )


# --------------------------------
# 7. New Manufacturing Task
# --------------------------------

print("\nAdapting to a NEW task...")

new_task = {
    "Pick": 4,
    "Place": 6,
    "Inspect": 12
}

# Robot begins with previously
# learned meta knowledge

new_q = meta_q.copy()


# --------------------------------
# 8. Fast Adaptation
# --------------------------------

for step in range(20):

    action = choose_action(new_q)

    reward = new_task[action]

    new_q[action] += alpha * (
        reward - new_q[action]
    )


# --------------------------------
# 9. Result
# --------------------------------

print("\nNew Task Q-Values:")

for action, value in new_q.items():

    print(
        action,
        ":",
        round(value, 2)
    )


best_action = max(
    new_q,
    key=new_q.get
)


print(
    "\nBest Action for New Task:",
    best_action
)


print(
    "Reward:",
    new_task[best_action]
)


print(
    "\nRobot successfully adapted "
    "to the new manufacturing task!"
)

print(
    "\nMeta-RL allows the robot to use "
    "knowledge from previous tasks to "
    "learn new tasks faster."
)