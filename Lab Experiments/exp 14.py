# Experiment 14
# A2C and A3C for Smart Elevator Scheduling

import random
import torch
import torch.nn as nn
import torch.optim as optim


# --------------------------------
# Elevator Environment
# --------------------------------

FLOORS = 5

# Actions
UP = 0
DOWN = 1
STAY = 2

action_names = ["UP", "DOWN", "STAY"]


def move_elevator(current_floor, action):

    if action == UP:
        current_floor = min(current_floor + 1, FLOORS - 1)

    elif action == DOWN:
        current_floor = max(current_floor - 1, 0)

    return current_floor


# --------------------------------
# Actor-Critic Network
# --------------------------------

class ActorCritic(nn.Module):

    def __init__(self):

        super().__init__()

        self.common = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU()
        )

        # Actor selects action
        self.actor = nn.Linear(32, 3)

        # Critic estimates state value
        self.critic = nn.Linear(32, 1)

    def forward(self, state):

        x = self.common(state)

        action_probabilities = torch.softmax(
            self.actor(x),
            dim=-1
        )

        state_value = self.critic(x)

        return action_probabilities, state_value


# --------------------------------
# Create A2C Model
# --------------------------------

model = ActorCritic()

optimizer = optim.Adam(
    model.parameters(),
    lr=0.01
)

gamma = 0.9

episodes = 500


print("Training Smart Elevator using A2C...")


# --------------------------------
# A2C Training
# --------------------------------

for episode in range(episodes):

    elevator_floor = random.randint(0, 4)

    passenger_floor = random.randint(0, 4)

    for step in range(20):

        # State:
        # elevator position + passenger position

        state = torch.tensor(
            [
                elevator_floor / 4,
                passenger_floor / 4
            ],
            dtype=torch.float32
        )

        probabilities, value = model(state)

        distribution = torch.distributions.Categorical(
            probabilities
        )

        action = distribution.sample()

        log_probability = distribution.log_prob(
            action
        )

        # Move elevator
        next_floor = move_elevator(
            elevator_floor,
            action.item()
        )

        # Reward
        if next_floor == passenger_floor:

            reward = 10.0
            done = True

        else:

            reward = -1.0
            done = False


        next_state = torch.tensor(
            [
                next_floor / 4,
                passenger_floor / 4
            ],
            dtype=torch.float32
        )


        with torch.no_grad():

            _, next_value = model(
                next_state
            )


        # TD Target
        if done:

            target = torch.tensor([reward])

        else:

            target = (
                torch.tensor([reward])
                + gamma * next_value
            )


        # Advantage
        advantage = target - value


        # Actor loss
        actor_loss = (
            -log_probability
            * advantage.detach()
        )


        # Critic loss
        critic_loss = advantage.pow(2)


        # Total A2C loss
        loss = (
            actor_loss
            + 0.5 * critic_loss
        ).mean()


        optimizer.zero_grad()

        loss.backward()

        optimizer.step()


        elevator_floor = next_floor


        if done:
            break


    if (episode + 1) % 100 == 0:

        print(
            f"Episode {episode + 1}/{episodes} completed"
        )


print("\nA2C Training Completed!")


# --------------------------------
# Test Elevator
# --------------------------------

elevator_floor = 0
passenger_floor = 4

path = [elevator_floor]


for step in range(10):

    state = torch.tensor(
        [
            elevator_floor / 4,
            passenger_floor / 4
        ],
        dtype=torch.float32
    )

    with torch.no_grad():

        probabilities, value = model(state)

        action = torch.argmax(
            probabilities
        ).item()


    elevator_floor = move_elevator(
        elevator_floor,
        action
    )

    path.append(elevator_floor)


    if elevator_floor == passenger_floor:
        break


# --------------------------------
# Output
# --------------------------------

print("\n" + "=" * 55)

print("SMART ELEVATOR - ACTOR CRITIC")

print("=" * 55)

print("\nElevator Start Floor : 0")

print("Passenger Floor      : 4")

print("\nElevator Path:")

print(
    " -> ".join(
        map(str, path)
    )
)


if elevator_floor == passenger_floor:

    print(
        "\nSUCCESS: Elevator reached the passenger!"
    )

    print(
        "Waiting Time:",
        len(path) - 1,
        "steps"
    )

else:

    print(
        "\nElevator failed to reach the passenger."
    )


print("\nActor-Critic Components:")

print("Actor  : Selects the elevator action")

print("Critic : Evaluates the selected action")

print("A2C    : Synchronous Actor-Critic")

print("A3C    : Uses multiple asynchronous agents")

print("\nExperiment completed successfully.")