# Experiment 7
# Dynamic Programming for Taxi Route Optimization

# Locations
locations = ["A", "B", "C", "D", "Destination"]

# Possible routes and travel costs
routes = {
    "A": {"B": 4, "C": 2},
    "B": {"C": 1, "D": 5},
    "C": {"B": 1, "D": 3, "Destination": 7},
    "D": {"Destination": 2},
    "Destination": {}
}

# Store minimum cost from each location to destination
cost = {
    location: float("inf")
    for location in locations
}

# Destination has zero remaining cost
cost["Destination"] = 0

# Store best next location
best_next = {}

# Dynamic Programming / Value Iteration
for iteration in range(10):

    for current in locations:

        if current == "Destination":
            continue

        for next_location, travel_cost in routes[current].items():

            new_cost = travel_cost + cost[next_location]

            if new_cost < cost[current]:

                cost[current] = new_cost
                best_next[current] = next_location


# Display minimum costs
print("=" * 55)
print("TAXI ROUTE OPTIMIZATION USING DYNAMIC PROGRAMMING")
print("=" * 55)

print("\nMinimum Cost to Destination:")

for location in locations:
    print(location, ":", cost[location])


# Find optimal path
print("\nOptimal Taxi Route:")

current = "A"

path = [current]

while current != "Destination":

    if current not in best_next:
        print("No route available.")
        break

    current = best_next[current]

    path.append(current)


print(" -> ".join(path))

print("\nMinimum Travel Cost:", cost["A"])

print("\nTaxi reached the destination using the optimal route.")