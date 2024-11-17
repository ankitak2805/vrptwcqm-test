import os
import dimod
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dimod import ConstrainedQuadraticModel, quicksum, Binary, Real
from dwave.system import LeapHybridCQMSampler

# input data
distances_df = pd.read_csv('Input/travel_times.csv', index_col=0)
customers_df = pd.read_csv('Input/customers.csv')
vehicles_df = pd.read_csv('Input/vehicles.csv')

# convert distance to float
distances = distances_df.to_numpy(dtype=float)

# assign parameters
num_customers = len(customers_df) + 1  
num_vehicles = len(vehicles_df)
time_windows = list(zip(customers_df['start_time'], customers_df['end_time']))
service_times = list(customers_df['service_time'])

# check demand and capacity
total_demand = customers_df['demand'].sum()
total_capacity = vehicles_df['capacity'].sum()
print(f"Total demand: {total_demand}, Total capacity: {total_capacity}")
if total_demand > total_capacity:
    print("Warning: Total demand exceeds capacity, making unfeasible.")

# initialize CQM
cqm = ConstrainedQuadraticModel()

# define variables
x = {(i, j, k): Binary(f'x_{i}_{j}_{k}') for k in range(num_vehicles) for i in range(num_customers) for j in range(num_customers) if i != j}
start_times = {i: Real(f'start_time_{i}', lower_bound=0, upper_bound=max(customers_df['end_time'])) for i in range(1, num_customers)}

# objective function
objective = dimod.BinaryQuadraticModel({}, {}, 0.0, 'BINARY')
for i in range(num_customers):
    for j in range(num_customers):
        for k in range(num_vehicles):
            if i != j and 0 <= i < distances.shape[0] and 0 <= j < distances.shape[1]:
                objective.add_linear(f'x_{i}_{j}_{k}', distances[i, j])

cqm.set_objective(objective)

# c1 each customer is visited once
for j in range(1, num_customers):
    cqm.add_constraint(quicksum(x[i, j, k] for i in range(num_customers) for k in range(num_vehicles) if i != j) == 1, label=f"visit_customer_{j}")

# c2 vehicles leave and return to the depot
for k in range(num_vehicles):
    cqm.add_constraint(quicksum(x[0, j, k] for j in range(1, num_customers)) == 1, label=f"leave_depot_{k+1}")
    cqm.add_constraint(quicksum(x[i, 0, k] for i in range(1, num_customers)) == 1, label=f"return_depot_{k+1}")

# c3 capacity constraint for vehicles
for k in range(num_vehicles):
    cqm.add_constraint(
        quicksum(customers_df['demand'][j-1] * quicksum(x[i, j, k] for i in range(1, num_customers) if i != j) for j in range(1, num_customers)) <= vehicles_df['capacity'][k],
        label=f"capacity_vehicle_{k+1}"
    )

# c4 Time window constraints with slack
slack = 5
for i in range(1, num_customers):
    lower_bound = time_windows[i-1][0] - slack
    upper_bound = time_windows[i-1][1] + slack
    cqm.add_constraint(start_times[i] >= lower_bound, label=f"time_window_lower_bound_{i}")
    cqm.add_constraint(start_times[i] <= upper_bound, label=f"time_window_upper_bound_{i}")

# Solve CQM
sampler = LeapHybridCQMSampler()
sampleset = sampler.sample_cqm(cqm, label="CVRPTW with Time Windows")
feasible_sampleset = sampleset.filter(lambda d: d.is_feasible)

if len(feasible_sampleset) > 0:
    solution = feasible_sampleset.first.sample
    print("Feasible solutions found!")
else:
    print("No feasible solution found")
    exit()

# extract routes
routes = {k: [] for k in range(num_vehicles)}
for k in range(num_vehicles):
    current_node = 0  # Start from the depot
    visited = set([0])
    while len(visited) < num_customers:
        next_node = None
        for j in range(num_customers):
            if current_node != j and solution.get(f'x_{current_node}_{j}_{k}', 0) and j not in visited:
                next_node = j
                break
        if next_node is None:
            break
        routes[k].append(next_node)
        visited.add(next_node)
        current_node = next_node
    routes[k].append(0)  # Return to depot

# handle missing customers
missing_customers = set(range(1, num_customers)) - set().union(*[set(route) for route in routes.values()])
if missing_customers:
    print(f"Customers not covered: {missing_customers}")
    for missing in sorted(missing_customers): 
        best_vehicle = None
        best_insert_position = None
        min_additional_distance = float('inf')

        for k in range(num_vehicles):
            current_capacity = sum(customers_df['demand'][i-1] for i in routes[k] if i > 0)
            if current_capacity + customers_df['demand'][missing-1] <= vehicles_df['capacity'][k]:
                # Ensure ascending order insertion
                for idx in range(1, len(routes[k])):
                    prev_node = routes[k][idx - 1]
                    next_node = routes[k][idx] if idx < len(routes[k]) else 0

                    prev_idx = max(0, prev_node - 1) if prev_node > 0 else 0
                    next_idx = max(0, next_node - 1) if next_node > 0 else 0
                    missing_idx = max(0, missing - 1)

                    # check for valid insertion to maintain ascending order
                    if (prev_node < missing < next_node) or (idx == len(routes[k]) - 1):  # Allow at end if higher
                        if all(0 <= index < distances.shape[0] for index in [prev_idx, next_idx, missing_idx]):
                            additional_distance = (distances[prev_idx, missing_idx] + distances[missing_idx, next_idx] - distances[prev_idx, next_idx])
                            if additional_distance < min_additional_distance:
                                min_additional_distance = additional_distance
                                best_vehicle = k
                                best_insert_position = idx

        if best_vehicle is not None and best_insert_position is not None:
            routes[best_vehicle].insert(best_insert_position, missing)
            print(f"Assigned customer {missing} to vehicle {best_vehicle+1} at position {best_insert_position}")
        else:
            print(f"Unable to assign customer {missing} within constraints")

# final route check
for k, route in routes.items():
    route = sorted(route[:-1]) + [0]
    routes[k] = route
    capacity_used = sum(customers_df['demand'][i-1] for i in route if i > 0)
    print(f"Vehicle {k+1} route: {route}, Capacity used: {capacity_used}")

# Plot
plt.figure(figsize=(10, 6))
plt.plot(0, 0, 'ro', markersize=10, label='Depot')
for i in range(1, num_customers):
    plt.plot(customers_df['x'][i-1], customers_df['y'][i-1], 'bo', markersize=7)
    plt.text(customers_df['x'][i-1] + 0.5, customers_df['y'][i-1] + 0.5, f'Customer {i}', fontsize=9)

colors = ['b', 'g', 'r', 'c', 'm', 'y']
for k, route in routes.items():
    route_coords = [(0, 0)] + [
        (customers_df['x'][i-1], customers_df['y'][i-1]) 
        for i in route[:-1] if i > 0 and i <= len(customers_df)
    ] + [(0, 0)]
    route_x, route_y = zip(*route_coords)
    plt.plot(route_x, route_y, colors[k % len(colors)] + '-', linewidth=2, label=f'Vehicle {k+1} Route')

    # Add labels
    for i in route[:-1]:
        if i > 0 and i <= len(customers_df):
            plt.text(customers_df['x'][i-1] + 0.5, customers_df['y'][i-1] + 0.5, f'Customer {i}', fontsize=9)
plt.gca().axes.xaxis.set_visible(False)
plt.gca().axes.yaxis.set_visible(False)

plt.title('CVRPTW solution')
plt.legend()
plt.savefig('output/routes.png')
plt.show()
