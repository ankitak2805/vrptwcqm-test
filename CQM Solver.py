import os
import dimod
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dimod import ConstrainedQuadraticModel, quicksum, Binary, Real
from dwave.system import LeapHybridCQMSampler

# input data from files
distances_df = pd.read_csv('Input/travel_times.csv', index_col=0)
customers_df = pd.read_csv('Input/customers.csv')
vehicles_df = pd.read_csv('Input/vehicles.csv')

# assign parameters
distances = distances_df.to_numpy(dtype=float)
num_customers = len(customers_df) + 1 
num_vehicles = len(vehicles_df)
time_windows = list(zip(customers_df['start_time'], customers_df['end_time']))
service_times = list(customers_df['service_time'])

# check demand & capacity
total_demand = customers_df['demand'].sum()
total_capacity = vehicles_df['capacity'].sum()
print(f"Total demand: {total_demand}, Total vehicle capacity: {total_capacity}")
if total_demand > total_capacity:
    print("Warning: Total demand exceeds vehicle capacity, making feasibility unlikely.")

# Initialize CQM
cqm = ConstrainedQuadraticModel()

# decision variables
x = {}
for k in range(num_vehicles):
    for i in range(num_customers):
        for j in range(num_customers):
            if i != j:
                x[i, j, k] = Binary(f'x_{i}_{j}_{k}')


start_times = {i: Real(f'start_time_{i}', lower_bound=0, upper_bound=max(customers_df['end_time'])) for i in range(1, num_customers)}

# objective function: minimize the travel time
objective = dimod.BinaryQuadraticModel({}, {}, 0.0, 'BINARY')

# Add linear terms
for i in range(num_customers):
    for j in range(num_customers):
        for k in range(num_vehicles):
            if i != j and i < distances.shape[0] and j < distances.shape[1]:
                objective.add_linear(f'x_{i}_{j}_{k}', distances[i, j])

# Constraints
for k in range(num_vehicles):
    lhs = quicksum(x[i, j, k] for i in range(num_customers) for j in range(num_customers) if i != j)
    cqm.add_constraint(lhs <= num_customers, label=f'vehicle_usage_constraint_upper_{k}')
    cqm.add_constraint(lhs >= 1, label=f'vehicle_usage_constraint_lower_{k}')

# Set objective function
cqm.set_objective(objective)

# Each customer is visited exactly once
for j in range(1, num_customers):
    cqm.add_constraint(
        quicksum(x[i, j, k] for i in range(num_customers) for k in range(num_vehicles) if i != j) == 1,
        label=f"visit_customer_{j}"
    )

# Each vehicle leaves the depot and returns
for k in range(num_vehicles):
    cqm.add_constraint(
        quicksum(x[0, j, k] for j in range(1, num_customers)) == 1,
        label=f"leave_depot_{k+1}"
    )
    cqm.add_constraint(
        quicksum(x[i, 0, k] for i in range(1, num_customers)) == 1,
        label=f"return_depot_{k+1}"
    )

# Capacity constraint
for k in range(num_vehicles):
    cqm.add_constraint(
        quicksum(
            customers_df['demand'][j-1] * quicksum(x[i, j, k] for i in range(1, num_customers) if i != j)
            for j in range(1, num_customers)
        ) <= vehicles_df['capacity'][k],
        label=f"capacity_vehicle_{k+1}"
    )

# Time window
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

# Extract solution
if len(feasible_sampleset) > 0:
    solution = feasible_sampleset.first.sample
    print("Feasible solutions found!")
else:
    print("No feasible solution found")
    exit()

# Extract routes
routes = {k: [] for k in range(num_vehicles)}
for k in range(num_vehicles):
    current_node = 0
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
    routes[k].append(0)  

# missing customers
all_visited = set()
for k in range(num_vehicles):
    all_visited.update(routes[k])

missing_customers = set(range(1, num_customers)) - all_visited
if missing_customers:
    print(f"Customers not covered: {missing_customers}")
    for missing in missing_customers:
        for k in range(num_vehicles):
            current_capacity = sum(customers_df['demand'][i-1] for i in routes[k] if i > 0)
            if current_capacity + customers_df['demand'][missing-1] <= vehicles_df['capacity'][k]:
                routes[k].insert(-1, missing)  
                print(f"Assigned customer {missing} to vehicle {k+1}")
                break
    print(f"Reassigned missing customers: {missing_customers}")
else:
    print("All customers are covered!")

# debug routes and capacities
for k, route in routes.items():
    capacity_used = sum(customers_df['demand'][i-1] for i in route if i > 0)
    print(f"Route for Vehicle {k+1}: {route}, Capacity Used: {capacity_used}")

# Check coverage
final_covered_customers = set()
for route in routes.values():
    final_covered_customers.update(route)

missing_customers_final = set(range(1, num_customers)) - final_covered_customers
if missing_customers_final:
    print(f"Final missing customers after adjustment: {missing_customers_final}")
else:
    print("All customers are covered in the final routes!")

# Plot routes
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
