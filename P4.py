import pandas as pd
import numpy as np

def read_transportation_problem(file_path):
    df = pd.read_csv(file_path, sep=';', header=None)
    supply = df.iloc[1:, 0].values.astype(float)
    demand = df.iloc[0, 1:].values.astype(float)
    cost_matrix = df.iloc[1:, 1:].values.astype(float)
    return cost_matrix, supply, demand

def northwest_corner_method(cost, supply, demand):
    supply_copy = supply.copy()
    demand_copy = demand.copy()
    allocation = np.zeros_like(cost)

    i, j = 0, 0
    while i < len(supply) and j < len(demand):
        allocated = min(supply_copy[i], demand_copy[j])
        allocation[i, j] = allocated
        supply_copy[i] -= allocated
        demand_copy[j] -= allocated

        if supply_copy[i] <= 0:
            i += 1
        if demand_copy[j] <= 0:
            j += 1

    return allocation

def minimum_cost_method(cost, supply, demand):
    supply_copy = supply.copy()
    demand_copy = demand.copy()
    allocation = np.zeros_like(cost)

    while np.sum(supply_copy) > 0 and np.sum(demand_copy) > 0:
        min_cost = np.inf
        min_i, min_j = -1, -1

        for i in range(len(supply)):
            for j in range(len(demand)):
                if supply_copy[i] > 0 and demand_copy[j] > 0 and cost[i, j] < min_cost:
                    min_cost = cost[i, j]
                    min_i, min_j = i, j

        allocated = min(supply_copy[min_i], demand_copy[min_j])
        allocation[min_i, min_j] = allocated
        supply_copy[min_i] -= allocated
        demand_copy[min_j] -= allocated

    return allocation

def minimum_row_cost_method(cost, supply, demand):
    supply_copy = supply.copy()
    demand_copy = demand.copy()
    allocation = np.zeros_like(cost)

    for i in range(len(supply)):
        while supply_copy[i] > 0:
            min_cost = np.inf
            min_j = -1

            for j in range(len(demand)):
                if demand_copy[j] > 0 and cost[i, j] < min_cost:
                    min_cost = cost[i, j]
                    min_j = j

            allocated = min(supply_copy[i], demand_copy[min_j])
            allocation[i, min_j] = allocated
            supply_copy[i] -= allocated
            demand_copy[min_j] -= allocated

    return allocation
def vogels_method(cost, supply, demand):
    supply_copy = supply.copy()
    demand_copy = demand.copy()
    allocation = np.zeros_like(cost)

    while np.sum(supply_copy) > 0 and np.sum(demand_copy) > 0:
        row_penalties = []
        for i in range(len(supply)):
            if supply_copy[i] <= 0:
                row_penalties.append(-1)
                continue
            row_costs = [cost[i, j] for j in range(len(demand)) if demand_copy[j] > 0]
            if len(row_costs) < 2:
                row_penalties.append(0)
            else:
                row_costs.sort()
                row_penalties.append(row_costs[1] - row_costs[0])

        col_penalties = []
        for j in range(len(demand)):
            if demand_copy[j] <= 0:
                col_penalties.append(-1)
                continue
            col_costs = [cost[i, j] for i in range(len(supply)) if supply_copy[i] > 0]
            if len(col_costs) < 2:
                col_penalties.append(0)
            else:
                col_costs.sort()
                col_penalties.append(col_costs[1] - col_costs[0])

        max_row_penalty = max(row_penalties)
        max_col_penalty = max(col_penalties)

        if max_row_penalty >= max_col_penalty:
            i = row_penalties.index(max_row_penalty)
            min_cost = np.inf
            min_j = -1
            for j in range(len(demand)):
                if demand_copy[j] > 0 and cost[i, j] < min_cost:
                    min_cost = cost[i, j]
                    min_j = j
        else:
            j = col_penalties.index(max_col_penalty)
            min_cost = np.inf
            min_i = -1
            for i in range(len(supply)):
                if supply_copy[i] > 0 and cost[i, j] < min_cost:
                    min_cost = cost[i, j]
                    min_i = i
            i = min_i
            min_j = j

        allocated = min(supply_copy[i], demand_copy[min_j])
        allocation[i, min_j] = allocated
        supply_copy[i] -= allocated
        demand_copy[min_j] -= allocated

    return allocation

def calculate_total_cost(allocation, cost):
    return np.sum(allocation * cost)


if __name__ == "__main__":
    file_path = "transportation_problem.csv"
    cost, supply, demand = read_transportation_problem(file_path)

    methods = {
        "Northwest Corner Rule": northwest_corner_method,
        "Minimum Cost Method": minimum_cost_method,
        "Minimum Row Cost Method": minimum_row_cost_method,
        "Vogel's Method": vogels_method
    }

    print("\nInitial feasible solutions using different methods:")
    print("-" * 50)

    for method_name, method_func in methods.items():
        allocation = method_func(cost, supply, demand)
        total_cost = calculate_total_cost(allocation, cost)
        print(f"\n{method_name}:")
        print(f"Allocation:\n{allocation}")
        print(f"Total Cost: {total_cost}")
