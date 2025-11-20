import pandas as pd
import numpy as np
import random
import copy
import time
import math

# === PARAMETER DASAR ===
NUM_TRUCKS = 5
TRUCK_CAPACITY = 2200
DEPOT = 0
TPA = 53
MAX_ITER = 1000
REMOVE_FRAC = 0.2
RANDOM_SEED = 42
start_time = time.time()
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# === MUAT DATA ===
distance_df = pd.read_excel("data.xlsx", sheet_name="jarak", index_col=0)
demand_df = pd.read_excel("data.xlsx", sheet_name="demand", index_col=0)

distance_matrix = distance_df.to_numpy()
demands = demand_df["Demand"].to_dict()
full_demands = {i: demands.get(i, 0) for i in range(54)}
customer_nodes = list(range(1, 53 + 1))

# === FUNGSI BANTU ===
def route_cost(route):
    return sum(distance_matrix[route[i]][route[i+1]] for i in range(len(route)-1))

def route_load(route):
    return sum(full_demands[n] for n in route if n not in [DEPOT, TPA])

def total_cost(solution):
    return sum(route_cost(r) for r in solution)

def is_feasible(route):
    load = 0
    for node in route:
        if node == TPA:
            load = 0
        else:
            load += full_demands[node]
            if load > TRUCK_CAPACITY:
                return False
    return route.count(TPA) == 1 and route[-2] == TPA and route[-1] == DEPOT

def finalize_route(route):
    route = [n for n in route if n != TPA and n != DEPOT]
    if not route:
        return [DEPOT, TPA, DEPOT]
    route = [DEPOT] + route + [TPA, DEPOT]
    return route

# === SOLUSI AWAL ===
def greedy_initial_solution():
    unvisited = set(customer_nodes)
    solution = []
    for _ in range(NUM_TRUCKS):
        route = [DEPOT]
        load = 0
        while unvisited:
            feasible = [n for n in unvisited if load + full_demands[n] <= TRUCK_CAPACITY]
            if not feasible:
                break
            nearest = min(feasible, key=lambda n: distance_matrix[route[-1]][n])
            route.append(nearest)
            load += full_demands[nearest]
            unvisited.remove(nearest)
        route = finalize_route(route)
        solution.append(route)
    return solution

# === OPERATOR DESTROY ===
def random_removal(solution, num_remove):
    flat = [n for r in solution for n in r if n not in [DEPOT, TPA]]
    removed = random.sample(flat, min(num_remove, len(flat)))
    new_solution = []
    for r in solution:
        filtered = [n for n in r if n not in removed]
        new_solution.append(finalize_route(filtered))
    return new_solution, removed

def worst_removal(solution, num_remove):
    node_saving = {}
    for r in solution:
        for i in range(1, len(r)-2):
            node = r[i]
            if node in [DEPOT, TPA]: continue
            saving = distance_matrix[r[i-1]][node] + distance_matrix[node][r[i+1]] - distance_matrix[r[i-1]][r[i+1]]
            node_saving[node] = saving
    sorted_nodes = sorted(node_saving, key=node_saving.get, reverse=True)
    removed = sorted_nodes[:num_remove]
    return random_removal(solution, len(removed))

# === OPERATOR REPAIR ===
def greedy_insert(solution, removed):
    for node in removed:
        best_cost = float('inf')
        best_pos = (0, 0)
        for r_idx, route in enumerate(solution):
            for i in range(1, len(route)-2):
                trial = route[:i] + [node] + route[i:]
                if is_feasible(trial):
                    c = route_cost(trial)
                    if c < best_cost:
                        best_cost = c
                        best_pos = (r_idx, i)
        r_idx, i = best_pos
        solution[r_idx].insert(i, node)
        solution[r_idx] = finalize_route(solution[r_idx])
    return solution

# === LOCAL SEARCH: 2-OPT, RELOCATE, SWAP ===
def two_opt(route):
    best = route
    improved = True
    while improved:
        improved = False
        for i in range(1, len(route) - 3):
            for j in range(i + 1, len(route) - 2):
                if j - i == 1:
                    continue
                new_route = route[:i] + route[i:j][::-1] + route[j:]
                if is_feasible(new_route) and route_cost(new_route) < route_cost(best):
                    best = new_route
                    improved = True
        route = best
    return best

def relocate(solution):
    for i in range(len(solution)):
        for j in range(len(solution)):
            if i == j:
                continue
            for idx, node in enumerate(solution[i][1:-2], start=1):
                new_route_i = solution[i][:idx] + solution[i][idx+1:]
                for k in range(1, len(solution[j]) - 2):
                    trial_route_j = solution[j][:k] + [node] + solution[j][k:]
                    if is_feasible(finalize_route(new_route_i)) and is_feasible(trial_route_j):
                        new_sol = copy.deepcopy(solution)
                        new_sol[i] = finalize_route(new_route_i)
                        new_sol[j] = finalize_route(trial_route_j)
                        if total_cost(new_sol) < total_cost(solution):
                            return new_sol
    return solution

def swap(solution):
    for i in range(len(solution)):
        for j in range(i + 1, len(solution)):
            for idx1, node1 in enumerate(solution[i][1:-2], start=1):
                for idx2, node2 in enumerate(solution[j][1:-2], start=1):
                    new_r1 = solution[i][:]
                    new_r2 = solution[j][:]
                    new_r1[idx1], new_r2[idx2] = new_r2[idx2], new_r1[idx1]
                    if is_feasible(new_r1) and is_feasible(new_r2):
                        new_sol = copy.deepcopy(solution)
                        new_sol[i] = finalize_route(new_r1)
                        new_sol[j] = finalize_route(new_r2)
                        if total_cost(new_sol) < total_cost(solution):
                            return new_sol
    return solution

def local_search(solution):
    improved = True
    while improved:
        improved = False
        for i in range(len(solution)):
            new_route = two_opt(solution[i])
            if route_cost(new_route) < route_cost(solution[i]):
                solution[i] = new_route
                improved = True
        new_sol = relocate(solution)
        if total_cost(new_sol) < total_cost(solution):
            solution = new_sol
            improved = True
        new_sol = swap(solution)
        if total_cost(new_sol) < total_cost(solution):
            solution = new_sol
            improved = True
    return solution

# === ALNS CORE ===
def alns(iter=1000):
    current = greedy_initial_solution()
    current = local_search(current)
    best = copy.deepcopy(current)
    best_cost = total_cost(best)

    destroy_ops = [random_removal, worst_removal]
    repair_op = greedy_insert
    weights = [1 for _ in destroy_ops]
    scores = [0 for _ in destroy_ops]

    for it in range(iter):
        d_idx = random.choices(range(len(destroy_ops)), weights=weights)[0]
        d_op = destroy_ops[d_idx]
        num_remove = int(REMOVE_FRAC * len(customer_nodes))
        destroyed, removed = d_op(current, num_remove)
        repaired = repair_op(destroyed, removed)

        if it % 5 == 0:
            improved = local_search(repaired)
        else:
            improved = repaired

        cost = total_cost(improved)

        if cost < best_cost:
            best = copy.deepcopy(improved)
            best_cost = cost
            scores[d_idx] += 1
        current = improved

        if (it + 1) % 100 == 0:
            for i in range(len(weights)):
                weights[i] = 1 + scores[i]
            scores = [0] * len(destroy_ops)

    return best, best_cost

# === JALANKAN ===
if __name__ == "__main__":
    best_solution, best_obj = alns(MAX_ITER)
    print(f"\nTotal Jarak: {best_obj:.2f} km\n")
    for idx, route in enumerate(best_solution, 1):
        cost = route_cost(route)
        load = route_load(route)
        print(f"Truk {idx}: {route} | Jarak: {cost:.2f} km | Muatan: {load} kg")
    finish_time = time.time()
    print(f"\nRunning time (detik): {finish_time - start_time:.2f}")
