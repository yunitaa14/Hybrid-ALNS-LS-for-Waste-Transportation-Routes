import pandas as pd
import numpy as np
import random
import copy
import time
import math
import os
from pathlib import Path

# === KONFIGURASI BENCHMARK ===
INSTANCES = [
    "E-n33-k4", "E-n51-k5", "E-n76-k8", "E-n76-k10", "E-n76-k14", 
    "E-n101-k8", "E-n101-k14", "X-n101-k25", "X-n106-k14"
]

MAX_ITER = 1000
REMOVE_FRAC = 0.2
RANDOM_SEED = 42

# === VARIABEL UNTUK PENGUKURAN WAKTU ===
operator_times = {
    'random_removal': 0.0,
    'worst_removal': 0.0,
    'greedy_insert': 0.0,
    'two_opt': 0.0,
    'relocate': 0.0,
    'swap': 0.0,
    'local_search': 0.0
}
iteration_times = []

# === FUNGSI UNTUK MEMBACA FILE VRP ===
def read_vrp_file(file_path):
    """Membaca file .vrp dan mengembalikan distance matrix, demands, dan parameter"""
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    # Parse basic information
    dimension = 0
    capacity = 0
    depot = 0
    node_coords = []
    demands = {}
    
    # State variables for parsing
    in_node_section = False
    in_demand_section = False
    in_depot_section = False
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        if line.startswith('DIMENSION'):
            dimension = int(line.split(':')[1].strip())
        elif line.startswith('CAPACITY'):
            capacity = int(line.split(':')[1].strip())
        elif line.startswith('NODE_COORD_SECTION'):
            in_node_section = True
            in_demand_section = False
            in_depot_section = False
            continue
        elif line.startswith('DEMAND_SECTION'):
            in_node_section = False
            in_demand_section = True
            in_depot_section = False
            continue
        elif line.startswith('DEPOT_SECTION'):
            in_node_section = False
            in_demand_section = False
            in_depot_section = True
            continue
        elif line.startswith('EOF'):
            break
        elif in_node_section:
            parts = line.split()
            if len(parts) >= 3:
                node_id = int(parts[0])
                x = float(parts[1])
                y = float(parts[2])
                node_coords.append((node_id, x, y))
        elif in_demand_section:
            parts = line.split()
            if len(parts) >= 2:
                node_id = int(parts[0])
                demand_val = int(parts[1])
                demands[node_id] = demand_val
        elif in_depot_section:
            parts = line.split()
            if parts and parts[0] != '-1':
                depot = int(parts[0])
    
    # Create distance matrix from coordinates
    num_nodes = dimension
    distance_matrix = np.zeros((num_nodes, num_nodes))
    
    coord_dict = {node_id: (x, y) for node_id, x, y in node_coords}
    
    for i in range(1, num_nodes + 1):
        for j in range(1, num_nodes + 1):
            if i in coord_dict and j in coord_dict:
                x1, y1 = coord_dict[i]
                x2, y2 = coord_dict[j]
                distance = math.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                distance_matrix[i-1][j-1] = distance
    
    # Adjust demands to 0-based indexing
    full_demands = {i-1: demands.get(i, 0) for i in range(1, num_nodes + 1)}
    
    # Depot is 0-based
    depot = depot - 1 if depot > 0 else 0
    
    return distance_matrix, full_demands, capacity, dimension, depot

# === FUNGSI UNTUK MEMBACA FILE SOL ===
def read_sol_file(file_path, dimension):
    """Membaca file .sol dan mengembalikan solusi optimal"""
    try:
        with open(file_path, 'r') as f:
            content = f.read().strip()
        
        routes = []
        current_route = []
        
        for line in content.split('\n'):
            line = line.strip()
            if not line:
                continue
                
            if line.startswith('Route') or line.startswith('route'):
                if current_route:
                    # Convert to 0-based indexing and ensure depot at start/end
                    if current_route[0] != 1:  # 1-based depot
                        current_route = [1] + current_route
                    if current_route[-1] != 1:
                        current_route.append(1)
                    # Filter nodes to valid range
                    valid_route = [x-1 for x in current_route if 1 <= x <= dimension]
                    if len(valid_route) >= 2:  # Minimal [depot, depot]
                        routes.append(valid_route)
                current_route = []
            else:
                # Parse nodes in the route
                nodes = [int(x) for x in line.split() if x.isdigit()]
                current_route.extend(nodes)
        
        if current_route:
            # Convert to 0-based indexing and ensure depot at start/end
            if current_route[0] != 1:
                current_route = [1] + current_route
            if current_route[-1] != 1:
                current_route.append(1)
            # Filter nodes to valid range
            valid_route = [x-1 for x in current_route if 1 <= x <= dimension]
            if len(valid_route) >= 2:
                routes.append(valid_route)
        
        return routes
    except Exception as e:
        print(f"Error reading solution file {file_path}: {e}")
        return None

# === FUNGSI ALNS-LS YANG SUDAH DIMODIFIKASI ===
def run_alns_for_instance(distance_matrix, full_demands, capacity, dimension, depot, instance_name):
    """Menjalankan ALNS-LS untuk instance tertentu"""
    
    # Set random seed untuk reproducibility
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    
    # Reset operator times untuk instance ini
    global operator_times, iteration_times
    operator_times = {key: 0.0 for key in operator_times}
    iteration_times = []
    
    # Parameter instance-specific
    TRUCK_CAPACITY = capacity
    DEPOT = depot
    customer_nodes = [i for i in range(dimension) if i != DEPOT and full_demands.get(i, 0) > 0]

    # === FUNGSI BANTU ===
    def route_cost(route):
        return sum(distance_matrix[route[i]][route[i+1]] for i in range(len(route)-1))

    def route_load(route):
        return sum(full_demands.get(n, 0) for n in route if n != DEPOT)

    def total_cost(solution):
        return sum(route_cost(r) for r in solution)

    def is_feasible(route):
        """Memeriksa feasibility rute: kapasitas dan depot di awal/akhir"""
        if route[0] != DEPOT or route[-1] != DEPOT:
            return False
            
        load = 0
        for node in route[1:-1]:  # Skip depot di awal dan akhir
            load += full_demands.get(node, 0)
            if load > TRUCK_CAPACITY:
                return False
        return True

    def finalize_route(route):
        """Memastikan rute diawali dan diakhiri depot"""
        route = [n for n in route if n != DEPOT]  # Hapus depot yang ada di tengah
        if not route:
            return [DEPOT, DEPOT]
        return [DEPOT] + route + [DEPOT]

    # === SOLUSI AWAL ===
    def greedy_initial_solution():
        """Membuat solusi awal dengan algoritma greedy tanpa fixed number of trucks"""
        unvisited = set(customer_nodes)
        solution = []
        
        while unvisited:
            route = [DEPOT]
            load = 0
            
            while unvisited:
                # Cari node yang feasible (muat) dan terdekat
                feasible = [n for n in unvisited if load + full_demands.get(n, 0) <= TRUCK_CAPACITY]
                if not feasible:
                    break
                    
                # Pilih node terdekat dari posisi terakhir
                last_node = route[-1]
                nearest = min(feasible, key=lambda n: distance_matrix[last_node][n])
                
                route.append(nearest)
                load += full_demands.get(nearest, 0)
                unvisited.remove(nearest)
            
            # Tutup rute dengan depot
            route.append(DEPOT)
            solution.append(route)
        
        return solution

    # === OPERATOR DESTROY ===
    def random_removal(solution, num_remove):
        start_time = time.time()
        flat = [n for r in solution for n in r if n != DEPOT]
        if not flat:
            result = (solution, [])
        else:
            removed = random.sample(flat, min(num_remove, len(flat)))
            new_solution = []
            for r in solution:
                filtered = [n for n in r if n not in removed]
                new_solution.append(finalize_route(filtered))
            result = (new_solution, removed)
        operator_times['random_removal'] += time.time() - start_time
        return result

    def worst_removal(solution, num_remove):
        start_time = time.time()
        node_saving = {}
        for r in solution:
            if len(r) <= 3:  # Skip rute yang hanya [depot, depot] atau sangat pendek
                continue
            for i in range(1, len(r)-1):  # Skip depot di awal dan akhir
                node = r[i]
                if node == DEPOT: 
                    continue
                saving = (distance_matrix[r[i-1]][node] + 
                         distance_matrix[node][r[i+1]] - 
                         distance_matrix[r[i-1]][r[i+1]])
                node_saving[node] = saving
        
        if not node_saving:
            result = random_removal(solution, num_remove)
        else:
            sorted_nodes = sorted(node_saving, key=node_saving.get, reverse=True)
            removed = sorted_nodes[:min(num_remove, len(sorted_nodes))]
            new_solution = []
            for r in solution:
                filtered = [n for n in r if n not in removed]
                new_solution.append(finalize_route(filtered))
            result = (new_solution, removed)
        operator_times['worst_removal'] += time.time() - start_time
        return result

    # === OPERATOR REPAIR ===
    def greedy_insert(solution, removed):
        start_time = time.time()
        for node in removed:
            best_cost = float('inf')
            best_pos = (0, 0)
            for r_idx, route in enumerate(solution):
                # Coba semua posisi yang mungkin dalam rute
                for i in range(1, len(route)-1):  # Skip depot di posisi 0 dan terakhir
                    trial = route[:i] + [node] + route[i:]
                    if is_feasible(trial):
                        c = route_cost(trial)
                        if c < best_cost:
                            best_cost = c
                            best_pos = (r_idx, i)
            
            if best_cost != float('inf'):
                r_idx, i = best_pos
                solution[r_idx].insert(i, node)
            else:
                # Jika tidak bisa dimasukkan di rute existing, buat rute baru
                new_route = [DEPOT, node, DEPOT]
                if is_feasible(new_route):
                    solution.append(new_route)
        operator_times['greedy_insert'] += time.time() - start_time
        return solution

    # === LOCAL SEARCH ===
    def two_opt(route):
        start_time = time.time()
        if len(route) <= 4:  # Terlalu pendek untuk 2-opt
            result = route
        else:
            best = route.copy()
            improved = True
            while improved:
                improved = False
                for i in range(1, len(route) - 2):
                    for j in range(i + 2, len(route) - 1):  # j harus minimal i+2
                        if j - i == 1:
                            continue
                        new_route = route[:i] + route[i:j][::-1] + route[j:]
                        if is_feasible(new_route) and route_cost(new_route) < route_cost(best):
                            best = new_route
                            improved = True
                route = best
            result = best
        operator_times['two_opt'] += time.time() - start_time
        return result

    def relocate(solution):
        start_time = time.time()
        best_sol = copy.deepcopy(solution)
        improved = False
        
        for i in range(len(solution)):
            if len(solution[i]) <= 3:  # Skip rute yang terlalu pendek
                continue
                
            for j in range(len(solution)):
                if i == j:
                    continue
                    
                for idx in range(1, len(solution[i]) - 1):  # Skip depot
                    node = solution[i][idx]
                    if node == DEPOT:
                        continue
                        
                    # Coba pindahkan node dari rute i ke rute j
                    new_route_i = solution[i][:idx] + solution[i][idx+1:]
                    for k in range(1, len(solution[j])):  # Posisi insert di rute j
                        new_route_j = solution[j][:k] + [node] + solution[j][k:]
                        
                        if is_feasible(new_route_i) and is_feasible(new_route_j):
                            new_sol = copy.deepcopy(solution)
                            new_sol[i] = finalize_route(new_route_i)
                            new_sol[j] = finalize_route(new_route_j)
                            
                            if total_cost(new_sol) < total_cost(best_sol):
                                best_sol = new_sol
                                improved = True
        
        result = best_sol if improved else solution
        operator_times['relocate'] += time.time() - start_time
        return result

    def swap(solution):
        start_time = time.time()
        best_sol = copy.deepcopy(solution)
        improved = False
        
        for i in range(len(solution)):
            for j in range(i + 1, len(solution)):
                for idx1 in range(1, len(solution[i]) - 1):
                    node1 = solution[i][idx1]
                    if node1 == DEPOT:
                        continue
                        
                    for idx2 in range(1, len(solution[j]) - 1):
                        node2 = solution[j][idx2]
                        if node2 == DEPOT:
                            continue
                            
                        # Tukar node1 dan node2
                        new_r1 = solution[i].copy()
                        new_r2 = solution[j].copy()
                        new_r1[idx1], new_r2[idx2] = node2, node1
                        
                        if is_feasible(new_r1) and is_feasible(new_r2):
                            new_sol = copy.deepcopy(solution)
                            new_sol[i] = finalize_route(new_r1)
                            new_sol[j] = finalize_route(new_r2)
                            
                            if total_cost(new_sol) < total_cost(best_sol):
                                best_sol = new_sol
                                improved = True
        
        result = best_sol if improved else solution
        operator_times['swap'] += time.time() - start_time
        return result

    def local_search(solution):
        start_time = time.time()
        improved = True
        current_sol = copy.deepcopy(solution)
        max_local_iter = 10  # Batasi iterasi local search
        
        for _ in range(max_local_iter):
            improved = False
            
            # 2-opt intra-route
            for i in range(len(current_sol)):
                new_route = two_opt(current_sol[i])
                if route_cost(new_route) < route_cost(current_sol[i]):
                    current_sol[i] = new_route
                    improved = True
            
            # Relocate inter-route
            new_sol_relocate = relocate(current_sol)
            if total_cost(new_sol_relocate) < total_cost(current_sol):
                current_sol = new_sol_relocate
                improved = True
            
            # Swap inter-route
            new_sol_swap = swap(current_sol)
            if total_cost(new_sol_swap) < total_cost(current_sol):
                current_sol = new_sol_swap
                improved = True
                
            if not improved:
                break
                
        operator_times['local_search'] += time.time() - start_time
        return current_sol

    # === ALNS CORE ===
    def alns(iter=MAX_ITER):
        current = greedy_initial_solution()
        current = local_search(current)
        best = copy.deepcopy(current)
        best_cost = total_cost(best)

        destroy_ops = [random_removal, worst_removal]
        repair_op = greedy_insert
        weights = [1 for _ in destroy_ops]
        scores = [0 for _ in destroy_ops]
        op_counts = [0 for _ in destroy_ops]

        for it in range(iter):
            iter_start_time = time.time()
            
            # Pilih destroy operator berdasarkan weights
            total_weight = sum(weights)
            if total_weight == 0:
                weights = [1 for _ in destroy_ops]
                total_weight = len(destroy_ops)
            
            r = random.uniform(0, total_weight)
            cumulative = 0
            d_idx = 0
            for i, w in enumerate(weights):
                cumulative += w
                if r <= cumulative:
                    d_idx = i
                    break
            
            d_op = destroy_ops[d_idx]
            op_counts[d_idx] += 1
            
            # Destroy and repair
            num_remove = max(1, int(REMOVE_FRAC * len(customer_nodes)))
            destroyed, removed = d_op(current, num_remove)
            repaired = repair_op(destroyed, removed)

            # Local search (dilakukan setiap 5 iterasi untuk efisiensi)
            if it % 5 == 0:
                candidate = local_search(repaired)
            else:
                candidate = repaired

            candidate_cost = total_cost(candidate)
            current_cost = total_cost(current)

            # Adaptive weight update
            if candidate_cost < best_cost:
                best = copy.deepcopy(candidate)
                best_cost = candidate_cost
                current = candidate
                scores[d_idx] += 1.5  # Reward besar untuk new best
            elif candidate_cost < current_cost:
                current = candidate
                scores[d_idx] += 1.2  # Reward untuk improvement
            elif candidate_cost < current_cost * 1.05:  # Terima solusi yang sedikit lebih buruk
                current = candidate
                scores[d_idx] += 0.5  # Small reward

            # Update weights periodically
            if (it + 1) % 100 == 0:
                for i in range(len(weights)):
                    if op_counts[i] > 0:
                        success_rate = scores[i] / op_counts[i]
                        weights[i] = weights[i] * 0.8 + success_rate * 10
                    else:
                        weights[i] = max(0.1, weights[i] * 0.8)
                
                # Reset scores dan counts
                scores = [0] * len(destroy_ops)
                op_counts = [0] * len(destroy_ops)

            # Progress reporting
            if (it + 1) % 50 == 0:
                print(f"    Iteration {it+1}/{iter}, Best: {best_cost:.2f}, Current: {current_cost:.2f}")
            
            # Catat waktu iterasi
            iteration_times.append(time.time() - iter_start_time)

        return best, best_cost

    # Jalankan ALNS
    print(f"  Running ALNS for {instance_name}...")
    start_time = time.time()
    best_solution, best_obj = alns(MAX_ITER)
    total_run_time = time.time() - start_time
    
    # Filter out empty routes
    best_solution = [route for route in best_solution if len(route) > 2]
    
    # Hitung statistik waktu
    avg_iteration_time = np.mean(iteration_times) if iteration_times else 0
    total_operator_time = sum(operator_times.values())
    
    return best_solution, best_obj, total_run_time, operator_times, avg_iteration_time

# === FUNGSI UTAMA BENCHMARK ===
def run_benchmark():
    """Menjalankan benchmark untuk semua instance"""
    results = []
    
    for instance in INSTANCES:
        print(f"\nMemproses instance: {instance}")
        
        # Cari file .vrp dan .sol
        vrp_file = None
        sol_file = None
        
        # Cari di berbagai lokasi yang mungkin
        possible_paths = [
            f"./{instance}.vrp",
            f"./instances/{instance}.vrp", 
            f"./dataset/{instance}.vrp",
            f"./benchmark/{instance}.vrp",
            f"./X/{instance}.vrp",
            f"./E/{instance}.vrp"
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                vrp_file = path
                break
                
        if vrp_file is None:
            print(f"  File VRP untuk {instance} tidak ditemukan!")
            continue
            
        # Cari file solusi
        for path in possible_paths:
            sol_path = path.replace('.vrp', '.sol')
            if os.path.exists(sol_path):
                sol_file = sol_path
                break
        
        # Baca data instance
        try:
            distance_matrix, full_demands, capacity, dimension, depot = read_vrp_file(vrp_file)
            
            print(f"  Instance info: {dimension} nodes, capacity: {capacity}, depot: {depot}")
            print(f"  Total demand: {sum(full_demands.values())}")
            
            # Jalankan ALNS-LS
            best_solution, best_cost, run_time, op_times, avg_iter_time = run_alns_for_instance(
                distance_matrix, full_demands, capacity, dimension, depot, instance
            )
            
            # Baca solusi optimal jika ada
            optimal_cost = None
            optimal_routes = None
            if sol_file and os.path.exists(sol_file):
                optimal_routes = read_sol_file(sol_file, dimension)
                if optimal_routes:
                    # Hitung cost solusi optimal
                    try:
                        optimal_cost = sum(
                            sum(distance_matrix[route[i]][route[i+1]] for i in range(len(route)-1))
                            for route in optimal_routes
                        )
                    except IndexError as e:
                        print(f"  Error calculating optimal cost: {e}")
                        optimal_cost = None
            
            # Simpan hasil
            result = {
                'Instance': instance,
                'ALNS_Cost': round(best_cost, 2),
                'Optimal_Cost': round(optimal_cost, 2) if optimal_cost else 'N/A',
                'Gap(%)': round(((best_cost - optimal_cost) / optimal_cost * 100), 2) 
                         if optimal_cost and optimal_cost > 0 else 'N/A',
                'Total_Time(s)': round(run_time, 2),
                'Avg_Iter_Time(ms)': round(avg_iter_time * 1000, 2),
                'Num_Routes': len(best_solution),
                'Total_Demand': sum(full_demands.values()),
                'Capacity': capacity,
                'Random_Removal_Time(s)': round(op_times['random_removal'], 4),
                'Worst_Removal_Time(s)': round(op_times['worst_removal'], 4),
                'Greedy_Insert_Time(s)': round(op_times['greedy_insert'], 4),
                'Two_Opt_Time(s)': round(op_times['two_opt'], 4),
                'Relocate_Time(s)': round(op_times['relocate'], 4),
                'Swap_Time(s)': round(op_times['swap'], 4),
                'Local_Search_Time(s)': round(op_times['local_search'], 4)
            }
            
            results.append(result)
            
            print(f"  Selesai: Cost = {best_cost:.2f}, Time = {run_time:.2f}s, Routes = {len(best_solution)}")
            print(f"  Waktu per operator: { {k: round(v, 4) for k, v in op_times.items()} }")
            
            # Tampilkan rute detail untuk instance kecil
            if dimension <= 100:
                for idx, route in enumerate(best_solution, 1):
                    cost = sum(distance_matrix[route[i]][route[i+1]] for i in range(len(route)-1))
                    load = sum(full_demands.get(n, 0) for n in route if n != depot)
                    print(f"    Route {idx}: {route} | Cost: {cost:.2f} | Load: {load}")
            
        except Exception as e:
            print(f"  Error processing {instance}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    
    # Simpan hasil ke DataFrame
    if results:
        results_df = pd.DataFrame(results)
        
        # Tampilkan hasil
        print("\n" + "="*120)
        print("HASIL BENCHMARK ALNS-LS")
        print("="*120)
        
        # Tampilkan hasil utama
        main_columns = ['Instance', 'ALNS_Cost', 'Optimal_Cost', 'Gap(%)', 'Total_Time(s)', 'Avg_Iter_Time(ms)', 'Num_Routes']
        print(results_df[main_columns].to_string(index=False))
        
        # Tampilkan detail waktu operator
        print("\n" + "="*80)
        print("DETAIL WAKTU OPERATOR (detik)")
        print("="*80)
        time_columns = ['Instance', 'Random_Removal_Time(s)', 'Worst_Removal_Time(s)', 'Greedy_Insert_Time(s)', 
                       'Two_Opt_Time(s)', 'Relocate_Time(s)', 'Swap_Time(s)', 'Local_Search_Time(s)']
        print(results_df[time_columns].to_string(index=False))
        
        # Simpan ke file Excel
        results_df.to_excel("alns_benchmark_results.xlsx", index=False)
        print(f"\nHasil disimpan ke: alns_benchmark_results.xlsx")
        
        # Hitung statistik
        valid_gaps = [r['Gap(%)'] for r in results if isinstance(r['Gap(%)'], (int, float))]
        if valid_gaps:
            avg_gap = sum(valid_gaps) / len(valid_gaps)
            best_gap = min(valid_gaps)
            worst_gap = max(valid_gaps)
            print(f"\nStatistik Gap: Rata-rata = {avg_gap:.2f}%, Terbaik = {best_gap:.2f}%, Terburuk = {worst_gap:.2f}%")
        
        # Hitung total waktu per operator
        total_op_times = {
            'Random Removal': sum(r['Random_Removal_Time(s)'] for r in results),
            'Worst Removal': sum(r['Worst_Removal_Time(s)'] for r in results),
            'Greedy Insert': sum(r['Greedy_Insert_Time(s)'] for r in results),
            '2-Opt': sum(r['Two_Opt_Time(s)'] for r in results),
            'Relocate': sum(r['Relocate_Time(s)'] for r in results),
            'Swap': sum(r['Swap_Time(s)'] for r in results),
            'Local Search': sum(r['Local_Search_Time(s)'] for r in results)
        }
        
        print(f"\nTotal Waktu Operator:")
        for op, time in total_op_times.items():
            print(f"  {op}: {time:.4f} detik")
        
        return results_df
    else:
        print("Tidak ada hasil yang berhasil dijalankan.")
        return None

# === JALANKAN BENCHMARK ===
if __name__ == "__main__":
    print("Memulai Benchmark ALNS-LS...")
    print(f"Instance yang akan diuji: {INSTANCES}")
    print(f"Konfigurasi: {MAX_ITER} iterasi, Random Seed: {RANDOM_SEED}")
    print("-" * 100)
    
    results = run_benchmark()