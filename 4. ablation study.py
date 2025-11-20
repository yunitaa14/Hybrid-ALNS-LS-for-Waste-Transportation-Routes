import pandas as pd
import numpy as np
import random
import copy
import time
import math
import matplotlib.pyplot as plt
import os
from scipy import stats
import itertools

# === PARAMETER DASAR ===
NUM_TRUCKS = 5
TRUCK_CAPACITY = 2200
DEPOT = 0
TPA = 53
MAX_ITER = 1000
REMOVE_FRAC = 0.2
NUM_RUNS = 30
RANDOM_SEEDS = [random.randint(1, 10000) for _ in range(NUM_RUNS)]
start_time = time.time()
# HAPUS INISIALISASI SEED GLOBAL DI SINI

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

def local_search(solution, tracker=None, disabled_ops=[]):
    improved = True
    while improved:
        improved = False
        
        # 2-opt
        if 'two_opt' not in disabled_ops:
            start_time = time.time()
            for i in range(len(solution)):
                old_cost = route_cost(solution[i])
                new_route = two_opt(solution[i])
                new_cost = route_cost(new_route)
                if new_cost < old_cost:
                    solution[i] = new_route
                    improved = True
                    if tracker:
                        tracker['two_opt']['improvement'] += (old_cost - new_cost)
                        tracker['two_opt']['success'] += 1
            if tracker:
                tracker['two_opt']['time'] += (time.time() - start_time)
                tracker['two_opt']['count'] += 1
        
        # Relocate
        if 'relocate' not in disabled_ops:
            start_time = time.time()
            old_total = total_cost(solution)
            new_sol = relocate(solution)
            new_total = total_cost(new_sol)
            if new_total < old_total:
                solution = new_sol
                improved = True
                if tracker:
                    tracker['relocate']['improvement'] += (old_total - new_total)
                    tracker['relocate']['success'] += 1
            if tracker:
                tracker['relocate']['time'] += (time.time() - start_time)
                tracker['relocate']['count'] += 1
        
        # Swap
        if 'swap' not in disabled_ops:
            start_time = time.time()
            old_total = total_cost(solution)
            new_sol = swap(solution)
            new_total = total_cost(new_sol)
            if new_total < old_total:
                solution = new_sol
                improved = True
                if tracker:
                    tracker['swap']['improvement'] += (old_total - new_total)
                    tracker['swap']['success'] += 1
            if tracker:
                tracker['swap']['time'] += (time.time() - start_time)
                tracker['swap']['count'] += 1
            
    return solution

# === ALNS CORE ===
def alns(random_seed, iter=1000, run_id=0, disabled_ops=[]):  # PERBAIKAN: random_seed tanpa default value
    # Set random seed untuk run ini
    random.seed(random_seed)
    np.random.seed(random_seed)
    
    # Inisialisasi tracker
    tracker = {
        'random_seed': random_seed,  # Catat seed yang digunakan
        'random_removal': {'count': 0, 'reward': 0},
        'worst_removal': {'count': 0, 'reward': 0},
        'greedy_insert': {'count': 0},
        'two_opt': {'count': 0, 'time': 0, 'improvement': 0, 'success': 0},
        'relocate': {'count': 0, 'time': 0, 'improvement': 0, 'success': 0},
        'swap': {'count': 0, 'time': 0, 'improvement': 0, 'success': 0},
        'history': [],
        'best_objective': float('inf'),
        'disabled_ops': disabled_ops
    }
    
    current = greedy_initial_solution()
    current = local_search(current, tracker, disabled_ops)
    best = copy.deepcopy(current)
    best_cost = total_cost(best)
    tracker['best_objective'] = best_cost

    destroy_ops = [random_removal, worst_removal]
    repair_op = greedy_insert
    weights = [1 for _ in destroy_ops]
    scores = [0 for _ in destroy_ops]

    # Filter operator yang tidak dinonaktifkan
    active_destroy_ops = []
    for i, op in enumerate(destroy_ops):
        op_name = "random_removal" if i == 0 else "worst_removal"
        if op_name not in disabled_ops:
            active_destroy_ops.append((i, op))
    
    for it in range(iter):
        # Track iterasi saat ini
        current_cost = total_cost(current)
        tracker['history'].append(current_cost)
        
        # Pilih dan eksekusi operator destroy (hanya yang aktif)
        if active_destroy_ops:
            choices, probs = zip(*[(idx, weights[idx]) for idx, _ in active_destroy_ops])
            d_idx = random.choices(choices, weights=probs)[0]
            d_op = destroy_ops[d_idx]
        else:
            # Jika semua operator destroy dinonaktifkan, gunakan random sebagai fallback
            d_idx = 0
            d_op = random_removal
        
        num_remove = int(REMOVE_FRAC * len(customer_nodes))
        
        # Track destroy operator
        op_name = "random_removal" if d_idx == 0 else "worst_removal"
        if op_name not in disabled_ops:
            tracker[op_name]['count'] += 1
            
        # Eksekusi destroy
        destroyed, removed = d_op(current, num_remove)
        
        # Track repair operator
        if 'greedy_insert' not in disabled_ops:
            tracker['greedy_insert']['count'] += 1
            
        # Eksekusi repair
        repaired = repair_op(destroyed, removed)

        # Eksekusi local search (dengan tracking)
        if it % 5 == 0:
            improved = local_search(repaired, tracker, disabled_ops)
        else:
            improved = repaired

        cost = total_cost(improved)

        # Update solusi dan beri reward
        if cost < best_cost:
            best = copy.deepcopy(improved)
            best_cost = cost
            tracker['best_objective'] = best_cost
            scores[d_idx] += 1
            # Beri reward tinggi untuk operator yang menghasilkan perbaikan
            if op_name not in disabled_ops:
                tracker[op_name]['reward'] += 1.5
        elif cost < current_cost:
            # Beri reward sedang untuk perbaikan lokal
            if op_name not in disabled_ops:
                tracker[op_name]['reward'] += 1.0
        
        current = improved

        # Update weights setiap 100 iterasi
        if (it + 1) % 100 == 0:
            for i in range(len(weights)):
                weights[i] = 1 + scores[i]
            scores = [0] * len(destroy_ops)

    return best, best_cost, tracker

# === ANALISIS HASIL ===
def analyze_results(all_trackers):
    # Hitung statistik agregat
    operator_names = ['random_removal', 'worst_removal', 'greedy_insert', 
                     'two_opt', 'relocate', 'swap']
    
    results = []
    for op_name in operator_names:
        counts = []
        improvements = []
        times = []
        rewards = []
        success_rates = []
        
        for tracker in all_trackers:
            if op_name in tracker:
                counts.append(tracker[op_name].get('count', 0))
                improvements.append(tracker[op_name].get('improvement', 0))
                times.append(tracker[op_name].get('time', 0))
                rewards.append(tracker[op_name].get('reward', 0))
                
                # Hitung success rate jika tersedia
                if 'success' in tracker[op_name] and tracker[op_name]['count'] > 0:
                    success_rates.append(tracker[op_name]['success'] / tracker[op_name]['count'])
                else:
                    success_rates.append(0)
        
        # Hitung rata-rata
        avg_count = sum(counts) / len(counts) if counts else 0
        avg_improvement = sum(improvements) / len(improvements) if improvements else 0
        avg_time = sum(times) / len(times) if times else 0
        avg_reward = sum(rewards) / len(rewards) if rewards else 0
        avg_success = sum(success_rates) / len(success_rates) if success_rates else 0
        
        results.append({
            'Operator': op_name,
            'Frekuensi': avg_count,
            'Kontribusi Improvement': avg_improvement,
            'Waktu Komputasi': avg_time,
            'Total Reward': avg_reward,
            'Success Rate': avg_success
        })
    
    return results

# === VISUALISASI KONVERGENSI ===
def plot_convergence(all_histories, best_objectives, scenario_name):
    plt.figure(figsize=(12, 8))
    
    for i, history in enumerate(all_histories):
        plt.plot(history, label=f'Run {i+1} (Best: {best_objectives[i]:.2f})', alpha=0.8)
    
    plt.title(f'Konvergensi Solusi: {scenario_name}')
    plt.xlabel('Iterasi')
    plt.ylabel('Total Jarak (km)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'convergence_{scenario_name}.png')
    plt.close()

# === VISUALISASI PERBANDINGAN SKENARIO ===
def plot_scenario_comparison(scenario_results):
    # Siapkan data
    scenarios = list(scenario_results.keys())
    avg_bests = [scenario_results[scen]['avg_best'] for scen in scenarios]
    baseline = scenario_results['Baseline']['avg_best']
    improvements = [((baseline - avg) / baseline) * 100 for avg in avg_bests]
    
    # Buat plot
    fig, ax1 = plt.subplots(figsize=(14, 8))
    
    # Plot nilai objektif
    ax1.bar(scenarios, avg_bests, color='skyblue')
    ax1.set_xlabel('Skenario')
    ax1.set_ylabel('Rata-rata Best Objective (km)', color='skyblue')
    ax1.tick_params(axis='y', labelcolor='skyblue')
    
    # Plot persentase perbaikan
    ax2 = ax1.twinx()
    ax2.plot(scenarios, improvements, 'ro-', markersize=8, linewidth=2)
    ax2.set_ylabel('Perbaikan Relatif (%)', color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    ax2.axhline(y=0, color='gray', linestyle='--')
    
    plt.title('Perbandingan Kinerja Antar Skenario')
    fig.tight_layout()
    plt.savefig('scenario_comparison.png')
    plt.close()

# === STATISTICAL TESTING ===
def perform_statistical_tests(scenario_results):
    """Perform statistical tests between scenarios"""
    print("\n=== STATISTICAL ANALYSIS ===")
    print("="*100)
    
    # Kruskal-Wallis test untuk semua skenario
    all_data = []
    scenario_names = []
    for scen_name, data in scenario_results.items():
        all_data.append(data['best_objectives'])
        scenario_names.append(scen_name)
    
    h_stat, p_value = stats.kruskal(*all_data)
    print(f"Kruskal-Wallis Test: H={h_stat:.4f}, p-value={p_value:.6f}")
    
    if p_value < 0.05:
        print("Significant differences found between scenarios (p < 0.05)")
        
        # Pairwise Mann-Whitney U tests dengan Bonferroni correction
        print("\nPairwise Mann-Whitney U Tests (with Bonferroni correction):")
        pairs = list(itertools.combinations(scenario_names, 2))
        alpha = 0.05 / len(pairs)  # Bonferroni correction
        
        for pair in pairs:
            data1 = scenario_results[pair[0]]['best_objectives']
            data2 = scenario_results[pair[1]]['best_objectives']
            u_stat, p_val = stats.mannwhitneyu(data1, data2, alternative='two-sided')
            significance = "***" if p_val < alpha else "NS"
            print(f"{pair[0]} vs {pair[1]}: U={u_stat:.1f}, p={p_val:.6f} {significance}")
    else:
        print("No significant differences found between scenarios (p >= 0.05)")
    
    print("="*100)

# === JALANKAN ABLATION STUDY ===
if __name__ == "__main__":
    # Daftar skenario ablation study
    scenarios = [
        {"name": "Baseline", "disabled_ops": []},
        {"name": "No_Random_Removal", "disabled_ops": ["random_removal"]},
        {"name": "No_Worst_Removal", "disabled_ops": ["worst_removal"]},
        {"name": "No_2opt", "disabled_ops": ["two_opt"]},
        {"name": "No_Relocate", "disabled_ops": ["relocate"]},
        {"name": "No_Swap", "disabled_ops": ["swap"]},
    ]
    
    num_runs = NUM_RUNS
    scenario_results = {}

    # Simpan semua random seeds yang digunakan
    print(f"Random Seeds used: {RANDOM_SEEDS}")
    
    for scenario in scenarios:
        print(f"\n=== SKENARIO: {scenario['name']} ===")
        all_trackers = []
        all_histories = []
        best_objectives = []
        
        # HANYA SATU LOOP - HAPUS LOOP DUPLIKAT
        for run_id in range(num_runs):
            current_seed = RANDOM_SEEDS[run_id]
            print(f"  Run {run_id+1}/{num_runs} - Seed: {current_seed}")
            
            best_solution, best_obj, tracker = alns(
                random_seed=current_seed,  # PERBAIKAN: gunakan parameter yang benar
                iter=MAX_ITER, 
                run_id=run_id, 
                disabled_ops=scenario['disabled_ops']
            )
            
            # Simpan hasil
            all_trackers.append(tracker)
            all_histories.append(tracker['history'])
            best_objectives.append(best_obj)
        
        # Hitung statistik untuk skenario ini
        avg_best = sum(best_objectives) / num_runs
        median_best = np.median(best_objectives)
        q1 = np.percentile(best_objectives, 25)
        q3 = np.percentile(best_objectives, 75)
        iqr = q3 - q1
        std_dev = np.std(best_objectives)
        
        # Simpan hasil skenario
        scenario_results[scenario['name']] = {
            "avg_best": avg_best,
            "median_best": median_best,
            "q1": q1,
            "q3": q3,
            "iqr": iqr,
            "std_dev": std_dev,
            "best_objectives": best_objectives,
            "all_histories": all_histories,
            "random_seeds": RANDOM_SEEDS.copy()  # Simpan salinan seeds
        }
        
        # Buat grafik konvergensi untuk skenario ini
        plot_convergence(all_histories, best_objectives, scenario['name'])
        
        # Cetak ringkasan skenario
        print(f"\nHasil Skenario {scenario['name']}:")
        print(f"Rata-rata Best Objective: {avg_best:.2f} km")
        print(f"Median Best Objective: {median_best:.2f} km")
        print(f"Q1-Q3: {q1:.2f} - {q3:.2f} km")
        print(f"IQR: {iqr:.2f} km")
        print(f"Standard Deviation: {std_dev:.2f} km")
        print(f"Best Objectives per Run: {[f'{obj:.2f}' for obj in best_objectives]}")
    
    # Buat grafik perbandingan antar skenario
    plot_scenario_comparison(scenario_results)
    
    # Panggil fungsi statistical tests
    perform_statistical_tests(scenario_results)
    
    # Buat laporan perbandingan yang diperbaiki
    print("\n\n=== PERBANDINGAN SKENARIO ABLATION STUDY (30 RUNS) ===")
    print("="*120)
    print(f"{'Skenario':<20} {'Median (km)':<12} {'Avg (km)':<12} {'Q1-Q3 (km)':<15} {'IQR (km)':<10} {'Std Dev':<10} {'Operator Dinonaktifkan'}")
    print("-"*120)
    
    baseline_median = scenario_results['Baseline']['median_best']
    
    for scen_name, data in scenario_results.items():
        median_best = data['median_best']
        avg_best = data['avg_best']
        q1_q3 = f"{data['q1']:.1f}-{data['q3']:.1f}"
        iqr = data['iqr']
        std_dev = data['std_dev']
        
        # Identifikasi operator yang dinonaktifkan
        scen_config = next(s for s in scenarios if s['name'] == scen_name)
        disabled_ops = scen_config['disabled_ops'] or ["-"]
        
        # Hitung perubahan relatif terhadap baseline
        change_median = median_best - baseline_median
        change_pct = (change_median / baseline_median) * 100 if baseline_median != 0 else 0
        
        print(f"{scen_name:<20} {median_best:<12.2f} {avg_best:<12.2f} {q1_q3:<15} {iqr:<10.2f} {std_dev:<10.2f} {', '.join(disabled_ops)}")
    
    print("="*120)
    
    # Cetak random seeds untuk reproducibility
    print(f"\nRandom Seeds used across all runs: {RANDOM_SEEDS}")
    
    finish_time = time.time()
    print(f"\nTotal running time (detik): {finish_time - start_time:.2f}")