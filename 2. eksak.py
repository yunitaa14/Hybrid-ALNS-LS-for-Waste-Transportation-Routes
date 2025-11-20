import pandas as pd
from pyomo.environ import *
import time

# === BACA DATA DARI EXCEL ===
df = pd.read_excel('data.xlsx', sheet_name='jarak2')           # kolom: From, To, Distance
df_demand = pd.read_excel('data.xlsx', sheet_name='demand')    # kolom: Node, Demand

# Ambil list/set node dan parameter demand
nodes = sorted(set(df['From']).union(set(df['To'])))            # Membaca node dari kolom 'From' dan kolom 'To' lalu digabung, duplikasi dibuang
demand = dict(zip(df_demand['Node'], df_demand['Demand']))      # Parameter 'Demand' diberi indeks 'Node'
depot = 0                                                       # Semua truk pengangkut sampah berangkat dari 1 gudang (depot = node 0)
TPA = 53                                                        # Setiap kendaraan harus buang sampah di TPA tepat sblm kembali ke depot (TPA = node 53)
K = range(1, 6)                                                 # Tersedia 5 truk pengangkut sampah
capacity = 2200                                                # Kapasitas sama, yaitu 3000Kg/truk
start = time.time()

# Simbol [] untuk list, sementara simbol () adalah set. Keduanya kombinasi indeks i,j dan j,i. Simbol {} untuk parameter jarak
# Jarak asymmetric
arcs = set()                                                       # Wadah kosong untuk set arcs (inisialisasi)
dist = {}                                                       # Wadah kosong untuk parameter dist (inisialisasi)
for row in df.itertuples(index=False):                          # Baca df baris demi baris lalu simpan sbg set arcs dan parameter dist
    i, j, d = row                                               # Ambil data dari kolom 'From', 'To', 'Distance'
    if i != j:
        arcs.add((i, j))                                         
        dist[i, j] = d                                              # Jarak yg tersimpan di sheet

# === BANGUN MODEL PYOMO ===
M = ConcreteModel()

# Variabel keputusan x[i,j,k]
M.x = Var(arcs, K, domain=Binary)

# Variabel muatan u[i,k] untuk MTZ
M.u = Var(nodes, K, bounds=(0, capacity))

# === OBJEKTIF ===
def obj_rule(m):
    return sum(dist[i, j] * m.x[i, j, k] for (i, j) in arcs for k in K)
M.obj = Objective(rule=obj_rule, sense=minimize)

# === KENDALA ===
# Setiap truk harus berangkat dari depot (start from depot)
def depart_depot_rule(m, k):
    return sum(m.x[depot, j, k] for (i, j) in arcs if i == depot) == 1
M.depart_depot = Constraint(K, rule=depart_depot_rule)

# Setiap kendaraan harus mengunjungi TPA 1x
def to_tpa_once(m, k):
    return sum(m.x[i, TPA, k] for (i, j) in arcs if j == TPA) == 1
M.to_tpa = Constraint(K, rule=to_tpa_once)

# Setelah mengunjungi TPA maka pasti node berikutnya adalah depot (wajib pulang ke depot setelah mengunjungi TPA)
def after_tpa_only_depot(m, k):
    return m.x[TPA, depot, k] == 1
M.after_tpa = Constraint(K, rule=after_tpa_only_depot)

# Setiap pelanggan dikunjungi tepat satu kali (pelanggan = node 1-52)
def visit_once_rule(m, j):
    if j in (depot, TPA):
        return Constraint.Skip
    return sum(m.x[i, j, k] for (i, jj) in arcs for k in K if jj == j) == 1
M.visit_once = Constraint(nodes, rule=visit_once_rule)

# Flow conservation
def flow_rule(m, i, k):
    if i in (depot, TPA):
        return Constraint.Skip
    return sum(m.x[i, j, k] for (ii, j) in arcs if ii == i) == sum(m.x[j, i, k] for (j, jj) in arcs if jj == i)
M.flow = Constraint([(i, k) for i in nodes for k in K], rule=flow_rule)

# Kapasitas kendaraan
def capacity_rule(m, k):
    return sum(demand[i] * m.x[i, j, k] for (i, j) in arcs if i not in (depot, TPA)) <= capacity
M.capacity_constr = Constraint(K, rule=capacity_rule)

# Subtour elimination MTZ (per kendaraan)
def mtz_rule(m, i, j, k):
    if i != depot and j != depot and i != TPA and j != TPA and i != j and (i, j) in arcs:
        return m.u[i, k] - m.u[j, k] + capacity * m.x[i, j, k] <= capacity - demand[j]
    return Constraint.Skip
M.mtz = Constraint([(i, j, k) for (i, j) in arcs for k in K], rule=mtz_rule)

# === SOLVER ===
solver = SolverFactory('gurobi')
solver.options['TimeLimit'] = 3600
result = solver.solve(M, tee=True)

# === AMBIL HASIL ===
edges_used = [(i, j, k) for (i, j) in arcs for k in K if value(M.x[i, j, k]) > 0.99]

# Cetak rute per kendaraan
print("\n=== RUTE PER KENDARAAN ===")
for k in K:
    route = [depot]
    current = depot
    visited = {depot}
    while True:
        next_nodes = [j for (i, j, kk) in edges_used if i == current and kk == k and j not in visited]
        if not next_nodes:
            break
        next_node = next_nodes[0]
        route.append(next_node)
        visited.add(next_node)
        current = next_node
    route.append(depot)
    print(f"Vehicle {k}: {' > '.join(map(str, route))}")
    print()
finish = time.time()
print(f"\nJarak total MIP = {value(M.obj):.2f}")
print(f"Running time gurobi pada VRP 54pt depot+52pt+TPA (detik) =  {finish-start:.2f}")
print("Syarat: data harus full tapi formatnya unpivot (sparse), cek format data pada file excel")
print()