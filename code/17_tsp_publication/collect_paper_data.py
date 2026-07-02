"""
collect_paper_data.py
Collects all data needed to fill the TODO sections in main.tex.

Runs locally using Qiskit Aer (no IBM account needed).
For gate count: just transpile, no execution needed.
For optimizer/timing/OR: use AerSimulator with shot-based sampling.
For noise: AerSimulator with noise models.

Output: paper_data.json  (human-readable summary)
        paper_data.csv   (all raw rows)
"""

import os, sys, json, csv, math, time, itertools
from collections import Counter
from datetime import datetime

import numpy as np
from scipy.optimize import minimize
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import (
    NoiseModel, depolarizing_error, thermal_relaxation_error
)

# ── paths ──────────────────────────────────────────────────────────────────
HERE   = os.path.dirname(os.path.abspath(__file__))
CASEY  = os.path.join(HERE, '..', '13_data gathering', 'cluster_complete_graphs.json')
OUTDIR = os.path.join(HERE, 'results')
os.makedirs(OUTDIR, exist_ok=True)

# ── constants ───────────────────────────────────────────────────────────────
PENALTY     = 1000.0
QUBO_A      = 1000.0
QUBO_B      = 1.0
COBYLA_ITER  = 30           # keep short for speed
SHOTS        = 128
QAOA_DEPTH   = 3
TSPLIB_SEEDS = [42, 123, 777]

# Max qubits for full COBYLA run on local Aer simulator
# Naive n=5 → 25 qubits, 900 gates/layer → too slow
# NVT n=6 → 15 qubits fine; NVT n=7 → 18 qubits fine
MAX_QUBITS_FOR_OPTIMIZER = 20

SIM_SHOT = AerSimulator()   # shot-based, works for all qubit counts
SIM_SV   = AerSimulator(method='statevector')  # fast for small circuits

# ── helpers: qubit counts ────────────────────────────────────────────────────
def q_nvt(n):
    return (n-1) * math.ceil(math.log2(n)) if n > 2 else 1

def q_naive(n):
    return n * n

def q_log(n):
    return math.ceil(math.log2(math.factorial(n-1))) if n > 2 else 1

# ── helpers: TSP instances ───────────────────────────────────────────────────
def random_tsp(n, seed):
    rng = np.random.default_rng(seed)
    xy  = rng.uniform(0, 100, (n, 2))
    D   = np.sqrt(((xy[:,None,:] - xy[None,:,:])**2).sum(-1))
    return D

def brute_force(D, n):
    best, opt = math.inf, None
    for p in itertools.permutations(range(1, n)):
        tour = [0] + list(p) + [0]
        c = sum(D[tour[i]][tour[i+1]] for i in range(len(tour)-1))
        if c < best:
            best, opt = c, tour
    return best, opt

def load_casey_instances():
    """Return list of (name, n, D_matrix, opt_cost) from Casey JSON."""
    if not os.path.exists(CASEY):
        print(f"  Casey JSON not found: {CASEY}")
        return []
    with open(CASEY) as f:
        data = json.load(f)
    instances = []
    for item in data.get('waste_areas_to_facilities_graphs', []):
        tid   = int(item['waste_facility_id'])
        raw   = item.get('distance_matrix_km', {})
        dist  = {int(r): {int(c): v for c,v in row.items()} for r,row in raw.items()}
        nodes = sorted(dist.keys())
        n     = len(nodes)
        idx   = {nid: i for i, nid in enumerate(nodes)}
        D     = np.zeros((n, n))
        for r in nodes:
            for c in nodes:
                D[idx[r]][idx[c]] = dist[r].get(c, 0.0)
        opt, _ = brute_force(D, n)
        instances.append((f'casey_{tid}', n, D, opt))
    return instances

# ── QAOA circuit ─────────────────────────────────────────────────────────────
def make_circuit(nq, p, gamma, beta):
    qc = QuantumCircuit(nq, nq)
    qc.h(range(nq))
    for k in range(p):
        for i in range(nq):
            qc.rz(2*gamma[k], i)
        for i in range(nq):
            for j in range(i+1, nq):
                qc.rzz(2*gamma[k], i, j)
        for i in range(nq):
            qc.rx(2*beta[k], i)
    qc.measure(range(nq), range(nq))
    return qc

# ── gate count via transpilation only (no circuit run) ───────────────────────
def gate_count(nq, p, backend=None):
    """
    Transpile and count 2Q gates. For circuits exceeding the backend's
    qubit limit, fall back to analytical count (logical, before routing).
    Analytical: p * nq*(nq-1)/2 RZZ gates (each = 2 CX after decomposition).
    """
    if backend is None:
        backend = SIM_SHOT
    gamma = np.ones(p) * 0.5
    beta  = np.ones(p) * 0.3
    qc  = make_circuit(nq, p, gamma, beta)
    t0  = time.perf_counter()
    try:
        qct = transpile(qc, backend=backend, optimization_level=1)
        t_transp = time.perf_counter() - t0
        gates_2q = sum(1 for instr in qct.data if len(instr.qubits) == 2)
        depth    = qct.depth()
        return gates_2q, depth, t_transp
    except Exception:
        # Circuit too wide for backend — compute analytically
        t_transp = time.perf_counter() - t0
        rzz_count = p * nq * (nq - 1) // 2
        gates_2q  = rzz_count * 2   # each RZZ ≈ 2 CX after decomposition
        depth     = -1              # -1 = not available (infeasible on hardware)
        return gates_2q, depth, t_transp

# ── NVT encoding ─────────────────────────────────────────────────────────────
def decode_nvt(bs, n):
    b   = math.ceil(math.log2(n)) if n > 1 else 1
    vts = [int(bs[i*b:(i+1)*b], 2) for i in range(n-1)]
    order = [i+1 for i,_ in sorted(enumerate(vts), key=lambda x: x[1])]
    tour  = [0] + order + [0]
    return tour, vts

def nvt_cost(bs, D, n):
    tour, vts = decode_nvt(bs, n)
    dc   = sum(D[tour[i]][tour[i+1]] for i in range(len(tour)-1))
    dup  = PENALTY * sum((c-1)**2 for c in Counter(vts).values() if c > 1)
    rang = PENALTY * sum(1 for v in vts if v < 1 or v > n-1)
    return dc + dup + rang

def is_valid_nvt(bs, n):
    _, vts = decode_nvt(bs, n)
    return set(vts) == set(range(1, n))

# ── Naive one-hot encoding ───────────────────────────────────────────────────
def naive_cost(bs, D, n):
    x   = np.array(list(map(int, bs)), dtype=float).reshape(n, n)
    pen = QUBO_A * (sum((1 - x[:,t].sum())**2 for t in range(n)) +
                    sum((1 - x[i,:].sum())**2 for i in range(n)))
    trav = QUBO_B * sum(D[i][j] * x[i,t] * x[j,(t+1)%n]
                        for i in range(n) for j in range(n) for t in range(n))
    return pen + trav

def is_valid_naive(bs, n):
    x = np.array(list(map(int, bs))).reshape(n, n)
    cities = [int(np.where(x[:,t]==1)[0][0]) if x[:,t].sum()==1 else -1
              for t in range(n)]
    return -1 not in cities and len(set(cities)) == n

# ── run one QAOA experiment ──────────────────────────────────────────────────
def run_one(name, n, D, opt, enc, p=QAOA_DEPTH, shots=SHOTS,
            sim=None, noise_model=None, max_iter=COBYLA_ITER):
    if sim is None:
        sim = SIM_SHOT

    if enc == 'nvt':
        nq      = q_nvt(n)
        cost_fn = lambda bs: nvt_cost(bs, D, n)
        valid_fn= lambda bs: is_valid_nvt(bs, n)
    else:
        nq      = q_naive(n)
        cost_fn = lambda bs: naive_cost(bs, D, n)
        valid_fn= lambda bs: is_valid_naive(bs, n)

    # gate count (transpile only)
    g2q, depth, t_transp = gate_count(nq, p, sim)

    # COBYLA optimisation
    iters = [0]; evals = [0]
    def objective(params):
        gamma, beta = params[:p], params[p:]
        qc  = make_circuit(nq, p, gamma, beta)
        qct = transpile(qc, sim, optimization_level=1)
        job = sim.run(qct, shots=shots, noise_model=noise_model)
        counts = job.result().get_counts()
        total  = sum(counts.values())
        exp = sum((cnt/total)*cost_fn(bs[::-1]) for bs, cnt in counts.items())
        evals[0] += 1
        return exp
    def cb(_): iters[0] += 1

    x0 = np.random.uniform(0, np.pi, 2*p)
    t0 = time.perf_counter()
    try:
        res = minimize(objective, x0, method='COBYLA',
                       options={'maxiter': max_iter, 'rhobeg': 0.8},
                       callback=cb)
    except Exception:
        res = type('R', (), {'x': x0, 'success': False})()
    t_opt = time.perf_counter() - t0

    # final measurement
    gamma_f, beta_f = res.x[:p], res.x[p:]
    qc_f  = make_circuit(nq, p, gamma_f, beta_f)
    qct_f = transpile(qc_f, sim, optimization_level=1)
    job_f = sim.run(qct_f, shots=shots, noise_model=noise_model)
    counts_f = job_f.result().get_counts()

    valid_c = []; valid_n = 0; inv_n = 0
    for bs, cnt in counts_f.items():
        bsr = bs[::-1]
        if valid_fn(bsr):
            c = cost_fn(bsr)
            if c < 1e8 and not math.isinf(c):
                valid_c.extend([c]*cnt); valid_n += cnt
        else:
            inv_n += cnt

    total = valid_n + inv_n
    exp_cost = (sum(valid_c)/len(valid_c)) if valid_c else math.nan
    or_val   = (opt/exp_cost) if (valid_c and exp_cost > 0) else math.nan

    return dict(
        name=name, n=n, enc=enc, p=p, shots=shots,
        qubits=nq, gates_2q=g2q, circuit_depth=depth,
        t_transp=round(t_transp,4), t_opt=round(t_opt,2),
        t_total=round(t_transp+t_opt,2),
        opt_iters=iters[0], opt_evals=evals[0],
        classical_opt=round(opt,6),
        exp_cost=round(exp_cost,6) if not math.isnan(exp_cost) else 'NaN',
        or_val=round(or_val,6) if not math.isnan(or_val) else 'NaN',
        valid_n=valid_n, inv_n=inv_n,
        success=(valid_n>0),
        valid_pct=round(valid_n/total*100,2) if total>0 else 0,
    )

# ── noise sensitivity ─────────────────────────────────────────────────────────
NOISE_CONFIGS = {
    'ideal':         None,
    'depol_1e-3':    ('depol', 1e-3),
    'depol_5e-3':    ('depol', 5e-3),
    'depol_1e-2':    ('depol', 1e-2),
    'thermal_200us': ('therm', 200e-6, 100e-6),
    'thermal_100us': ('therm', 100e-6,  50e-6),
    'thermal_50us':  ('therm',  50e-6,  25e-6),
}

def make_noise_model(cfg):
    if cfg is None:
        return None
    nm = NoiseModel()
    if cfg[0] == 'depol':
        p1, p2 = cfg[1], min(cfg[1]*10, 0.5)
        nm.add_all_qubit_quantum_error(depolarizing_error(p1, 1), ['rx','rz','h'])
        nm.add_all_qubit_quantum_error(depolarizing_error(p2, 2), ['cx','rzz','ecr'])
    elif cfg[0] == 'therm':
        T1, T2 = cfg[1], cfg[2]
        t1q, t2q = 50e-9, 300e-9
        e1 = thermal_relaxation_error(T1, T2, t1q)
        e2 = thermal_relaxation_error(T1, T2, t2q).expand(
             thermal_relaxation_error(T1, T2, t2q))
        nm.add_all_qubit_quantum_error(e1, ['rx','rz','h'])
        nm.add_all_qubit_quantum_error(e2, ['cx','rzz','ecr'])
    return nm

def run_noise_sweep(n, D, opt, enc, n_trials=2):
    rows = []
    for label, cfg in NOISE_CONFIGS.items():
        nm  = make_noise_model(cfg)
        sim = AerSimulator(noise_model=nm) if nm else SIM_SV
        ors, vfs = [], []
        for seed in range(n_trials):
            np.random.seed(seed*7 + n)
            r = run_one(f'noise_n{n}', n, D, opt, enc, sim=sim, noise_model=nm,
                        max_iter=30)   # fewer iters for noise sweep
            ors.append(r['or_val'] if r['or_val'] != 'NaN' else math.nan)
            vfs.append(r['valid_pct'])
        rows.append(dict(
            noise=label, enc=enc, n=n,
            or_mean=round(np.nanmean(ors),4) if not all(math.isnan(v) for v in ors) else 'NaN',
            or_std =round(np.nanstd(ors) ,4) if not all(math.isnan(v) for v in ors) else 'NaN',
            valid_mean=round(np.mean(vfs),2),
        ))
        print(f"    {label:20s} enc={enc} OR={rows[-1]['or_mean']} valid={rows[-1]['valid_mean']:.1f}%")
    return rows

# ── main ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("Collecting paper data for TSP QAOA publication")
    print("=" * 60)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')

    # ── 1. Gate count table (just transpile, very fast) ───────────────────
    print("\n[1] Gate count table (transpilation only) ...")
    gate_rows = []
    for n in [3, 4, 5, 6]:
        for enc, nq_fn in [('nvt', q_nvt), ('naive', q_naive)]:
            nq = nq_fn(n)
            g2q, depth, t = gate_count(nq, QAOA_DEPTH)
            gate_rows.append(dict(n=n, enc=enc, qubits=nq,
                                  gates_2q=g2q, depth=depth,
                                  t_transp=round(t,4)))
            print(f"  n={n} {enc:6s}: {nq:3d} qubits, {g2q:5d} 2Q gates, depth={depth}")

    # ── 2. TSPLIB random instances: optimizer + timing + OR ──────────────
    print("\n[2] TSPLIB random instances (optimizer + timing + OR) ...")
    main_rows = []
    for n in [3, 4, 5]:
        for seed in TSPLIB_SEEDS:
            D   = random_tsp(n, seed)
            opt, _ = brute_force(D, n)
            name = f'tsp_n{n}_s{seed}'
            for enc in ['nvt', 'naive']:
                nq_enc = q_nvt(n) if enc == 'nvt' else q_naive(n)
                if nq_enc > MAX_QUBITS_FOR_OPTIMIZER:
                    print(f"  {name} enc={enc}: SKIP (nq={nq_enc} > {MAX_QUBITS_FOR_OPTIMIZER})")
                    continue
                print(f"  {name} enc={enc} ...", end=' ', flush=True)
                try:
                    r = run_one(name, n, D, opt, enc)
                    main_rows.append(r)
                    print(f"OR={r['or_val']} gates={r['gates_2q']} "
                          f"iters={r['opt_iters']} t={r['t_total']}s")
                except Exception as e:
                    print(f"ERROR: {e}")

    # ── 3. Casey instances (first 5 only, n<=5, NVT only for n>4)
    print("\n[3] Casey instances (first 5 instances) ...")
    casey = load_casey_instances()
    for (name, n, D, opt) in casey[:5]:
        for enc in ['nvt', 'naive']:
            nq_enc = q_nvt(n) if enc == 'nvt' else q_naive(n)
            if nq_enc > MAX_QUBITS_FOR_OPTIMIZER:
                print(f"  {name} n={n} enc={enc}: SKIP (nq={nq_enc} > {MAX_QUBITS_FOR_OPTIMIZER})")
                continue
            print(f"  {name} n={n} enc={enc} ...", end=' ', flush=True)
            try:
                r = run_one(name, n, D, opt, enc)
                main_rows.append(r)
                print(f"OR={r['or_val']} valid={r['valid_pct']}%")
            except Exception as e:
                print(f"ERROR: {e}")

    # ── 4. Noise sensitivity (NVT on n=5, random instance) ───────────────
    print("\n[4] Noise sensitivity sweep (n=5, NVT + naive) ...")
    D5 = random_tsp(5, 42)
    opt5, _ = brute_force(D5, 5)
    noise_rows = []
    for enc in ['nvt']:   # naive n=5 too slow for noise sweep
        print(f"  enc={enc}:")
        noise_rows += run_noise_sweep(5, D5, opt5, enc)

    # ── 5. Scalability sweep (NVT only, n=3..7) ──────────────────────────
    print("\n[5] Scalability sweep (NVT, n=3..7) ...")
    scale_rows = []
    for n in [3, 4, 5, 6, 7]:
        D  = random_tsp(n, 42)
        opt, _ = brute_force(D, n)
        print(f"  n={n} NVT (nq={q_nvt(n)}) ...", end=' ', flush=True)
        try:
            r = run_one(f'scale_n{n}', n, D, opt, 'nvt')
            scale_rows.append(r)
            print(f"OR={r['or_val']} success={r['success']} valid={r['valid_pct']}%")
        except Exception as e:
            print(f"ERROR: {e}")

    # ── Save results ───────────────────────────────────────────────────────
    out = {
        'timestamp': ts,
        'gate_count_table': gate_rows,
        'main_results': main_rows,
        'noise_sensitivity': noise_rows,
        'scalability': scale_rows,
    }
    json_path = os.path.join(OUTDIR, f'paper_data_{ts}.json')
    with open(json_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {json_path}")

    # ── Pretty print summary for copy-paste into paper ────────────────────
    print("\n" + "="*60)
    print("TABLE: Gate count (p=3)")
    print(f"{'n':>4} {'Encoding':>12} {'Qubits':>8} {'2Q gates':>10} {'Depth':>8}")
    for r in gate_rows:
        print(f"{r['n']:>4} {r['enc']:>12} {r['qubits']:>8} {r['gates_2q']:>10} {r['depth']:>8}")

    print("\nTABLE: Noise sensitivity (NVT, n=5)")
    print(f"{'Scenario':>22} {'OR':>8} {'Valid%':>8}")
    for r in noise_rows:
        if r['enc'] == 'nvt':
            print(f"{r['noise']:>22} {r['or_mean']:>8} {r['valid_mean']:>8}")

    print("\nTABLE: Scalability (NVT)")
    print(f"{'n':>4} {'Qubits':>8} {'OR':>8} {'Valid%':>8} {'Success':>8}")
    for r in scale_rows:
        print(f"{r['n']:>4} {r['qubits']:>8} {r['or_val']:>8} {r['valid_pct']:>8} {r['success']:>8}")

    return out

if __name__ == '__main__':
    main()
