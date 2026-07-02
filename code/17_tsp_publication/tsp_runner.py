"""
tsp_runner.py
Main experiment runner for the TSP publication.

Collects ALL metrics required by the supervisor:
  1. Qubit count (theoretical, printed in summary)
  2. Gate count (from transpiled circuit on IBM backend)
  3. Optimizer iterations + evaluations to convergence
  4. Compute time (wall-clock)
  5. Optimality ratio
  (6. Noise sensitivity -> see tsp_noise.py)
  (7. Scalability -> parameterize --nodes flag)

Usage:
  # With IBM Quantum (real hardware):
  python tsp_runner.py --backend ibm_brisbane --shots 1024 --depth 3

  # With fake provider (local, no IBM account):
  python tsp_runner.py --fake --shots 1024 --depth 3

  # Specify problem size range:
  python tsp_runner.py --fake --nodes 3,4,5,6,7,8

  # Use TSPLIB random instances instead of Casey:
  python tsp_runner.py --fake --dataset tsplib --nodes 3,4,5,6

Output:
  results/tsp_results.csv   -- one row per (instance, encoding, depth, shots)
"""

import os
import csv
import math
import time
import json
import argparse
import itertools
from collections import Counter
from datetime import datetime

import numpy as np
from scipy.optimize import minimize
from qiskit import QuantumCircuit, transpile

# ── IBM Runtime (real hardware) ──────────────────────────────────────────────
try:
    from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler
    IBM_AVAILABLE = True
except ImportError:
    IBM_AVAILABLE = False

# ── Fake provider (local simulation) ─────────────────────────────────────────
try:
    from qiskit_ibm_runtime.fake_provider import FakeManilaV2 as FakeBackend
    from qiskit_ibm_runtime import StatevectorSampler
    FAKE_AVAILABLE = True
except ImportError:
    try:
        from qiskit.providers.fake_provider import FakeNairobi as FakeBackend
        FAKE_AVAILABLE = True
    except ImportError:
        FAKE_AVAILABLE = False

from tsplib_utils import (
    generate_random_tsp, brute_force_tsp, load_casey_instance,
    get_all_casey_instance_ids
)

# ── Configuration ─────────────────────────────────────────────────────────────
PENALTY_DUPLICATE = 1000.0
PENALTY_OUT_OF_RANGE = 1000.0
QUBO_PENALTY_A = 1000.0
QUBO_WEIGHT_B = 1.0

RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')
CASEY_JSON = os.path.join(os.path.dirname(__file__), '..', '13_data gathering',
                          'cluster_complete_graphs.json')

ENCODINGS = ['node_visit_time', 'naive_adjacency']

CSV_FIELDS = [
    'timestamp', 'dataset', 'instance_name', 'n_cities', 'encoding',
    'qaoa_depth', 'shots',
    # Metric 1: qubit count
    'qubit_count',
    # Metric 2: gate count
    'gate_count_2q', 'circuit_depth_transpiled',
    # Metric 3: optimizer
    'optimizer_iterations', 'optimizer_evaluations',
    # Metric 4: compute time
    'time_transpile_s', 'time_optimize_s', 'time_total_s',
    # Metric 5: optimality
    'classical_optimal', 'expected_cost', 'optimality_ratio',
    'success', 'valid_count', 'invalid_count',
]


# =============================================================================
# QUBIT COUNT (theoretical)
# =============================================================================

def qubit_count_nvt(n):
    """NVT: (n-1) * ceil(log2(n)) — fix city 0 as start."""
    if n <= 1:
        return 0
    bits_per_city = math.ceil(math.log2(n)) if n > 1 else 1
    return (n - 1) * bits_per_city


def qubit_count_naive(n):
    """Naive one-hot: n^2."""
    return n * n


def qubit_count_log(n):
    """Logarithmic permutation: ceil(log2((n-1)!))."""
    if n <= 2:
        return 1
    factorial = math.factorial(n - 1)
    return math.ceil(math.log2(factorial))


# =============================================================================
# NODE-VISIT-TIME ENCODING
# =============================================================================

def decode_nvt(bitstring, n):
    """
    Decode NVT bitstring into a TSP tour.
    n = total cities (city 0 is fixed start).
    Remaining n-1 cities each have ceil(log2(n)) bits.

    Returns:
        tour: list of city indices [0, c1, c2, ..., c_{n-1}, 0]
        visit_times: decoded integer visit times for cities 1..n-1
    """
    bits_per = math.ceil(math.log2(n)) if n > 1 else 1
    n_customers = n - 1
    visit_times = []
    for i in range(n_customers):
        chunk = bitstring[i * bits_per: (i + 1) * bits_per]
        try:
            vt = int(chunk, 2)
        except ValueError:
            vt = -1
        visit_times.append(vt)

    # Sort cities 1..n-1 by their decoded visit time
    indexed = sorted(enumerate(visit_times), key=lambda x: x[1])
    ordered_cities = [i + 1 for i, _ in indexed]  # +1 because city 0 is fixed
    tour = [0] + ordered_cities + [0]
    return tour, visit_times


def tour_cost(tour, dist):
    """Total distance along a tour (list of city indices, first = last = depot)."""
    return sum(dist[tour[i]][tour[i+1]] for i in range(len(tour)-1))


def nvt_cost(bitstring, dist, n):
    """Full NVT cost including distance + duplicate + out-of-range penalties."""
    bits_per = math.ceil(math.log2(n)) if n > 1 else 1
    tour, visit_times = decode_nvt(bitstring, n)

    # Distance cost
    dist_cost = tour_cost(tour, dist)
    if math.isinf(dist_cost):
        dist_cost = 1e9

    # Duplicate penalty
    counts = Counter(visit_times)
    dup_penalty = PENALTY_DUPLICATE * sum(
        (c - 1) ** 2 for c in counts.values() if c > 1
    )

    # Out-of-range penalty (valid range: 1..n-1)
    range_penalty = PENALTY_OUT_OF_RANGE * sum(
        1 for vt in visit_times if vt < 1 or vt > n - 1
    )

    return dist_cost + dup_penalty + range_penalty


def is_valid_nvt(bitstring, n):
    """Return True if bitstring decodes to a valid permutation of cities 1..n-1."""
    _, visit_times = decode_nvt(bitstring, n)
    valid_set = set(range(1, n))
    return set(visit_times) == valid_set


# =============================================================================
# NAIVE ONE-HOT ENCODING
# =============================================================================

def decode_naive(bitstring, n):
    """
    Decode naive one-hot bitstring (n^2 qubits) into a TSP tour.
    x[i][t] = 1 if city i is at position t.

    Returns:
        tour: list of city indices [c0, c1, ..., c_{n-1}, c0]
              or None if invalid
        is_valid: bool
    """
    if len(bitstring) != n * n:
        return None, False

    x = np.array(list(map(int, bitstring))).reshape((n, n))
    tour = []
    for t in range(n):
        col = x[:, t]
        ones = np.where(col == 1)[0]
        if len(ones) == 1:
            tour.append(int(ones[0]))
        else:
            tour.append(-1)

    is_valid = (
        -1 not in tour and
        len(set(tour)) == n and
        set(tour) == set(range(n))
    )
    if is_valid:
        tour = tour + [tour[0]]
    return tour, is_valid


def naive_cost(bitstring, dist, n):
    """Full naive one-hot cost including QUBO penalties."""
    x = np.array(list(map(int, bitstring)), dtype=float).reshape((n, n))

    # Constraint 1: exactly one city per time step
    row_violation = sum((1 - x[:, t].sum()) ** 2 for t in range(n))
    # Constraint 2: each city visited exactly once
    col_violation = sum((1 - x[i, :].sum()) ** 2 for i in range(n))

    penalty = QUBO_PENALTY_A * (row_violation + col_violation)

    # Travel cost
    travel = 0.0
    for i in range(n):
        for j in range(n):
            for t in range(n):
                travel += dist[i][j] * x[i, t] * x[j, (t + 1) % n]

    return penalty + QUBO_WEIGHT_B * travel


# =============================================================================
# QAOA CIRCUIT
# =============================================================================

def build_qaoa_circuit(num_qubits, p, gamma, beta):
    """
    Build a QAOA circuit with p layers.
    Cost layer: RZ on all qubits + RZZ on all pairs.
    Mixer layer: RX on all qubits.
    """
    qc = QuantumCircuit(num_qubits, num_qubits)
    qc.h(range(num_qubits))
    qc.barrier()

    for k in range(p):
        # Cost layer
        for i in range(num_qubits):
            qc.rz(2 * gamma[k], i)
        for i in range(num_qubits):
            for j in range(i + 1, num_qubits):
                qc.rzz(2 * gamma[k], i, j)
        qc.barrier()
        # Mixer layer
        for i in range(num_qubits):
            qc.rx(2 * beta[k], i)
        qc.barrier()

    qc.measure(range(num_qubits), range(num_qubits))
    return qc


def get_transpiled_stats(circuit, backend):
    """
    Transpile circuit for backend and return:
      gate_count_2q: number of two-qubit gates
      circuit_depth: depth of transpiled circuit
    """
    t_start = time.perf_counter()
    transpiled = transpile(circuit, backend=backend, optimization_level=1)
    t_transpile = time.perf_counter() - t_start

    gate_count_2q = sum(
        1 for _, qargs, _ in transpiled.data if len(qargs) == 2
    )
    circuit_depth = transpiled.depth()
    return transpiled, gate_count_2q, circuit_depth, t_transpile


# =============================================================================
# EXPERIMENT RUNNER
# =============================================================================

def run_experiment(instance_name, n, dist, classical_optimal, encoding, p,
                   shots, backend, use_fake=False):
    """
    Run one QAOA experiment and return a results dict.
    """
    result = {
        'instance_name': instance_name,
        'n_cities': n,
        'encoding': encoding,
        'qaoa_depth': p,
        'shots': shots,
    }

    # ── Qubit count ─────────────────────────────────────────────────────────
    if encoding == 'node_visit_time':
        num_qubits = qubit_count_nvt(n)
        cost_fn = lambda bs: nvt_cost(bs, dist, n)
        valid_fn = lambda bs: is_valid_nvt(bs, n)
    else:  # naive_adjacency
        num_qubits = qubit_count_naive(n)
        cost_fn = lambda bs: naive_cost(bs, dist, n)
        valid_fn = lambda bs: decode_naive(bs, n)[1]

    result['qubit_count'] = num_qubits
    result['classical_optimal'] = classical_optimal

    if num_qubits == 0:
        result['success'] = False
        return result

    # ── Build initial circuit (for transpilation stats) ─────────────────────
    init_gamma = np.random.uniform(0, 2 * np.pi, p)
    init_beta  = np.random.uniform(0, np.pi, p)
    qc_init = build_qaoa_circuit(num_qubits, p, init_gamma, init_beta)

    transpiled_init, gate_count_2q, circuit_depth, t_transpile = \
        get_transpiled_stats(qc_init, backend)

    result['gate_count_2q'] = gate_count_2q
    result['circuit_depth_transpiled'] = circuit_depth
    result['time_transpile_s'] = round(t_transpile, 4)

    # ── COBYLA optimization loop ─────────────────────────────────────────────
    optimizer_evals = [0]
    optimizer_iters = [0]
    cost_history = []

    def objective(params):
        gamma = params[:p]
        beta  = params[p:]
        qc = build_qaoa_circuit(num_qubits, p, gamma, beta)
        transpiled = transpile(qc, backend=backend, optimization_level=1)

        # Run on backend
        if use_fake:
            from qiskit_ibm_runtime import StatevectorSampler
            sampler = StatevectorSampler()
            job = sampler.run([transpiled], shots=shots)
        else:
            sampler = Sampler(backend)
            job = sampler.run([transpiled], shots=shots)

        result_obj = job.result()
        counts = result_obj[0].data.meas.get_counts()

        # Expected cost over all measured bitstrings
        total_shots = sum(counts.values())
        expected = 0.0
        for bs, cnt in counts.items():
            bs_str = bs if isinstance(bs, str) else format(bs, f'0{num_qubits}b')
            expected += (cnt / total_shots) * cost_fn(bs_str)

        optimizer_evals[0] += 1
        cost_history.append(expected)
        return expected

    def callback(xk):
        optimizer_iters[0] += 1

    x0 = np.concatenate([init_gamma, init_beta])
    t_opt_start = time.perf_counter()

    try:
        opt_result = minimize(
            objective, x0, method='COBYLA',
            options={'maxiter': 200, 'rhobeg': 1.0},
            callback=callback
        )
        converged = opt_result.success
    except Exception as e:
        print(f"  Optimizer failed: {e}")
        converged = False

    t_opt_end = time.perf_counter()
    t_optimize = t_opt_end - t_opt_start

    result['optimizer_iterations'] = optimizer_iters[0]
    result['optimizer_evaluations'] = optimizer_evals[0]
    result['time_optimize_s'] = round(t_optimize, 2)
    result['time_total_s'] = round(t_transpile + t_optimize, 2)

    # ── Final measurement pass ───────────────────────────────────────────────
    best_params = opt_result.x if converged else x0
    gamma_best = best_params[:p]
    beta_best  = best_params[p:]
    qc_final = build_qaoa_circuit(num_qubits, p, gamma_best, beta_best)
    transpiled_final = transpile(qc_final, backend=backend, optimization_level=1)

    if use_fake:
        from qiskit_ibm_runtime import StatevectorSampler
        sampler = StatevectorSampler()
        job = sampler.run([transpiled_final], shots=shots)
    else:
        sampler = Sampler(backend)
        job = sampler.run([transpiled_final], shots=shots)

    counts = job.result()[0].data.meas.get_counts()

    # ── Compute metrics from final counts ────────────────────────────────────
    valid_costs = []
    valid_count = 0
    invalid_count = 0

    for bs, cnt in counts.items():
        bs_str = bs if isinstance(bs, str) else format(bs, f'0{num_qubits}b')
        if valid_fn(bs_str):
            c = cost_fn(bs_str)
            if not math.isinf(c) and c < 1e8:
                valid_costs.extend([c] * cnt)
                valid_count += cnt
        else:
            invalid_count += cnt

    result['valid_count'] = valid_count
    result['invalid_count'] = invalid_count
    result['success'] = valid_count > 0

    if valid_count > 0:
        expected_valid = sum(valid_costs) / len(valid_costs)
        result['expected_cost'] = round(expected_valid, 6)
        result['optimality_ratio'] = round(classical_optimal / expected_valid, 6) \
            if expected_valid > 0 else float('nan')
    else:
        result['expected_cost'] = float('nan')
        result['optimality_ratio'] = float('nan')

    return result


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='TSP QAOA Publication Experiment Runner')
    parser.add_argument('--fake', action='store_true',
                        help='Use fake (simulated) IBM backend instead of real hardware')
    parser.add_argument('--backend', default='ibm_brisbane',
                        help='IBM backend name (ignored if --fake)')
    parser.add_argument('--shots', type=int, default=1024)
    parser.add_argument('--depth', type=int, default=3)
    parser.add_argument('--nodes', default='3,4,5,6',
                        help='Comma-separated list of city counts to test '
                             '(for TSPLIB random instances)')
    parser.add_argument('--dataset', choices=['casey', 'tsplib', 'both'],
                        default='both', help='Which dataset to use')
    parser.add_argument('--encodings', default='node_visit_time,naive_adjacency',
                        help='Comma-separated list of encodings to test')
    parser.add_argument('--runs', type=int, default=1,
                        help='Number of repeated runs per instance (for averaging)')
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    encodings = args.encodings.split(',')
    node_sizes = [int(x) for x in args.nodes.split(',')]

    # ── Set up backend ───────────────────────────────────────────────────────
    if args.fake:
        if not FAKE_AVAILABLE:
            raise RuntimeError("Fake provider not available. Install qiskit-ibm-runtime.")
        backend = FakeBackend()
        print(f"Using fake backend: {backend.name}")
    else:
        if not IBM_AVAILABLE:
            raise RuntimeError("qiskit-ibm-runtime not installed.")
        service = QiskitRuntimeService()
        backend = service.backend(args.backend)
        print(f"Using IBM backend: {backend.name}")

    # ── Collect instances ────────────────────────────────────────────────────
    instances = []  # list of (name, n, dist_matrix, classical_optimal)

    if args.dataset in ('casey', 'both'):
        if os.path.exists(CASEY_JSON):
            all_ids = get_all_casey_instance_ids(CASEY_JSON)
            for cluster_type, tid in all_ids:
                try:
                    name, n, dist, _, opt = load_casey_instance(
                        CASEY_JSON, cluster_type, tid
                    )
                    if n in node_sizes:
                        instances.append(('casey', name, n, dist, opt))
                        print(f"Loaded Casey instance: {name} (n={n}, opt={opt:.3f})")
                except Exception as e:
                    print(f"  Skipped Casey {cluster_type} {tid}: {e}")
        else:
            print(f"Casey JSON not found at {CASEY_JSON}, skipping Casey instances.")

    if args.dataset in ('tsplib', 'both'):
        for n in node_sizes:
            for seed in range(3):  # 3 random instances per size
                name, n_out, dist, coords = generate_random_tsp(n, seed=seed * 100 + n)
                opt, _ = brute_force_tsp(dist, n_out)
                instances.append(('tsplib', name, n_out, dist, opt))
                print(f"Generated TSPLIB instance: {name} (n={n_out}, opt={opt:.3f})")

    print(f"\nTotal instances: {len(instances)}")
    print(f"Encodings: {encodings}")
    print(f"QAOA depth: {args.depth}, shots: {args.shots}, runs: {args.runs}")
    print()

    # ── Output CSV ───────────────────────────────────────────────────────────
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    csv_path = os.path.join(RESULTS_DIR, f'tsp_results_{ts}.csv')

    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()

        for dataset_label, name, n, dist, opt in instances:
            for enc in encodings:
                for run in range(args.runs):
                    print(f"  [{dataset_label}] {name} | n={n} | {enc} | "
                          f"p={args.depth} | shots={args.shots} | run={run+1}/{args.runs}")
                    try:
                        res = run_experiment(
                            instance_name=name,
                            n=n,
                            dist=dist,
                            classical_optimal=opt,
                            encoding=enc,
                            p=args.depth,
                            shots=args.shots,
                            backend=backend,
                            use_fake=args.fake,
                        )
                        res['timestamp'] = ts
                        res['dataset'] = dataset_label

                        # Fill any missing fields with empty string
                        row = {field: res.get(field, '') for field in CSV_FIELDS}
                        writer.writerow(row)
                        f.flush()

                        print(f"    -> OR={res.get('optimality_ratio', 'NaN')} | "
                              f"valid={res.get('valid_count', 0)} | "
                              f"gates={res.get('gate_count_2q', '?')} | "
                              f"iters={res.get('optimizer_iterations', '?')} | "
                              f"time={res.get('time_total_s', '?')}s")

                    except Exception as e:
                        print(f"    ERROR: {e}")
                        row = {field: '' for field in CSV_FIELDS}
                        row.update({'timestamp': ts, 'dataset': dataset_label,
                                    'instance_name': name, 'n_cities': n,
                                    'encoding': enc, 'qaoa_depth': args.depth,
                                    'shots': args.shots, 'success': False})
                        writer.writerow(row)

    print(f"\nResults saved to: {csv_path}")

    # ── Quick summary ────────────────────────────────────────────────────────
    print("\n=== Qubit count summary (theoretical) ===")
    print(f"{'n':>6} {'Naive':>8} {'Log':>8} {'NVT':>8}")
    for n in sorted(set(node_sizes)):
        print(f"{n:>6} {qubit_count_naive(n):>8} "
              f"{qubit_count_log(n):>8} {qubit_count_nvt(n):>8}")


if __name__ == '__main__':
    main()
