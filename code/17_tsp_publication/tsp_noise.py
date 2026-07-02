"""
tsp_noise.py
Noise sensitivity analysis for TSP QAOA encodings.

Runs NVT and naive one-hot under:
  (a) Ideal simulation (statevector)
  (b) Depolarizing noise at multiple error rates
  (c) Thermal relaxation (T1/T2 decoherence)
  (d) Real IBM hardware (if available)

Requires: qiskit-aer (pip install qiskit-aer)

Usage:
  python tsp_noise.py --n 5 --depth 3 --shots 1024
  python tsp_noise.py --n 5 --depth 3 --shots 1024 --real-hardware --backend ibm_brisbane
"""

import os
import csv
import math
import time
import argparse
import itertools
from collections import Counter
from datetime import datetime

import numpy as np
from scipy.optimize import minimize
from qiskit import QuantumCircuit, transpile

try:
    from qiskit_aer import AerSimulator
    from qiskit_aer.noise import (
        NoiseModel, depolarizing_error, thermal_relaxation_error
    )
    AER_AVAILABLE = True
except ImportError:
    AER_AVAILABLE = False
    print("WARNING: qiskit-aer not installed. Install with: pip install qiskit-aer")

from tsplib_utils import generate_random_tsp, brute_force_tsp
from tsp_runner import (
    qubit_count_nvt, qubit_count_naive,
    decode_nvt, decode_naive, nvt_cost, naive_cost,
    is_valid_nvt, build_qaoa_circuit, tour_cost
)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')

NOISE_SCENARIOS = {
    'ideal':            {'type': 'ideal'},
    'depolar_1e-3':     {'type': 'depolarizing', 'p1': 1e-3, 'p2': 1e-2},
    'depolar_5e-3':     {'type': 'depolarizing', 'p1': 5e-3, 'p2': 5e-2},
    'depolar_1e-2':     {'type': 'depolarizing', 'p1': 1e-2, 'p2': 1e-1},
    'thermal_200us':    {'type': 'thermal', 'T1': 200e-6, 'T2': 100e-6, 'gate_time_1q': 50e-9, 'gate_time_2q': 300e-9},
    'thermal_100us':    {'type': 'thermal', 'T1': 100e-6, 'T2': 50e-6, 'gate_time_1q': 50e-9, 'gate_time_2q': 300e-9},
    'thermal_50us':     {'type': 'thermal', 'T1': 50e-6,  'T2': 25e-6, 'gate_time_1q': 50e-9, 'gate_time_2q': 300e-9},
}

CSV_FIELDS = [
    'timestamp', 'n_cities', 'encoding', 'qaoa_depth', 'shots',
    'noise_scenario', 'qubit_count',
    'optimality_ratio', 'success_rate', 'valid_fraction',
    'classical_optimal', 'expected_cost',
]


def build_noise_model(scenario):
    """Build a Qiskit Aer NoiseModel from a scenario dict."""
    if not AER_AVAILABLE:
        raise RuntimeError("qiskit-aer required for noise models.")

    nm = NoiseModel()
    stype = scenario['type']

    if stype == 'ideal':
        return None

    elif stype == 'depolarizing':
        p1 = scenario['p1']  # single-qubit error rate
        p2 = scenario['p2']  # two-qubit error rate
        err_1q = depolarizing_error(p1, 1)
        err_2q = depolarizing_error(p2, 2)
        nm.add_all_qubit_quantum_error(err_1q, ['rx', 'rz', 'h'])
        nm.add_all_qubit_quantum_error(err_2q, ['cx', 'rzz'])
        return nm

    elif stype == 'thermal':
        T1 = scenario['T1']
        T2 = scenario['T2']
        t1q = scenario['gate_time_1q']
        t2q = scenario['gate_time_2q']
        err_1q = thermal_relaxation_error(T1, T2, t1q)
        err_2q = thermal_relaxation_error(T1, T2, t2q).expand(
                 thermal_relaxation_error(T1, T2, t2q))
        nm.add_all_qubit_quantum_error(err_1q, ['rx', 'rz', 'h'])
        nm.add_all_qubit_quantum_error(err_2q, ['cx', 'rzz'])
        return nm

    else:
        raise ValueError(f"Unknown noise scenario type: {stype}")


def run_noise_experiment(n, dist, classical_optimal, encoding, p, shots,
                         scenario_name, scenario, n_trials=5):
    """
    Run QAOA under a specific noise scenario and return performance metrics.
    Averages over n_trials random parameter initializations to smooth variance.
    """
    if encoding == 'node_visit_time':
        num_qubits = qubit_count_nvt(n)
        cost_fn = lambda bs: nvt_cost(bs, dist, n)
        valid_fn = lambda bs: is_valid_nvt(bs, n)
    else:
        num_qubits = qubit_count_naive(n)
        cost_fn = lambda bs: naive_cost(bs, dist, n)
        valid_fn = lambda bs: decode_naive(bs, n)[1]

    # Build simulator
    noise_model = build_noise_model(scenario)
    if scenario['type'] == 'ideal':
        sim = AerSimulator(method='statevector')
    else:
        sim = AerSimulator(noise_model=noise_model)

    all_valid_fractions = []
    all_optimality_ratios = []

    for trial in range(n_trials):
        gamma = np.random.uniform(0, 2 * np.pi, p)
        beta  = np.random.uniform(0, np.pi, p)
        qc = build_qaoa_circuit(num_qubits, p, gamma, beta)
        qc_t = transpile(qc, sim, optimization_level=0)
        job = sim.run(qc_t, shots=shots)
        counts = job.result().get_counts()

        valid_costs = []
        valid_count = 0
        for bs, cnt in counts.items():
            bs_str = bs[::-1]  # Aer returns in reversed order
            if valid_fn(bs_str):
                c = cost_fn(bs_str)
                if not math.isinf(c) and c < 1e8:
                    valid_costs.extend([c] * cnt)
                    valid_count += cnt

        total = sum(counts.values())
        valid_frac = valid_count / total if total > 0 else 0.0
        all_valid_fractions.append(valid_frac)

        if valid_costs:
            exp_cost = sum(valid_costs) / len(valid_costs)
            or_val = classical_optimal / exp_cost if exp_cost > 0 else float('nan')
            all_optimality_ratios.append(or_val)

    avg_valid_frac = np.mean(all_valid_fractions)
    avg_or = np.nanmean(all_optimality_ratios) if all_optimality_ratios else float('nan')
    success_rate = sum(1 for vf in all_valid_fractions if vf > 0) / n_trials

    return {
        'qubit_count': num_qubits,
        'valid_fraction': round(avg_valid_frac, 4),
        'optimality_ratio': round(avg_or, 6),
        'success_rate': round(success_rate, 4),
        'classical_optimal': classical_optimal,
        'expected_cost': round(classical_optimal / avg_or, 6) if avg_or > 0 else float('nan'),
    }


def main():
    parser = argparse.ArgumentParser(description='TSP QAOA Noise Sensitivity Analysis')
    parser.add_argument('--n', type=int, default=5, help='Number of TSP cities')
    parser.add_argument('--depth', type=int, default=3)
    parser.add_argument('--shots', type=int, default=1024)
    parser.add_argument('--trials', type=int, default=5,
                        help='Random parameter initializations per scenario')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    if not AER_AVAILABLE:
        print("ERROR: Install qiskit-aer: pip install qiskit-aer")
        return

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Generate a random TSP instance
    name, n, dist, coords = generate_random_tsp(args.n, seed=args.seed)
    optimal, _ = brute_force_tsp(dist, n)
    print(f"Instance: {name}, n={n}, optimal={optimal:.4f}")
    print(f"Scenarios: {list(NOISE_SCENARIOS.keys())}")
    print()

    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    csv_path = os.path.join(RESULTS_DIR, f'noise_sensitivity_{ts}.csv')

    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()

        for scenario_name, scenario in NOISE_SCENARIOS.items():
            for encoding in ['node_visit_time', 'naive_adjacency']:
                print(f"  {scenario_name} | {encoding} ...", end=' ', flush=True)
                try:
                    res = run_noise_experiment(
                        n=n, dist=dist, classical_optimal=optimal,
                        encoding=encoding, p=args.depth, shots=args.shots,
                        scenario_name=scenario_name, scenario=scenario,
                        n_trials=args.trials,
                    )
                    row = {
                        'timestamp': ts,
                        'n_cities': n,
                        'encoding': encoding,
                        'qaoa_depth': args.depth,
                        'shots': args.shots,
                        'noise_scenario': scenario_name,
                        **res,
                    }
                    writer.writerow(row)
                    f.flush()
                    print(f"OR={res['optimality_ratio']} valid={res['valid_fraction']:.1%}")
                except Exception as e:
                    print(f"ERROR: {e}")

    print(f"\nResults saved to: {csv_path}")


if __name__ == '__main__':
    main()
