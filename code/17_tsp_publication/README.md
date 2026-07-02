# 17_tsp_publication — Experiment Code for TSP QAOA Paper

This directory contains the experiment scripts for the arXiv publication:

> **Qubit-Efficient Encoding Strategies for the Travelling Salesman Problem via QAOA**
> Dyasputro, Bhat, Parampalli (2026)

---

## Files

| File | Purpose |
|------|---------|
| `tsp_runner.py` | Main experiment runner — collects ALL supervisor-required metrics |
| `tsp_noise.py` | Noise sensitivity analysis (depolarizing, thermal relaxation) |
| `tsplib_utils.py` | TSPLIB instance loader + random TSP generator + brute-force solver |
| `results/` | Output CSVs (created on first run) |

---

## Metrics Collected

Per supervisor approval (Shashank Bhat email, July 2026):

| # | Metric | Script | Status |
|---|--------|--------|--------|
| 1 | Qubit count | `tsp_runner.py` (theoretical, printed in summary) | ✅ Theoretical |
| 2 | Gate count (2-qubit, transpiled) | `tsp_runner.py` | 🔲 Run needed |
| 3 | Optimizer iterations + evaluations | `tsp_runner.py` | 🔲 Run needed |
| 4 | Compute time (wall clock) | `tsp_runner.py` | 🔲 Run needed |
| 5 | Optimality ratio | `tsp_runner.py` | ✅ Have thesis data; new runs extend it |
| 6 | Noise sensitivity | `tsp_noise.py` | 🔲 Run needed |
| 7 | Scalability / breakpoint | `tsp_runner.py --nodes 3,4,5,6,7,8,10` | 🔲 7–10 nodes needed |

---

## Quick Start

### Option A: Local simulation (no IBM account needed)

```bash
pip install qiskit qiskit-ibm-runtime qiskit-aer scipy numpy

# Main experiment (fake backend, 5-city instances)
python tsp_runner.py --fake --nodes 3,4,5,6 --shots 1024 --depth 3

# Noise sensitivity
python tsp_noise.py --n 5 --depth 3 --shots 1024
```

### Option B: Real IBM Quantum hardware

```bash
# Set up IBM Quantum account (one-time)
python -c "from qiskit_ibm_runtime import QiskitRuntimeService; QiskitRuntimeService.save_account(channel='ibm_quantum', token='YOUR_TOKEN')"

# Run experiments
python tsp_runner.py --backend ibm_brisbane --shots 1024 --depth 3 --nodes 3,4,5,6

# Extended scalability (7–10 nodes)
python tsp_runner.py --backend ibm_brisbane --shots 1024 --depth 3 --nodes 7,8,10
```

---

## Dataset Options

### City of Casey (real-world)
Loaded automatically from `../13_data gathering/cluster_complete_graphs.json`.
23 instances, 3–6 cities, distances in km.
```bash
python tsp_runner.py --fake --dataset casey
```

### TSPLIB-style random instances
Generates random Euclidean instances (reproducible with seed).
```bash
python tsp_runner.py --fake --dataset tsplib --nodes 3,4,5,6,7,8
```

### Both (default)
```bash
python tsp_runner.py --fake --dataset both
```

---

## Output Format

Results are saved to `results/tsp_results_TIMESTAMP.csv` with columns:

```
timestamp, dataset, instance_name, n_cities, encoding, qaoa_depth, shots,
qubit_count, gate_count_2q, circuit_depth_transpiled,
optimizer_iterations, optimizer_evaluations,
time_transpile_s, time_optimize_s, time_total_s,
classical_optimal, expected_cost, optimality_ratio,
success, valid_count, invalid_count
```

---

## Generating Paper Figures

After running experiments, use the Jupyter notebook in `../14_data analysis/` or
create a new analysis notebook to plot:

1. **Gate count comparison**: `gate_count_2q` vs `n_cities` for each encoding
2. **Optimizer convergence**: cost vs iteration number (logged internally per run)
3. **Compute time**: `time_total_s` vs `n_cities`
4. **Noise sensitivity**: OR vs noise level (from `tsp_noise.py` output)
5. **Scalability**: `optimality_ratio` and `success` vs `n_cities` (extend existing Fig. 7)

All figures should be saved at 300 DPI to `../../../final/figures/`.

---

## Notes

- **Encoding summary**:
  - `node_visit_time`: (n-1)·⌈log₂n⌉ qubits, city 0 fixed as start
  - `naive_adjacency`: n² qubits, one-hot QUBO
  - Logarithmic permutation: ⌈log₂(n-1)!⌉ qubits — impractical for QAOA (not implemented as QAOA circuit)

- **COBYLA optimizer**: gradient-free, robust to NISQ noise. Max 200 iterations.

- **Default QAOA parameters**: p=3 depth, 1024 shots. Results stable for p=3–5, 512–1024 shots (from thesis experiments).
