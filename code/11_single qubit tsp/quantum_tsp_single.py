# single_qubit_tsp.py
import json
import math
import cmath
import numpy as np
from math import pi
from collections import defaultdict
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Operator
from qiskit.circuit.library import UnitaryGate
from qiskit.providers.fake_provider import GenericBackendV2
# Changed import to use scipy's minimize function
from scipy.optimize import minimize
from scipy.linalg import norm

# ---------- User config ----------
JSON_FILE_PATH = 'cluster_complete_graphs.json'
cluster_type = 'waste_facility'
target_id = 5

SHOTS = 4096
# Changed SPSA_MAXITER to COBYLA_MAXITER
COBYLA_MAXITER = 200
VARY_NUM_OPS = None
# ---------------------------------

def load_distance_matrix(json_file_path, cluster_type, target_id):
    with open(json_file_path, 'r') as f:
        all_cluster_graphs = json.load(f)
    if cluster_type == 'waste_facility':
        cluster_list = all_cluster_graphs.get('waste_areas_to_facilities_graphs', [])
        id_key = 'waste_facility_id'
    else:
        cluster_list = all_cluster_graphs.get('litter_bins_to_waste_areas_graphs', [])
        id_key = 'waste_collection_area_id'
    for c in cluster_list:
        if int(c.get(id_key)) == target_id:
            raw = c.get('distance_matrix_km', {})
            M = {}
            for r_str, row in raw.items():
                r = int(r_str)
                M[r] = {}
                for c_str, v in row.items():
                    M[r][int(c_str)] = float(v)
            nodes = sorted(M.keys())
            return M, nodes
    raise FileNotFoundError("cluster or distance matrix not found")

def equator_states(n):
    return [np.array([math.cos(2*pi*k/n), math.sin(2*pi*k/n), 0.0]) for k in range(n)]

def geodesic_state(base_vec, pole_vec, frac):
    dot = np.dot(base_vec, pole_vec)
    dot = max(-1.0, min(1.0, dot))
    angle = math.acos(dot)
    if angle == 0:
        return base_vec.copy()
    axis = np.cross(base_vec, pole_vec)
    axis_norm = norm(axis)
    if axis_norm < 1e-12:
        return base_vec * (1-frac) + pole_vec * frac
    n = axis / axis_norm
    a = frac * angle
    v = (base_vec * math.cos(a)
          + np.cross(n, base_vec) * math.sin(a)
          + n * (np.dot(n, base_vec)) * (1 - math.cos(a)))
    return v / norm(v)

def bloch_vector_to_statevec(v):
    x, y, z = v
    # These functions operate on real numbers, so use math
    theta = math.acos(max(-1.0, min(1.0, z)))
    phi = math.atan2(y, x)
    
    a = math.cos(theta/2)
    
    # This operation involves a complex number (1j), so use cmath
    b = cmath.exp(1j*phi) * math.sin(theta/2)
    
    return np.array([a, b], dtype=complex)

def unitary_take_a_to_b(a_vec, b_vec):
    a = np.array(a_vec, dtype=float) / norm(a_vec)
    b = np.array(b_vec, dtype=float) / norm(b_vec)
    dot = np.clip(np.dot(a, b), -1.0, 1.0)
    delta = math.acos(dot)
    if delta < 1e-12:
        return np.eye(2, dtype=complex)
    n = np.cross(a, b)
    if norm(n) < 1e-12:
        n = np.array([1.0, 0.0, 0.0]) if abs(a[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    else:
        n = n / norm(n)
    nx, ny, nz = n
    sx = np.array([[0, 1],[1, 0]], dtype=complex)
    sy = np.array([[0, -1j],[1j, 0]], dtype=complex)
    sz = np.array([[1, 0],[0, -1]], dtype=complex)
    n_dot_sigma = nx*sx + ny*sy + nz*sz
    return math.cos(delta/2)*np.eye(2, dtype=complex) - 1j*math.sin(delta/2)*n_dot_sigma

def build_param_circuit(initial_state_vec, target_ops, params):
    qc = QuantumCircuit(1)
    z_axis = np.array([0.0,0.0,1.0])
    z = abs(initial_state_vec[0])**2 - abs(initial_state_vec[1])**2
    x = 2*np.real(np.conj(initial_state_vec[0])*initial_state_vec[1])
    y = 2*np.imag(np.conj(initial_state_vec[0])*initial_state_vec[1])
    target_bloch = np.array([x,y,z])
    U0 = unitary_take_a_to_b(z_axis, target_bloch)
    qc.append(UnitaryGate(U0), [0])
    for (axis_vec, angle_scale), param in zip(target_ops, params):
        nx, ny, nz = axis_vec
        sx = np.array([[0, 1],[1, 0]], dtype=complex)
        sy = np.array([[0, -1j],[1j, 0]], dtype=complex)
        sz = np.array([[1, 0],[0, -1]], dtype=complex)
        n_dot_sigma = nx*sx + ny*sy + nz*sz
        angle = float(param) * float(angle_scale)
        U = math.cos(angle/2)*np.eye(2, dtype=complex) - 1j*math.sin(angle/2)*n_dot_sigma
        qc.append(UnitaryGate(U), [0])
    return qc

def tomography_expectations(qc, backend, shots=2048):
    circuits = []
    qc_z = qc.copy()
    qc_z.measure_all()
    circuits.append(qc_z)
    qc_x = qc.copy()
    qc_x.h(0)
    qc_x.measure_all()
    circuits.append(qc_x)
    qc_y = qc.copy()
    qc_y.sdg(0)
    qc_y.h(0)
    qc_y.measure_all()
    circuits.append(qc_y)
    transpiled = [transpile(c, backend=backend, basis_gates=['u3', 'rz', 'sx', 'x', 'h', 'id']) for c in circuits]
    job = backend.run(transpiled, shots=shots)
    res = job.result()
    exps = []
    for circ in transpiled:
        counts = res.get_counts(circ)
        p0 = counts.get('0', 0) / shots
        p1 = counts.get('1', 0) / shots
        exps.append(p0 - p1)
    return tuple(exps)

def bloch_vector_to_statevec(v):
    x, y, z = v
    # These functions operate on real numbers, so use math
    theta = math.acos(max(-1.0, min(1.0, z)))
    phi = math.atan2(y, x)
    
    a = math.cos(theta/2)
    
    # This operation involves a complex number (1j), so use cmath
    b = cmath.exp(1j*phi) * math.sin(theta/2)
    
    return np.array([a, b], dtype=complex)

def decode_penultimate(state_vec, basis_statevecs):
    n = len(basis_statevecs)
    E = np.zeros((n, n), dtype=complex)
    K = np.zeros(n, dtype=complex)
    for i in range(n):
        for j in range(n):
            E[i,j] = np.vdot(basis_statevecs[i], basis_statevecs[j])
        K[i] = np.vdot(basis_statevecs[i], state_vec)
    try:
        X = np.linalg.solve(E, K)
    except np.linalg.LinAlgError:
        X, *_ = np.linalg.lstsq(E, K, rcond=None)
    return X

def cost_from_order(order, distance_matrix):
    return sum(distance_matrix[order[i]][order[(i+1)%len(order)]] for i in range(len(order)))

def run_single_qubit_tsp(distance_matrix, nodes):
    n = len(nodes)
    equators = equator_states(n)
    pole = np.array([0.0, 0.0, 1.0])
    maxd = max(distance_matrix[i][j] for i in nodes for j in nodes if i != j)
    initial_idx = 0
    initial_equator_vec = equators[initial_idx]
    basis_statevecs = []
    for j_idx in range(1, n):
        s = distance_matrix[nodes[j_idx]][nodes[initial_idx]]
        frac = min(1.0, s / maxd)
        base_vec = equators[j_idx]
        gvec = geodesic_state(base_vec, pole, frac)
        basis_statevecs.append(bloch_vector_to_statevec(gvec))
    num_ops = VARY_NUM_OPS or max(2*n, 8)
    target_ops = []
    for k in range(num_ops):
        i = k % n
        base = equators[i]
        axis = np.cross(base, pole)
        if norm(axis) < 1e-8:
            axis = np.array([1.0, 0.0, 0.0])
        axis = axis / norm(axis)
        target_ops.append((axis, pi/2.0))

    backend = GenericBackendV2(num_qubits=1, basis_gates=['id', 'rz', 'sx', 'x', 'h', 'u'])

    def objective_fn(params):
        initial_state_vec = bloch_vector_to_statevec(initial_equator_vec)
        qc = build_param_circuit(initial_state_vec, target_ops, params)
        expx, expy, expz = tomography_expectations(qc, backend, shots=SHOTS)
        g_vec = bloch_vector_to_statevec((expx, expy, expz))
        betas = decode_penultimate(g_vec, basis_statevecs)
        mags = np.abs(betas)**2
        ordering_indices = [0] + [1 + int(i) for i in np.argsort(-mags)]
        route_nodes = [nodes[i] for i in ordering_indices]
        c = cost_from_order(route_nodes, distance_matrix)
        print("Trial params (first 4):", np.round(params[:4],3), " -> cost:", c)
        return c

    x0 = np.random.uniform(low=-1.0, high=1.0, size=(num_ops,))

    # Use minimize from scipy.optimize with method='COBYLA'
    res = minimize(fun=objective_fn, x0=x0, method='COBYLA', options={'maxiter': COBYLA_MAXITER})

    x_opt = res.x
    print("Optimization finished (COBYLA). Best params (first 8):", np.round(x_opt[:8], 3))
    initial_state_vec = bloch_vector_to_statevec(initial_equator_vec)
    qc_final = build_param_circuit(initial_state_vec, target_ops, x_opt)
    expx, expy, expz = tomography_expectations(qc_final, backend, shots=SHOTS)
    g_vec = bloch_vector_to_statevec((expx, expy, expz))
    betas = decode_penultimate(g_vec, basis_statevecs)
    order = [0] + [1 + int(i) for i in np.argsort(-np.abs(betas)**2)]
    route_nodes = [nodes[i] for i in order]
    final_cost = cost_from_order(route_nodes, distance_matrix)
    print("Final candidate route:", route_nodes, "cost:", final_cost)
    return route_nodes, final_cost

if __name__ == "__main__":
    distance_matrix, nodes = load_distance_matrix(JSON_FILE_PATH, cluster_type, target_id)
    print("Nodes:", nodes)
    route, c = run_single_qubit_tsp(distance_matrix, nodes)
    print("Done; route:", route, "cost:", c)