from qiskit import QuantumCircuit
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit.primitives import Sampler

def run_real_quantum_circuit():
    # Load IBM Quantum account (make sure you saved your token before running this)
    service = QiskitRuntimeService(channel="ibm_quantum")

    # Create a simple 2-qubit entangled circuit
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure_all()

    # Create a Sampler using the IBM Quantum runtime service
    sampler = Sampler(session=service)

    # Run the circuit on a real backend (or simulator) available via runtime
    job = sampler.run([qc])

    result = job.result()

    # Print the measurement quasi-probabilities
    print("Measurement probabilities on real quantum backend:")
    for outcome, prob in result.quasi_dists[0].items():
        print(f"  {outcome} : {prob:.4f}")

if __name__ == "__main__":
    run_real_quantum_circuit()
