import numpy as np # Ensure numpy is imported
from qiskit_ibm_runtime import QiskitRuntimeService, Sampler
from qiskit import QuantumCircuit, transpile
from collections import Counter

def format_measurement_counts(bit_array_data):
    """
    Takes a BitArray object from Sampler results and returns a Counter
    with bitstring keys formatted to the correct number of qubits.
    """
    # Extract the number of bits (qubits) from the BitArray object
    num_qubits = bit_array_data.num_bits

    # Access raw numpy array of shape (num_shots, num_qubits)
    meas_array = meas.array

    # Convert each shot to bitstring (reverse for Qiskit ordering)
    bitstrings = ["".join(str(bit) for bit in shot[::-1]) for shot in meas_array]

    # Count occurrences of each bitstring
    counts = Counter(bitstrings)
    d = dict(counts)
    return {format(int(k), f'0{num_qubits}b'): v for k, v in d.items()}


print("Starting script...")

# STEP 0: Load API key
print("\nSTEP 0: Loading API key...")
my_token = "0hoS4-Q7nUHa75Q_6CgsjsxRytnnYJt5JfXGOCQo3pGD"
QiskitRuntimeService.save_account(
    token=my_token,
    channel="ibm_cloud", # Or "ibm_quantum"
    instance="crn:v1:bluemix:public:quantum-computing:us-east:a/7cdd1aaa459f4725bfd9cf0bf5235c8a:1a968479-f9fb-44d8-930a-06e97c63335d::",
    overwrite=True
)
print("API key and instance saved successfully!")

# STEP 1: Load your IBM Quantum credentials
print("\nSTEP 1: Loading IBM Quantum credentials...")
service = QiskitRuntimeService()
print("IBM Quantum service loaded.")

# STEP 2: Choose a backend (real device or simulator)
print("\nSTEP 2: Choosing a backend...")
# Remember to replace "ibm_qasm_simulator" with a valid simulator name from your account if it's not found.
# E.g., backend = service.backend("simulator_statevector")
backend = service.backend("ibm_sherbrooke") # This might need to be adjusted based on your available backends.
print(f"Backend selected: {backend.name}")

# --- New variable for number of qubits ---
num_qubits = 2 # You can adjust this value later for other circuits!
# ------------------------------------------

# STEP 3: Create a simple quantum circuit
print("\nSTEP 3: Creating a simple quantum circuit...")
qc = QuantumCircuit(num_qubits) # Use the num_qubits variable here
qc.h(0)
qc.cx(0, 1)
qc.measure_all()
print(f"Quantum circuit created with {num_qubits} qubits.")

# Transpile the circuit to fit backend basis gates and topology
print("Transpiling circuit...")
qc_transpiled = transpile(qc, backend=backend)
print("Circuit transpiled successfully.")

# STEP 4: Use the Sampler primitive with the transpiled circuit
print("\nSTEP 4: Initializing Sampler and running job...")
sampler = Sampler(mode=backend)
print("Sampler initialized.")
job = sampler.run([qc_transpiled], shots=256) # Adding shots parameter for clarity
print(f"Job submitted with ID: {job.job_id}. Waiting for results...")

# STEP 5: Fetch and process results
print("\nSTEP 5: Fetching and processing results...")
result = job.result()
print("Raw result object:", result)

# Extract measurement bitstrings (BitArray object)
print("Extracting measurement data...")
meas = result[0].data.meas
print("Measurement data extracted.")

# --- Use the new function to get formatted counts ---
print("\nSTEP 6: Formatting measurement counts...")
formatted_counts = format_measurement_counts(meas)
print("Measurement counts (bitstrings):", formatted_counts)

print("\nScript finished successfully!")