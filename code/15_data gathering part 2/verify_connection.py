from qiskit_ibm_runtime import QiskitRuntimeService

service = QiskitRuntimeService()
print("✅ Instance:", service.active_account()["instance"])
print("✅ Backends:")
for b in service.backends():
    print(" -", b.name)
