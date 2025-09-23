my_token = "secret"

from qiskit_ibm_runtime import QiskitRuntimeService

QiskitRuntimeService.save_account(
    token=my_token,
    channel="ibm_cloud",
    instance="crn:v1:bluemix:public:quantum-computing:us-east:a/1a51810c8dba49c999d49dfaef60b0c4:2d36d4fe-1af9-462f-a117-bb2bb84055a2::",
    overwrite=True
)

print("API key and instance saved successfully!")
