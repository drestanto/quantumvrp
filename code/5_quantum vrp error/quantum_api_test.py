my_token = "0hoS4-Q7nUHa75Q_6CgsjsxRytnnYJt5JfXGOCQo3pGD"

from qiskit_ibm_runtime import QiskitRuntimeService

QiskitRuntimeService.save_account(
    token=my_token,
    channel="ibm_cloud",
    instance="crn:v1:bluemix:public:quantum-computing:us-east:a/7cdd1aaa459f4725bfd9cf0bf5235c8a:1a968479-f9fb-44d8-930a-06e97c63335d::",
    overwrite=True
)

print("API key and instance saved successfully!")
