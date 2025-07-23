import sys

try:
    # Python 3.8+
    from importlib.metadata import version, PackageNotFoundError
except ImportError:
    # For older Python, use pkg_resources
    from pkg_resources import get_distribution, DistributionNotFound

    def version(pkg):
        try:
            return get_distribution(pkg).version
        except DistributionNotFound:
            raise PackageNotFoundError

    class PackageNotFoundError(Exception):
        pass

# List of key Qiskit-related packages to check
packages = [
    'qiskit',
    'qiskit-terra',
    'qiskit-aer',
    'qiskit-ibm-runtime',
    'qiskit-optimization',
    'qiskit-ibm-provider',
    'qiskit-machine-learning',
    'qiskit-nature',
    'qiskit-finance',
    'qiskit-aqua',
]

print("Installed Qiskit-related package versions:\n")

for pkg in packages:
    try:
        v = version(pkg)
        print(f"{pkg}: {v}")
    except PackageNotFoundError:
        print(f"{pkg}: NOT INSTALLED")

