
from __future__ import annotations

from typing import TypeAlias

from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

QuantumState: TypeAlias = QuantumCircuit | Statevector
