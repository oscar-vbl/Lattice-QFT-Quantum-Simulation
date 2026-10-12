from qiskit.quantum_info import SparsePauliOp
from QuantumSimulation.operators.base import BaseOperator
from QuantumSimulation.core.types import SchwingerParams

class SchwingerHamiltonianTemporalGauge(BaseOperator):
    """
    Schwinger Hamiltonian in temporal gauge.
    Inherits from BaseOperator to categorize terms (mass, hopping, gauge) 
    and provide a method `get_total_hamiltonian` to get the total Hamiltonian as a SparsePauliOp.
    """
    def __init__(self, params: SchwingerParams):
        self.params = params
        super().__init__(name="Schwinger_Hamiltonian_Temporal_Gauge_Operator", num_qubits=params.L)

    def _build_terms(self) -> dict[str, list[tuple[str, list[int], float]]]:
        L = self.params.L
        a = self.params.a
        m = self.params.m
        e0 = self.params.e0
        periodic = self.params.periodic

        if periodic:
            raise NotImplementedError(
                "Periodic boundary conditions are not supported "
                "by the gauge-eliminated Hamiltonian."
            )
        
        # 1. Hopping term
        # σ±(n) = [σ1(n) ± iσ2(n)]/2.
        # w = 1/2a
        # Hopping term: w * sum_n [σ+(n)σ-(n+1) + h.c.] = w * sum_n [σ1(n)σ1(n+1) + σ2(n)σ2(n+1)]
        # wn∑​[σn+​σn+1−​+h.c.]=w/2 ​n∑​(σxn​σxn+1​+σyn​σyn+1​)
        # 1. Homogeneous hopping
        w = 1.0 / (2.0 * a)
        hopping_terms = []
        if periodic:
            hopping_range = range(L)  # include the last term for periodic boundary
        else:
            hopping_range = range(L - 1)  # exclude the last term for
        for n in hopping_range:
            position = [n, (n + 1) % L]  # wrap around for periodic
            hopping_terms.append(("XX", position, w / 2))
            hopping_terms.append(("YY", position, w / 2))

        # 2. Mass terms
        mass_terms = []
        for n in range(L):
            mass_terms.append(("Z", [n], (-1) ** n * m / 2.0))

        # 3. Electric field (Gauss) terms
        # J = g^2 a / 2 = e^2 a / 2
        # HJ	​=Jϵ0	​n=1∑N−1	​m=1∑n	​σmz	​+4J	​n=1∑N−1	​m<k≤n∑	​σmz	​σkz

        electric_terms = []
        e = 1.0  # coupling constant (can be set to 1 for simplicity)
        J = e**2 * a / 2.0
        if periodic:
            electric_range = range(L)  # include the last term for periodic boundary
        else:
            electric_range = range(L - 1)  # exclude the last term for

        electric_positions = {}
        for n in electric_range:
            # We discard constant contributions

            # 3.1 Linear term:
            # 2 * epsilon0 * (1/2 sum_{n<=m} sigma_z_m)
            alternating_sum = sum(((-1) ** m) for m in range(n + 1))
            linear_coeff = J * (e0 + 0.5 * alternating_sum)
            if abs(linear_coeff) > 0:
                for m_idx in range(n + 1):
                    key = ("Z", (m_idx,))
                    electric_positions[key] = electric_positions.get(key, 0) + linear_coeff

            # 3.2 Quadratic term
            # (1/4) sum_{m,k<=n} sigma_z(m) sigma_z(k)
            # m = k gives identity (drop)
            # m < k gives ZZ terms with coefficient J/2
            pair_coeff = J * 0.5

            for m_idx in range(n + 1):
                for k_idx in range(m_idx + 1, n + 1):
                    key = ("ZZ", (m_idx, k_idx))
                    electric_positions[key] = electric_positions.get(key, 0) + pair_coeff

        # 4. Convert electric dict to standard list format safely
        electric_terms = [
            (label, list(position_tuple), coeff) 
            for (label, position_tuple), coeff in electric_positions.items()
        ]

        # We return the categorized dictionary matching BaseOperator signature
        return {
            "hopping": hopping_terms,
            "mass": mass_terms,
            "electric": electric_terms
        }

    def get_total_hamiltonian(self) -> SparsePauliOp:
        """
        Sums all physical components into a single matrix for VQE and Trotterization.
        """
        # self.to_qiskit() returns [HoppingOp, MassOp, ElectricOp]
        ops_list = self.to_qiskit()
        return sum(ops_list).simplify()