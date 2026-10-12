# QuantumSimulation/operators/fermions.py
from .base import BaseOperator
from QuantumSimulation.core.types import SchwingerParams

class TotalChargeOperator(BaseOperator):
    """
    Build charge operator, which includes the staggered background charge.

    This is the operator that commutes with the Hamiltonian and is used in Gauss' law.
    
    Q = sum_n q_n, where q_n = (sigma_z_n + (-1)^n) / 2
    """
    def __init__(self, params: SchwingerParams):
        self.params = params
        super().__init__(name="Total_Charge_Operator", num_qubits=params.L)

    def _build_terms(self) -> dict[str, list[tuple[str, list[int], float]]]:
        """Constructs the local charge operators and sums them."""
        terms = []
        for n in range(self.num_qubits):
            terms.extend(get_local_charge_terms(n))
            
        return {"total": terms}


class MinimalChargeOperator(BaseOperator):
    """
    Build minimal charge operator, discarding constant contributions.

    This is mainly used for simplifying calculations on commutation checks.
    
    Q_minimal = sum_n sigma_z_n / 2
    """
    def __init__(self, params: SchwingerParams):
        self.params = params
        super().__init__(name="Minimal_Charge_Operator", num_qubits=params.L)

    def _build_terms(self) -> dict[str, list[tuple[str, list[int], float]]]:
        """Constructs the local charge operators and sums them."""
        terms = []
        for n in range(self.num_qubits):
            terms.append(("Z", [n], 0.5))  # Only include the Z term, discard the constant identity term

        return {"total": terms}


class PairCreationOperator(BaseOperator):
    """
    Operator measuring the creation of electron-positron pairs.

    Number of electrons: N_e = sum_{n even} (1 - n_occ(n)) = sum_{n even} (1 - (1 + Z_n)/2) = sum_{n even} (0.5 - 0.5 * Z_n)

    Number of positrons: N_p = sum_{n odd} n_occ(n) = sum_{n odd} (1 + Z_n)/2 = sum_{n odd} (0.5 + 0.5 * Z_n)
    """
    def __init__(self, params: SchwingerParams):
        self.params = params
        super().__init__(name="Pair_Creation_Operator", num_qubits=params.L)

    def _build_terms(self) -> dict[str, list[tuple[str, list[int], float]]]:
        # ... logic mapped to 0.5 * I - 0.5 * Z and 0.5 * I + 0.5 * Z ...
        ne_paulis = []
        np_paulis = []

        for n in range(self.num_qubits):
            # Occupation number: n_occ = (1 + <Z>) / 2
            if n % 2 == 0:  # Electrons (Even sites)
                # Electron site with charge (n_occ - 1)
                # Electrons created are the loss of occupation, so the number of electrons created is 1 - n_occ
                ne_paulis.append(("I", [], 0.5))  # 0.5 * I
                ne_paulis.append(("Z", [n], -0.5))  # -0.5 * Z_n
            else:  # Positrons (Odd sites)
                # Positron site with charge n_occ
                # Positrons created are the increase of occupation, so the number of positrons created is n_occ
                np_paulis.append(("I", [], 0.5))  # 0.5 * I
                np_paulis.append(("Z", [n], 0.5))  # +0.5 * Z_n

        terms = {
            "electrons": ne_paulis,
            "positrons": np_paulis
        }

        return terms


class GaussOperator(BaseOperator):
    """
    Build Gauss operator G_n.
    
    G_n = E(n) - E(n-1) - q_n
    """
    def __init__(self, params: SchwingerParams):
        self.params = params
        super().__init__(name="Gauss_Operator", num_qubits=params.L)

    def _build_terms(self) -> dict[str, list[tuple[str, list[int], float]]]:
        """Constructs the local charge operators and sums them."""
        terms = {}
        for n in range(self.num_qubits):
            terms[f"site_{n}"] = get_local_gauss_operator(n)
            
        return terms


def get_local_charge_terms(i: int) -> list[tuple[str, list[int], float]]:
    """
    Build the charge operator for site i in a lattice of size L, with the given expression:

    q_i = (Z_i + (-1)^i)/2

    Parameters
    ----------
    i: int
        Site index for which to build the charge operator.

    Returns
    -------
    list
        Charge operator terms for link n.
    """
    return [
        ("Z", [i], 0.5), 
        ("I", [], 0.5 * ((-1) ** i))  # Empty list = global identity
    ]

def get_local_number_terms(i: int) -> list[tuple[str, list[int], float]]:
    """
    Build the number operator for site i in a lattice of size L, with the given expression:

    n_i = (1 + Z_i)/2

    Parameters
    ----------
    i: int
        Site index for which to build the number operator.

    Returns
    -------
    list
        Number operator terms for link n.
    """
    return [
        ("Z", [i], 0.5), 
        ("I", [], 0.5) # Empty list = global identity
    ]

def get_local_electric_field(i: int, E_0: float = 0.0):
    """
    Measure the electric field at link i as

    E(i) = E_0 + sum_{k=0..i} Q_k,

    where Q_k is the charge operator at site k.

    Parameters
    ----------
    i: int
        Link index for which to build the electric field operator (0 <= i < L-1).
    E_0: float, optional
        Background electric field at the left of the lattice (default 0).

    Returns
    -------
    list
        Electric field terms for link n.
    """

    E_n = [("I", [0], E_0)]

    for k in range(i + 1):
        E_n.extend(get_local_charge_terms(k))

    return E_n

def get_local_gauss_operator(i: int):
    """
    Build the Gauss operator at site i, which has the form:

    G_i = E(i) - E(i-1) - q_i,

    where E(i) is the electric field at link i and q_i is the charge operator at site i.
    For i=0, E(-1) is defined to be 0.

    Parameters
    ----------
    n: int
        Site index for which to build the Gauss operator (0 <= n < L).

    Returns
    -------
    list
        Gauss operator at site i.
    """
    def scale_terms(terms, factor):
        return [
            (pauli, positions, factor * coefficient)
            for pauli, positions, coefficient in terms
        ]
    E_i = get_local_electric_field(i)
    if i > 0:
        E_i_m_1 = scale_terms(get_local_electric_field(i - 1), -1.0)
    else:
        E_i_m_1 = []

    qn = scale_terms(get_local_charge_terms(i), -1.0)

    return E_i + E_i_m_1 + qn
