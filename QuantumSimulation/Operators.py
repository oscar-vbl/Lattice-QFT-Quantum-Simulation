from qiskit.quantum_info import SparsePauliOp


def buildSchwingerHamiltonianTemporalGauge(
    L: int, a: float, m: float, e0: float = 0, periodic: bool = False
):
    """
    Build the Schwinger Hamiltonian in temporal gauge (including Gauss' law as a constraint).

    Parameters
    ----------
    L: int
        Number of lattice sites
    a: float
        Lattice spacing
    m: float
        Mass of the fermions
    e0: float, optional
        Background electric field at the left of the lattice (default 0)

    Returns
    -------
    SparsePauliOp
        Hamiltonian as a SparsePauliOp
    """
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
    # Electric term must be OPEN boundary (Gauss law requires it)
    electric_range = range(L - 1)

    electric_positions = {}
    for n in electric_range:
        # We discard constant contributions

        # 3.1 Linear term:
        # 2 * epsilon0 * (1/2 sum_{n<=m} sigma_z_m)
        alternating_sum = sum(((-1) ** m) for m in range(n + 1))
        linear_coeff = J * (e0 + 0.5 * alternating_sum)
        if abs(linear_coeff) > 0:
            for m_idx in range(n + 1):
                key = ("Z", m_idx)
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

    # Convert electric_positions dict to list
    for (label, position), coeff in electric_positions.items():
        try:
            position_list = list(position)
        except:
            position_list = [position]
        electric_terms.append((label, position_list, coeff))

    ham_list = hopping_terms + mass_terms + electric_terms
    return SparsePauliOp.from_sparse_list(ham_list, num_qubits=L).simplify()


def buildChargeOperatorMinimal(L: int):
    """
    Charge operator without constant contributions

    Parameters
    ----------
    L: int
        Number of lattice sites

    Returns
    -------
    SparsePauliOp
        Charge operator as a SparsePauliOp
    """
    pauli_terms = []
    for n in range(L):
        pauli_terms.append(("Z", [n], 0.5))
    return SparsePauliOp.from_sparse_list(pauli_terms, num_qubits=L).simplify()


def buildChargeOperator(L: int):
    """
    Build charge operator, which includes the staggered background charge.
    This is the operator that commutes with the Hamiltonian and is used in Gauss' law.
    It has the form:

    Q = sum_n q_n, where q_n = (sigma_z_n + (-1)^n) / 2

    Parameters
    ----------
    L: int
        Number of lattice sites

    Returns
    -------
    SparsePauliOp
        Charge operator as a SparsePauliOp
    """
    pauli_terms = []
    for n in range(L):
        pauli_terms.append(("Z", [n], 0.5))
        staggered_coeff = 0.5 * ((-1) ** n)
        pauli_terms.append(("I", [], staggered_coeff))
    return SparsePauliOp.from_sparse_list(pauli_terms, num_qubits=L).simplify()


def buildPairCreationOperators(num_qubits: int):
    """
    Global observables for counting created pairs.

    Number of electrons: N_e = sum_{n even} (1 - n_occ(n)) = sum_{n even} (1 - (1 + Z_n)/2) = sum_{n even} (0.5 - 0.5 * Z_n)

    Number of positrons: N_p = sum_{n odd} n_occ(n) = sum_{n odd} (1 + Z_n)/2 = sum_{n odd} (0.5 + 0.5 * Z_n)

    Parameters
    ----------
    num_qubits: int
        Number of lattice sites

    Returns
    -------
    tuple of SparsePauliOp
        (op_electrons, op_positrons) where each is a SparsePauliOp representing the number operator for electrons and positrons respectively, including the staggered background charge.
    """
    ne_paulis = []
    np_paulis = []

    for n in range(num_qubits):
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

    # Create SparsePauliOp for electrons and positrons and simplify
    op_ne = SparsePauliOp.from_sparse_list(ne_paulis, num_qubits=num_qubits).simplify()
    op_np = SparsePauliOp.from_sparse_list(np_paulis, num_qubits=num_qubits).simplify()

    return op_ne, op_np


def buildElectricFieldOperators(L: int, e0: float = 0.0):
    """
    Build the electric field operators for all L-1 links, derived from Gauss' law. It has the form:

    E(n) = e0 + sum_{k=0..n} Q_k, n>0; E(0) = e0,

    where Q_k is the charge operator at site k.
    Returns
    -------
    list of SparsePauliOp
        List of electric field operators for each link as SparsePauliOp.
    """
    ef_ops = []

    # Base field is E_0 * I
    if e0 != 0:
        current_op = SparsePauliOp.from_list([("I" * L, e0)])
    else:
        current_op = SparsePauliOp.from_list([("I" * L, 0.0)])

    # The lattice has L sites, but only L-1 links between them
    for n in range(L - 1):
        # Q_n = chargeOperator_i(n, L)
        # Sum the charge operator of site n to the accumulated field
        current_op = (current_op + chargeOperator_i(n, L)).simplify()
        ef_ops.append(current_op)

    return ef_ops


def numberOperator_i(i: int, L: int):
    """
    Build the number operator for site i in a lattice of size L, with the given expression:

    n_i = (1 + Z_i)/2

    Parameters
    ----------
    i: int
        Site index for which to build the number operator.
    L: int
        Total number of lattice sites (qubits).
    Returns
    -------
    SparsePauliOp
        Number operator for site i as a SparsePauliOp.
    """
    return (
        SparsePauliOp.from_sparse_list([("Z", [i], 0.5)], num_qubits=L).simplify()
        + SparsePauliOp.from_sparse_list([("I", [], 0.5)], num_qubits=L).simplify()
    )


def chargeOperator_i(i: int, L: int):
    """
    Build the charge operator for site i in a lattice of size L, with the given expression:

    Q_i = (1 + Z_i)/2

    Parameters
    ----------
    i: int
        Site index for which to build the charge operator.
    L: int
        Total number of lattice sites (qubits).
    Returns
    -------
    SparsePauliOp
        Charge operator for site i as a SparsePauliOp.

    """
    paulis = [("Z", [i], 0.5), ("I", [], 0.5 * ((-1) ** i))]
    return SparsePauliOp.from_sparse_list(paulis, num_qubits=L).simplify()


def electric_field(n: int, L: int, E_0: float = 0.0):
    """
    Measure the electric field at link n as

    E(n) = E_0 + sum_{k=0..n} Q_k,

    where Q_k is the charge operator at site k.

    Parameters
    ----------
    n: int
        Link index for which to build the electric field operator (0 <= n < L-1).
    L: int
        Total number of lattice sites (qubits).
    E_0: float, optional
        Background electric field at the left of the lattice (default 0).

    Returns
    -------
    SparsePauliOp
        Electric field operator for link n as a SparsePauliOp.
    """

    E_n = E_0

    for k in range(n + 1):
        E_n += chargeOperator_i(k, L)

    return E_n


def gauss_operator(n: int, L: int):
    """
    Build the Gauss operator at site n, which has the form:

    G_n = E(n) - E(n-1) - q_n,

    where E(n) is the electric field at link n and q_n is the charge operator at site n. For n=0, E(-1) is defined to be 0.

    Parameters
    ----------
    n: int
        Site index for which to build the Gauss operator (0 <= n < L).
    L: int
        Total number of lattice sites (qubits).
    """

    E_n = electric_field(n, L)
    if n > 0:
        E_n_1 = electric_field(n - 1, L)
    else:
        E_n_1 = 0

    qn = chargeOperator_i(n, L)

    if n > 0:
        return E_n - E_n_1 - qn
    else:
        return E_n - qn
