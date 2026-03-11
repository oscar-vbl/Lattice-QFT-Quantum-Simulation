from qiskit.quantum_info import SparsePauliOp

def buildSchwingerHamiltonianTemporalGauge_(L, a, m, e0=0, periodic=True):
    '''
    Build the Schwinger Hamiltonian in temporal gauge (including Gauss' law).
    
    Params:
    - L: int, number of lattice sites
    - a: float, lattice spacing
    - m: float, mass of the fermions
    - e0: float, background electric field (default 0)
    Returns:
    - Hamiltonian as a SparsePauliOp
    '''
    # 1. Hopping term
    # σ±(n) = [σ1(n) ± iσ2(n)]/2.
    # w = 1/2a
    # Hopping term: w * sum_n [σ+(n)σ-(n+1) + h.c.] = w * sum_n [σ1(n)σ1(n+1) + σ2(n)σ2(n+1)]
    # wn∑​[σn+​σn+1−​+h.c.]=w/2 ​n∑​(σxn​σxn+1​+σyn​σyn+1​)
    # 1. Hopping HOMOGENEO (Sin factores raros en los bordes)
    w = 1.0 / (2.0 * a)
    hopping_terms = []
    if periodic: hopping_range = range(L)  # include the last term for periodic boundary
    else:        hopping_range = range(L - 1)     # exclude the last term for
    for n in hopping_range:
        position = [n, (n + 1) % L]  # wrap around for periodic
        hopping_terms.append(("XX", position, w / 2))
        hopping_terms.append(("YY", position, w / 2))
    
    # 2. Mass terms
    mass_terms = []
    for n in range(L):
        mass_terms.append(("Z", [n], (-1)**n * m  / 2.0))
    
    # 3. Electric field (Gauss) terms
    # J = g^2 a / 2 = e^2 a / 2
    # HJ	​=Jϵ0	​n=1∑N−1	​m=1∑n	​σmz	​+4J	​n=1∑N−1	​m<k≤n∑	​σmz	​σkz

    electric_terms = []
    e = 1.0  # coupling constant (can be set to 1 for simplicity)
    J = e**2 * a / 2.0
    if periodic: electric_range = range(L)  # include the last term for periodic boundary
    else:        electric_range = range(L - 1)     # exclude the last term for
    # Electric term must be OPEN boundary (Gauss law requires it)
    electric_range = range(L - 1)

    electric_positions = {}
    for n in electric_range:
        # compute Q_n = e0 + 1/2 * sum_{m=0..n} (-1)^m
        alternating_sum = sum(((-1) ** m) for m in range(n + 1))
        Q_n = e0 + 0.5 * alternating_sum

        # constant contribution
        const_coeff = J * (Q_n * Q_n + 0.25 * (n + 1))
        if abs(const_coeff) > 0:
            key = ("I", ())
            electric_positions[key] = electric_positions.get(key, 0) + const_coeff
            #electric_terms.append(("I", [], const_coeff))

        # 3.1 Linear term:
        # 2 * epsilon0 * (1/2 sum_{n<=m} sigma_z_m)
        linear_coeff = J * e0
        if abs(linear_coeff) > 0:
            for m_idx in range(n + 1):
                key = ("Z", m_idx)
                electric_positions[key] = electric_positions.get(key, 0) + linear_coeff


        # single-site Z terms for m=0..n with coefficient J * Q_n
        single_coeff = J * Q_n
        if abs(single_coeff) > 0:
            for m_idx in range(n + 1):
                key = ("Z", m_idx)
                electric_positions[key] = electric_positions.get(key, 0) + single_coeff
                #electric_terms.append(("Z", [m_idx], single_coeff))

        # two-site ZZ long-range terms for m<k with coefficient J * 1/2
        pair_coeff = J * 0.5
        if abs(pair_coeff) > 0:
            for m_idx in range(n + 1):
                for k_idx in range(m_idx + 1, n + 1):
                    key = ("ZZ", (m_idx, k_idx))
                    electric_positions[key] = electric_positions.get(key, 0) + pair_coeff
                    #electric_terms.append(("ZZ", [m_idx, k_idx], pair_coeff))

    # Convert electric_positions dict to list
    for (label, position), coeff in electric_positions.items():
        try:    position_list = list(position)
        except: position_list = [position]
        electric_terms.append((label, position_list, coeff))

    ham_list = hopping_terms + mass_terms + electric_terms
    return SparsePauliOp.from_sparse_list(ham_list, num_qubits=L).simplify()    

def buildSchwingerHamiltonianTemporalGauge(L, a, m, e0=0, periodic=True):
    '''
    Build the Schwinger Hamiltonian in temporal gauge (including Gauss' law).
    
    Params:
    - L: int, number of lattice sites
    - a: float, lattice spacing
    - m: float, mass of the fermions
    - e0: float, background electric field (default 0)
    Returns:
    - Hamiltonian as a SparsePauliOp
    '''
    # 1. Hopping term
    # σ±(n) = [σ1(n) ± iσ2(n)]/2.
    # w = 1/2a
    # Hopping term: w * sum_n [σ+(n)σ-(n+1) + h.c.] = w * sum_n [σ1(n)σ1(n+1) + σ2(n)σ2(n+1)]
    # wn∑​[σn+​σn+1−​+h.c.]=w/2 ​n∑​(σxn​σxn+1​+σyn​σyn+1​)
    # 1. Hopping HOMOGENEO (Sin factores raros en los bordes)
    w = 1.0 / (2.0 * a)
    hopping_terms = []
    if periodic: hopping_range = range(L)  # include the last term for periodic boundary
    else:        hopping_range = range(L - 1)     # exclude the last term for
    for n in hopping_range:
        position = [n, (n + 1) % L]  # wrap around for periodic
        hopping_terms.append(("XX", position, w / 2))
        hopping_terms.append(("YY", position, w / 2))
    
    # 2. Mass terms
    mass_terms = []
    for n in range(L):
        mass_terms.append(("Z", [n], (-1)**n * m  / 2.0))
    
    # 3. Electric field (Gauss) terms
    # J = g^2 a / 2 = e^2 a / 2
    # HJ	​=Jϵ0	​n=1∑N−1	​m=1∑n	​σmz	​+4J	​n=1∑N−1	​m<k≤n∑	​σmz	​σkz

    electric_terms = []
    e = 1.0  # coupling constant (can be set to 1 for simplicity)
    J = e**2 * a / 2.0
    if periodic: electric_range = range(L)  # include the last term for periodic boundary
    else:        electric_range = range(L - 1)     # exclude the last term for
    # Electric term must be OPEN boundary (Gauss law requires it)
    electric_range = range(L - 1)

    electric_positions = {}
    for n in electric_range:
        # We discard constant contributions

        # 3.1 Linear term:
        # 2 * epsilon0 * (1/2 sum_{n<=m} sigma_z_m)
        linear_coeff = J * e0
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
        try:    position_list = list(position)
        except: position_list = [position]
        electric_terms.append((label, position_list, coeff))

    ham_list = hopping_terms + mass_terms + electric_terms
    return SparsePauliOp.from_sparse_list(ham_list, num_qubits=L).simplify()    

def buildChargeOperatorMinimal(L):
    pauli_terms = []
    for n in range(L):
        pauli_terms.append(("Z", [n], 0.5))
    return SparsePauliOp.from_sparse_list(pauli_terms, num_qubits=L)

def numberOperator_i(i, L):
    '''
    Build the number operator n_i = (1 + Z_i)/2 for site i in a lattice of size L.
    '''
    return SparsePauliOp.from_sparse_list([("Z", [i], 0.5)], num_qubits=L).simplify() + SparsePauliOp.from_sparse_list([("I", [], 0.5)], num_qubits=L).simplify()

def chargeOperator_i(i, L):
    '''
    Build the charge operator Q_i = (1 + Z_i)/2 - (1 + (-1)^i)/2 for site i in a lattice of size L.
    '''
    coeff = 0.5 * ((-1) ** i)
    number_part = numberOperator_i(i, L)
    staggered_part = coeff * SparsePauliOp("I"*L)
    return number_part - staggered_part

def electric_field(n, L, E_0=0):

    E_n = E_0

    for k in range(n+1):
        E_n += chargeOperator_i(k, L)

    return E_n

def gauss_operator(n, L):

    E_n = electric_field(n, L)
    if n > 0: E_n_1 = electric_field(n-1, L)
    else:     E_n_1 = 0

    qn = chargeOperator_i(n, L)

    if n > 0: return E_n - E_n_1 - qn
    else:     return E_n - qn

if __name__ == "__main__":
    import numpy as np
    L, a, m, e0 = 4, 0.5, 1.0, 0.0
    H = buildSchwingerHamiltonianTemporalGauge(L, a, m, e0, periodic=True)
    L, a, m, e0 = 6, 0.5, 1.0, 0.0
    H = buildSchwingerHamiltonianTemporalGauge(L, a, m, e0, periodic=True)

    # 1) Hermiticity
    Hm = H.to_matrix()
    print("Hermitian? |H-H^†| =", np.linalg.norm(Hm - Hm.conj().T))

    # 2) Build total Q_op (use same convention as Hamiltonian's Q_n)
    Q_op = buildChargeOperatorMinimal(L)
    Qm = Q_op.to_matrix()

    # 3) Commutator [H,Q]
    comm = Hm @ Qm - Qm @ Hm
    print("||[H,Q]|| =", np.linalg.norm(comm))

    # 4) diagonalization sanity (small L)
    eigvals, eigvecs = np.linalg.eigh(Hm)
    print("lowest eigenvalues:", eigvals[:6])