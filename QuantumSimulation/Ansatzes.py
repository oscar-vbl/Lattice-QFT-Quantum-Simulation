from qiskit.circuit import QuantumCircuit, Parameter
from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.synthesis import SuzukiTrotter
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import XXPlusYYGate

def build_schwinger_hva(num_qubits: int,
                        reps: int,
                        hamiltonian: SparsePauliOp,
                        optimization_level: int | None = 3,
                        **kwargs):
    '''
    Build simple Hamiltonian Variational Ansatz (HVA) with original hamiltonian terms.
    '''
    z_list = []
    xy_even_list = []
    xy_odd_list = []

    # 1. Classify hamiltonian terms
    for term in hamiltonian.to_sparse_list():
        pauli_str, indices, coeff = term
        
        if "Z" in pauli_str:
            z_list.append(term)
        elif "X" in pauli_str or "Y" in pauli_str:
            # hopping terms affect two contiguous qubits
            first_idx = min(indices)
            if first_idx % 2 == 0:
                xy_even_list.append(term)
            else:
                xy_odd_list.append(term)

    # 2. Build Sparse operators for each block
    H_Z       = SparsePauliOp.from_sparse_list(z_list, num_qubits=num_qubits)
    H_XY_even = SparsePauliOp.from_sparse_list(xy_even_list, num_qubits=num_qubits) if xy_even_list else None
    H_XY_odd  = SparsePauliOp.from_sparse_list(xy_odd_list, num_qubits=num_qubits)  if xy_odd_list  else None

    # 3. Build HVA circuit
    hva_circuit = QuantumCircuit(num_qubits)
    
    # We use Trotter order one because each block commutes
    synthesis = SuzukiTrotter(order=1)

    for rep in range(reps):
        # Free parameters (3 per layer)
        theta_z    = Parameter(rf"$\theta_Z^{{({rep})}}$")
        theta_even = Parameter(rf"$\theta_E^{{({rep})}}$")
        theta_odd  = Parameter(rf"$\theta_O^{{({rep})}}$")

        # Apply e^{-i \theta H_Z}
        hva_circuit.append(PauliEvolutionGate(H_Z, time=theta_z, synthesis=synthesis), range(num_qubits))
        
        # Apply e^{-i \theta H_{XY, even}}
        if H_XY_even is not None:
            hva_circuit.append(PauliEvolutionGate(H_XY_even, time=theta_even, synthesis=synthesis), range(num_qubits))
            
        # Apply e^{-i \theta H_{XY, odd}}
        if H_XY_odd is not None:
            hva_circuit.append(PauliEvolutionGate(H_XY_odd, time=theta_odd, synthesis=synthesis), range(num_qubits))

        hva_circuit.barrier()

    if optimization_level is not None:
        hva_circuit = transpile(hva_circuit, optimization_level=optimization_level)

    return hva_circuit

def build_schwinger_hva_full(num_qubits: int,
                                        reps: int,
                                        hamiltonian: SparsePauliOp,
                                        **kwargs):
    '''
    Build simple Hamiltonian Variational Ansatz (HVA) with original hamiltonian terms.

    XY terms are added with XXPlusYYGate.
    '''
    # 1. Classify hamiltonian terms
    mass_terms  = []
    gauge_terms = []
    xy_terms    = []

    for term in hamiltonian.to_sparse_list():
        pauli_str, indices, coeff = term
        coeff = coeff.real # Ensure real part
        
        if "Z" in pauli_str:
            if len(indices) == 1:
                mass_terms.append((indices[0], coeff))
            elif len(indices) >= 2:
                # Assume Gauge ZZ interactions
                gauge_terms.append((indices[0], indices[1], coeff))
                
        elif "X" in pauli_str:
            xy_terms.append((min(indices), max(indices), coeff))

    hva_circuit = QuantumCircuit(num_qubits)

    # 2. Build per layer
    for rep in range(reps):
        
        # Parameter 1: Mass
        if mass_terms:
            theta_m = Parameter(rf"$\theta_{{M}}^{{({rep})}}$")
            for q, coeff in mass_terms:
                # exp(-i * coeff * theta * Z) -> Rz(2 * coeff * theta)
                hva_circuit.rz(2 * coeff * theta_m, q)
                
        # Parameter 2: Gauge (Electric Field)
        if gauge_terms:
            theta_g = Parameter(rf"$\theta_{{G}}^{{({rep})}}$")
            for q1, q2, coeff in gauge_terms:
                # exp(-i * coeff * theta * ZZ) -> RZZ(2 * coeff * theta)
                hva_circuit.rzz(2 * coeff * theta_g, q1, q2)

        # Parameter 3: Hopping (XX+YY jumps)
        if xy_terms:
            theta_h = Parameter(rf"$\theta_{{H}}^{{({rep})}}$")
            for q1, q2, coeff in xy_terms:
                # Qiskit convenction: XXPlusYYGate(angle) = exp(-i * (angle/4) * (XX+YY))
                #for exp(-i * coeff * theta * (XX+YY)), angle = 4 * coeff * theta
                hva_circuit.append(XXPlusYYGate(4 * coeff * theta_h), [q1, q2])

        hva_circuit.barrier()

    return hva_circuit

def build_schwinger_hva_balanced(num_qubits: int,
                                 reps: int,
                                 hamiltonian,
                                 **kwargs):
    '''
    Build simple Hamiltonian Variational Ansatz (HVA) with original hamiltonian terms.

    XY terms are added with XXPlusYYGate.

    OBC conditions are taken into account
    '''
    # 1. Classify hamiltonian terms
    mass_terms  = []
    gauge_terms = []
    xy_even     = []
    xy_odd      = []

    for term in hamiltonian.to_sparse_list():
        pauli_str, indices, coeff = term
        coeff = coeff.real 
        
        if "Z" in pauli_str:
            if len(indices) == 1:
                mass_terms.append((indices[0], coeff))
            elif len(indices) >= 2:
                # Gauge ZZ electric field
                gauge_terms.append((indices[0], indices[1], coeff))
                
        elif "X" in pauli_str:
            q1, q2 = min(indices), max(indices)
            if q1 % 2 == 0:
                xy_even.append((q1, q2, coeff))
            else:
                xy_odd.append((q1, q2, coeff))

    hva_circuit = QuantumCircuit(num_qubits)

    # 2. Build per layer
    for rep in range(reps):
        
        # Parameter 1: Mass
        if mass_terms:
            theta_m = Parameter(rf"$\theta_{{M}}^{{({rep})}}$")
            for q, coeff in mass_terms:
                hva_circuit.rz(2 * coeff * theta_m, q)
                
        # Parameter 2: Gauge ZZ
        if gauge_terms:
            theta_g = Parameter(rf"$\theta_{{E}}^{{({rep})}}$")
            for q1, q2, coeff in gauge_terms:
                hva_circuit.rzz(2 * coeff * theta_g, q1, q2)

        # Hopping parameters (Free edges + Bulk (inner)) ---
        theta_h_L = Parameter(rf"$\theta_{{HL}}^{{({rep})}}$")
        theta_h_R = Parameter(rf"$\theta_{{HR}}^{{({rep})}}$")
        theta_h_bulk = Parameter(rf"$\theta_{{HB}}^{{({rep})}}$")

        # Apply hopping dividing the qubits in inner and border
        def apply_hopping_with_edges(q1, q2, coeff):
            if q1 == 0:
                current_theta = theta_h_L       # Left edge
            elif q2 == num_qubits - 1:
                current_theta = theta_h_R       # Right edge
            else:
                current_theta = theta_h_bulk    # Inner
            
            hva_circuit.append(XXPlusYYGate(4 * coeff * current_theta), [q1, q2])

        # Apply for even
        if xy_even:
            for q1, q2, coeff in xy_even:
                apply_hopping_with_edges(q1, q2, coeff)
                
        # Apply for odd
        if xy_odd:
            for q1, q2, coeff in xy_odd:
                apply_hopping_with_edges(q1, q2, coeff)

        hva_circuit.barrier()

    return hva_circuit


