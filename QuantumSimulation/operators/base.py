# QuantumSimulation/operators/base.py
from abc import ABC, abstractmethod
import copy
from qiskit.quantum_info import SparsePauliOp

class BaseOperator(ABC):
    """
    Base class for all physical lattice operators.
    Enforces a standard format: a list of symbolic terms (Pauli, positions, coefficient).
    """
    def __init__(self, name: str, num_qubits: int):
        self.name = name
        self.num_qubits = num_qubits
        self.terms_dict: dict[str, list[tuple[str, list[int], float]]] = self._build_terms()

    @abstractmethod
    def _build_terms(self) -> dict[str, list[tuple[str, list[int], float]]]:
        """
        Must return a dictionary where keys are the physical labels (e.g., 'total' or 'link_0')
        and values are the sparse Pauli lists.

        Returns
        -------
        dict[str, list[tuple[str, list[int], float]]]
            Dictionary mapping physical labels to lists of Pauli terms,
            (e.g., {"total": [("Z", [0], 0.5), ("XX", [0, 1], 1.0)])}.
        """
        pass

    def to_qiskit(self, sparse=False) -> list[SparsePauliOp] | SparsePauliOp:
        """
        Compiles the term dictionary into a list of Qiskit SparsePauliOps.
        This list can be passed directly to a Qiskit V2 Estimator PUB.

        If `sparse` is True, returns the SparsePauliOp object as a sum of all terms.
        """
        if not self.terms_dict:
            raise ValueError("Operator terms dictionary is empty.")
            
        ops = []
        for term_list in self.terms_dict.values():
            op = SparsePauliOp.from_sparse_list(term_list, num_qubits=self.num_qubits).simplify()
            ops.append(op)

        if sparse:
            return sum(ops).simplify()
        return ops

    def get_labels(self) -> list[str]:
        """Returns the physical labels for DataFrame column naming."""
        return list(self.terms_dict.keys())

    def update_with_weight(self, weight: float) -> None:
        """Scales all coefficients in the operator by a given weight."""
        for key in self.terms_dict:
            self.terms_dict[key] = [(pauli, pos, coeff * weight) for pauli, pos, coeff in self.terms_dict[key]]

    def add_terms(self, new_terms: dict[str, list[tuple[str, list[int], float]]]) -> None:
        """Adds new terms to the operator, merging with existing terms."""
        for key, term_list in new_terms.items():
            if key in self.terms_dict:
                self.terms_dict[key].extend(term_list)
            else:
                self.terms_dict[key] = term_list

    def simplify(self, tolerance: float = 1e-12) -> None:
        """
        Groups identical Pauli strings and positions, summing their coefficients.
        Removes terms where the absolute summed coefficient is below the tolerance.
        """
        for key, term_list in self.terms_dict.items():
            # Temp dict to store grouped terms: key = (pauli, pos_tuple)
            grouped_terms = {}
            for pauli, pos, coeff in term_list:
                # Convert the position list to a tuple to use it as a dictionary key
                pos_tuple = tuple(pos) if isinstance(pos, list) else pos
                group_key = (pauli, pos_tuple)
                
                # Sum to the existing coefficient (starts at 0.0 if not present)
                grouped_terms[group_key] = grouped_terms.get(group_key, 0.0) + coeff

            # Rebuild the term list, filtering out mathematically null coefficients
            new_term_list = []
            for (pauli, pos_tuple), total_coeff in grouped_terms.items():
                if abs(total_coeff) > tolerance:
                    # Return the position to its original list format
                    pos_list = list(pos_tuple) if isinstance(pos_tuple, tuple) else pos_tuple
                    new_term_list.append((pauli, pos_list, total_coeff))
                    
            self.terms_dict[key] = new_term_list

    def apply_layout(self, layout: dict[int, int] | list[int]) -> None:
        """
        Maps logical qubit indices to physical hardware indices based on a transpiled layout.
        
        Parameters
        ----------
        layout : dict[int, int] | list[int]
            The mapping from logical to physical qubits. 
            If a list is provided, the index represents the logical qubit 
            and the value represents the physical qubit (e.g., layout[logical] = physical).
        """
        for key, term_list in self.terms_dict.items():
            new_term_list = []
            
            for pauli, pos, coeff in term_list:
                # Map single integer positions (e.g., Mass terms)
                if isinstance(pos, int):
                    new_pos = layout[pos]
                # Map lists of integer positions (e.g., Hopping or Gauge terms)
                else:
                    new_pos = [layout[p] for p in pos]
                    
                new_term_list.append((pauli, new_pos, coeff))
                
            self.terms_dict[key] = new_term_list

    def copy(self) -> "BaseOperator":
        """Creates a deep copy of the operator."""
        return copy.deepcopy(self)

    def __repr__(self) -> str:
        return f"{self.name} Operator with {self.num_qubits} qubits and terms: {self.terms_dict}"