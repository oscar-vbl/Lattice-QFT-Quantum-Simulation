# QuantumSimulation/diagnostics/validator.py
from typing import Iterable

import numpy as np

from QuantumSimulation.core.types import SymmetryCheckConfig
from QuantumSimulation.operators import BaseOperator
from QuantumSimulation.observables import BaseObservable
from QuantumSimulation.Utils import getTimer

class PhysicsValidator:
    """
    Generic engine to validate Hamiltonian symmetries and quantum state properties.

    Attributes
    ----------
    operators: list[BaseOperator | BaseObservable]
        A list of preloaded operators and observables to be used for validation.
    """
    def __init__(self, operators_to_check: list[BaseOperator | BaseObservable]):
        self.operators_to_check = operators_to_check  # Ej: [TotalChargeOperator, MinimalChargeOperator, ElectricFieldOperator]
        self.operator_names = [op.name for op in operators_to_check]

    def enforce_symmetries(self, operator: BaseOperator, checks: list[SymmetryCheckConfig]) -> BaseOperator:
        """
        Checks [O, O'] = 0 for all configured symmetries and applies penalties if provided.
        """
        modified_operator = operator.copy()  # Create a copy to avoid modifying the original operator
        
        for check in checks:
            if check.name not in self.operator_names:
                print(f"{getTimer()} WARNING: Operator {check.name} not provided for symmetry check.")
                continue
                
            check_op = self.operators_to_check[self.operator_names.index(check.name)]

            if not isinstance(check_op, BaseOperator):
                print(f"{getTimer()} WARNING: Operator {check.name} is not a BaseOperator, and it must be for symmetry validation.")
                continue

            # Use qiskit operators
            modified_operator_qiskit = modified_operator.to_qiskit(sparse=True)
            check_op_qiskit = check_op.to_qiskit(sparse=True)
            
            # Commutator check [O, O'] = O O' - O' O
            commutator = (
                modified_operator_qiskit.compose(check_op_qiskit) -
                check_op_qiskit.compose(modified_operator_qiskit)
            ).simplify()

            if np.any(np.abs(commutator.coeffs) > check.tolerance):
                raise ValueError(f"Operator violates {check.name} symmetry ([O, {check.name}] != 0).")
            print(f"{getTimer()} INFO: Symmetry {check.name} mathematically verified.")

            # Apply Penalty if requested
            if check.penalty_weight is not None:
                # Handle "auto" weight logic here or parse it beforehand
                weight = float(check.penalty_weight) if check.penalty_weight != "auto" else np.max(np.abs(modified_operator_qiskit.coeffs)) * 10
                print(f"{getTimer()} INFO: Adding {check.name} penalty with weight {weight}")
                weighted_check_op = check_op.copy()
                weighted_check_op.update_with_weight(weight)
                weighted_check_op_terms_list = []
                for term_list in weighted_check_op.terms_dict.values():
                    weighted_check_op_terms_list.extend(term_list)
                update_dict = {f"{check.name}_penalty": weighted_check_op_terms_list}
                modified_operator.add_terms(update_dict)
                
        return modified_operator

    def validate_state(self, state, checks: list[SymmetryCheckConfig]) -> dict[str, float]:
        """
        Evaluates the operators on a given state and checks against target values.
        Returns a dictionary of the measured diagnostics.
        """
        results = {}
        for check in checks:
            if check.name not in self.operator_names:
                print(
                    f"{getTimer()} WARNING: "
                    f"Validation observable {check.name} "
                    "was not loaded."
                )
                continue
                
            check_op = self.operators_to_check[self.operator_names.index(check.name)]

            if not isinstance(check_op, BaseObservable):
                print(f"{getTimer()} WARNING: Operator {check.name} is not a BaseObservable, and it must be for state validation.")
                continue

            # For exact validation, Statevector is usually preferred
            val, error = check_op.calculate_exact(state)
            
            vals_to_check = val if isinstance(val, Iterable) else [val]

            deviations = np.abs(np.array(vals_to_check) - check.target_value)
            
            if np.any(deviations > check.tolerance):
                print(f"{getTimer()} WARNING: State violates {check.name}. Expected {check.target_value}, got {val}")
                passed = False
            else:
                passed = True

            results[check.name] = {
                "value": val,
                "error": error,
                "target": check.target_value,
                "tolerance": check.tolerance,
                "passed": passed,
            }
                
        return results

        