# QuantumSimulation/operators/gauge.py
from .base import BaseOperator
from .fermions import get_local_charge_terms
from QuantumSimulation.core.types import SchwingerParams

class ElectricFieldOperator(BaseOperator):
    def __init__(self, params: SchwingerParams):
        self.params = params
        super().__init__(name="Electric_Field_Operator", num_qubits=params.L)

    def _build_terms(self) -> dict[str, list[tuple[str, list[int], float]]]:
        terms = {}
        for link in range(self.params.L - 1):
            link_terms = []
            
            if self.params.e0 != 0.0:
                link_terms.append(("I", [0], self.params.e0))
                
            # Accumulate the charge operators for all sites up to the current link
            for k in range(link + 1):
                link_terms.extend(get_local_charge_terms(k))
                
            terms[f"link_{link}"] = link_terms
            
        return terms