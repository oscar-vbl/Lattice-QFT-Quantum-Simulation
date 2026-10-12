# QuantumSimulation/algorithms/vqe.py
import time
import sys
import numpy as np
from scipy.optimize import minimize, OptimizeResult
from qiskit import QuantumCircuit
from tqdm import tqdm

from QuantumSimulation.core.types import OptimizerConfig, VQEResult
from QuantumSimulation.core.types import PrimitiveBundle
from QuantumSimulation.observables import BaseObservable
from QuantumSimulation.operators import BaseOperator
from QuantumSimulation.Utils import getTimer

try:
    from qiskit_algorithms.gradients import ParamShiftEstimatorGradient
except ImportError:
    ParamShiftEstimatorGradient = None


class VQESolver:
    """
    Variational Quantum Eigensolver execution engine.
    Handles IBM Runtime Sessions, progress tracking, and BaseObservable evaluation.
    """
    def __init__(
        self,
        ansatz: QuantumCircuit,
        minimize_observable: BaseObservable,
        hamiltonian: BaseOperator,
        config: OptimizerConfig,
        primitives: PrimitiveBundle
    ):
        self.ansatz = ansatz
        self.minimize_observable = minimize_observable
        self.hamiltonian = hamiltonian
        self.config = config
        self.primitives = primitives
        
        self.gradient_calculator = None
        self.optimization_history: list[float] = []

        if self.config.use_gradient and self.config.method in ["L-BFGS-B", "SLSQP", "CG"]:
            if self.primitives.estimator is not None and ParamShiftEstimatorGradient is not None:
                self.gradient_calculator = ParamShiftEstimatorGradient(self.primitives.estimator)
                print(f"{getTimer()} INFO: Analytic gradient (ParamShift) enabled.")
            else:
                print(f"{getTimer()} WARNING: Gradient requested but no estimator provided. Falling back to finite differences.")

    def _generate_initial_params(self) -> np.ndarray:
        """Generates initial parameters based on the configured strategy."""
        num_params = self.ansatz.num_parameters
        strategy = self.config.init_strategy
        
        if strategy == "random_small":
            return np.random.normal(0, self.config.init_max, num_params)
        elif strategy == "zeros":
            return np.zeros(num_params)
        elif strategy == "uniform_random":
            return np.random.random(num_params) * 2 * np.pi
        else:
            print(f"{getTimer()} WARNING: Unknown init_strategy '{strategy}'. Defaulting to random_small.")
            return np.random.normal(0, 0.1, num_params)

    def _energy_cost_function(self, params: np.ndarray) -> float:
        """Evaluates observable expectation value using the injected BaseObservable."""
        ansatz_circuit = self.ansatz.assign_parameters(params)
        
        if self.primitives.estimator is not None:
            pub = self.minimize_observable.get_pub(ansatz_circuit)
            result = self.primitives.estimator.run([pub]).result()
            energy = float(self.minimize_observable.process_pub_result(result[0])[0])
        else:
            energy = float(self.minimize_observable.calculate_exact(ansatz_circuit)[0])
        self._last_energy = energy
        return energy

    def _gradient_cost_function(self, params: np.ndarray) -> np.ndarray:
        """Evaluates analytical gradient using Parameter Shift Rule."""
        bound_circuit = self.ansatz.assign_parameters(params)
        job = self.gradient_calculator.run([bound_circuit], self.hamiltonian.to_qiskit(sparse=True), [params])
        return np.array(job.result().gradients[0])

    def solve(self, initial_params: np.ndarray | None = None) -> VQEResult:
        """Executes the VQE pipeline and returns a structured result object."""
        # 1. Use provided parameters or generate them according to config
        if initial_params is None:
            current_initial_params = self._generate_initial_params()
        else:
            current_initial_params = initial_params
            print(f"{getTimer()} INFO: Warm-starting VQE with provided initial parameters.")
        
        self.optimization_history = []
        
        minimize_kwargs = {
            "method": self.config.method,
            "options": {"maxiter": self.config.maxiter, "tol": self.config.tol, **self.config.options}
        }
        
        if self.gradient_calculator is not None:
            minimize_kwargs["jac"] = self._gradient_cost_function

        print(f"{getTimer()} INFO: Minimization config: method={self.config.method}, maxiter={minimize_kwargs['options']['maxiter']}")
        print(f"{getTimer()} INFO: Initial parameters strategy: {self.config.init_strategy}")

        # Callback must resolve 'pbar' from the local scope dynamically
        def callback(xk):
            energy = self._last_energy
            self.optimization_history.append(energy)
            try:
                pbar.update(1)
            except NameError:
                pass

        vqe_start = time.time()
        max_iter = minimize_kwargs["options"]["maxiter"]

        # ---------------------------------------------------------
        # Execution with IBM Runtime Session Context Management
        # ---------------------------------------------------------
        if self.primitives.backend_type == "IBM_Runtime":
            # If backend is IBM Runtime, we need to open a session
            # and use the session estimator for the optimization,
            # then restore the original estimator for the rest of the simulation.
            # This is because the RuntimeEstimator needs to be used within a session context
            # to work properly, and we want to keep the session open
            # only for the duration of the VQE optimization to avoid unnecessary costs and resource usage.
            # Session is used to avoid making a single call (and wait) for every energy measurement
            from qiskit_ibm_runtime import Session
            from qiskit_ibm_runtime import EstimatorV2 as RuntimeEstimatorV2

            print(f"{getTimer()} INFO: Opening IBM Runtime Session for VQE optimization...")
            with Session(backend=self.primitives.backend) as session:
                session_estimator = RuntimeEstimatorV2(mode=session)
                session_estimator.options.resilience_level = self.primitives.estimator.options.resilience_level
                
                original_estimator = self.primitives.estimator
                self.primitives.estimator = session_estimator

                with tqdm(total=max_iter, desc="VQE Optimization (Session)", unit="iter", file=sys.stdout) as pbar:
                    result: OptimizeResult = minimize(
                        self._energy_cost_function, current_initial_params, **minimize_kwargs, callback=callback
                    )
                    if hasattr(result, "nit"):
                        pbar.total = pbar.n = result.nit
                        pbar.refresh()

                self.primitives.estimator = original_estimator
                print(f"{getTimer()} INFO: IBM Runtime Session closed successfully.")
        
        # ---------------------------------------------------------
        # Standard Local / Aer Execution
        # ---------------------------------------------------------
        else:
            with tqdm(total=max_iter, desc="VQE Optimization", unit="iter", file=sys.stdout, leave=True, dynamic_ncols=False) as pbar:
                result: OptimizeResult = minimize(
                    self._energy_cost_function, current_initial_params, **minimize_kwargs, callback=callback
                )
                if hasattr(result, "nit"):
                    pbar.total = pbar.n = result.nit
                    pbar.refresh()

        # Diagnostics & Results
        duration = time.time() - vqe_start
        vacuum_parameters = result.x
        vacuum_energy = result.fun
        
        print(f"{getTimer()} INFO: Optimization completed in {duration:.2f}s.")
        print(f"{getTimer()} INFO:   Converged: {result.success}")
        print(f"{getTimer()} INFO:   Iterations: {result.nit if hasattr(result, 'nit') else 'N/A'}")
        print(f"{getTimer()} INFO:   Final energy: {vacuum_energy:.8f}")
        
        if len(self.optimization_history) > 1:
            improvement = self.optimization_history[0] - self.optimization_history[-1]
            print(f"{getTimer()} INFO:   Energy improvement: {improvement:.8f}")

        return VQEResult(
            state=self.ansatz.assign_parameters(vacuum_parameters),
            energy=vacuum_energy,
            parameters=vacuum_parameters,
            history=self.optimization_history,
            duration=duration,
            success=result.success,
            iterations=result.nit if hasattr(result, "nit") else None
        )