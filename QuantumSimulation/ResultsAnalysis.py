"""
Analysis utilities for Schwinger simulation calculations.

This module provides tools to fit simulation results and validate theoretical formulas.
"""

import sys
from pathlib import Path
import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import argrelmin, savgol_filter
from qiskit.quantum_info import Statevector
from qiskit.circuit.quantumcircuit import QuantumCircuit
import pandas as pd
from typing import Callable

sys.path.append(Path(__file__).parent.as_posix())
from .Utils import getTimer, parseDictToPlot
from .observables.electric_field import ElectricFieldObservable


def check_regime(L, a, m, e0):
    """
    Print parameters of the hamiltonian to check if we are in the regime where Schwinger formula is valid and non-trivial.
    """
    w = 1 / (2 * a)
    J = a / 2
    L_phys = (L - 1) * a
    eE = e0
    M_S = 1 / np.sqrt(np.pi)  # Boson mass (e=1, m=0 approx)

    print(f"{getTimer()} INFO: Parameters of the hamiltonian to check regime:")
    print(f"    w = {w:.2f}  (hopping)")
    print(f"    J = {J:.4f}  (gauge coupling)")
    print(f"    w/J = {w / J:.1f}  (must be O(1) for non trivial physics)")
    print(f"    eE/πm² = {eE / (np.pi * m**2):.2f}  (must be >> 1)")
    print(f"    L_phys·M_S = {L_phys * M_S:.2f}  (must be >> 1)")
    print(f"    eE·a = {eE * a:.3f}  (must be << 1 for continous limit)")

def fit_persistence(
    evolution_data: pd.DataFrame,
    config: dict,
    initial_state: QuantumCircuit | None = None,
    interference_method="log_derivative",
    smooth_window=15,
    smooth_polyorder=3,
    zeno_method="inflection",
    sigma=None,
    evolution_error=None,
    use_offset=False,
    return_plot=False,
    print_info=True,
):
    """
    Fit the persistence curve to an exponential decay.

    Refactored from the original: all cutoff times are computed on the
    original (unmodified) time/persistence arrays via a smoothed auxiliary
    curve; a single mask is applied at the very end, so t_fit, p_fit and
    sigma_fit are always aligned and no intermediate array is ever mutated.

    Parameters
    ----------
    evolution_data : pd.DataFrame
        DataFrame with index = time values and column "Persistence".
    config : dict
        Simulation configuration dictionary.
    initial_state : QuantumCircuit | None
        Initial state circuit used to compute the analytical Gamma.
    interference_method : {"log_derivative", "concavity_change"}
        Algorithm used to detect the end of the pure-exponential Schwinger
        regime (before finite-size revivals contaminate the signal).
    smooth_window : int
        Savitzky-Golay window length used for cutoff detection only.
        Must be odd and > smooth_polyorder. Default 15.
    smooth_polyorder : int
        Savitzky-Golay polynomial order for cutoff detection. Default 3.
    zeno_method : {"inflection", "min_derivative"}
        How to detect the end of the Quantum Zeno plateau.
        "inflection"     — first inflection point of the smoothed curve
                           (d²P/dt² changes sign neg→pos). Preferred: marks
                           where the exponential regime begins, not where it
                           is fastest.
        "min_derivative" — point of maximum slope magnitude (original
                           behaviour). Tends to over-cut, discarding part
                           of the exponential regime.
    sigma : np.ndarray | None
        Per-point standard deviation of the persistence values (e.g. shot
        noise propagated to G(t)). If provided, passed to curve_fit with
        absolute_sigma=True so that gamma_err reflects the true measurement
        uncertainty. Must have the same length as evolution_data.
    evolution_error: pd.DataFrame, optional
        DataFrame with index = time values and electric field in its columns, with the uncertainties.
    use_offset : bool
        If True, fit A·exp(-Γt) + C instead of A·exp(-Γt).
    return_plot : bool
        If True, return (fig, axes) instead of numerical results.
    print_info : bool
        Print regime check and fit results to stdout.

    Returns
    -------
    If return_plot is False:
        (simulated_gamma, gamma_analytical, gamma_err, eE_evol, cut_off_times)
    If return_plot is True:
        (fig, axes)
    """
    # ------------------------------------------------------------------ #
    #  0. Raw arrays — never modified beyond this point                  #
    # ------------------------------------------------------------------ #
    p_orig = evolution_data["Persistence"].values.copy()
    t_orig = evolution_data.index.values.copy()

    cut_off_times = {
        "T_Zeno_End":         None,
        "T_Schwinger_End":    None,
        "T_Interference_End": None,
        "T_Revivals_End":     t_orig[-1],
    }

    # Fit models (defined once, reused below)
    def decay_model(t, gamma, A):
        return A * np.exp(-gamma * t)

    def decay_model_offset(t, gamma, A, C):
        return A * np.exp(-gamma * t) + C

    # ------------------------------------------------------------------ #
    #  1. Smoothed curve — used ONLY for cutoff detection, never for fit  #
    # ------------------------------------------------------------------ #
    p_smooth = savgol_filter(p_orig, window_length=smooth_window,
                             polyorder=smooth_polyorder)

    # First derivative of smoothed curve: shape (n-1,), times t_orig[1:]
    dp   = np.diff(p_smooth) / np.diff(t_orig)
    t_dp = t_orig[1:]

    # Second derivative: shape (n-2,), times t_dp[1:]
    d2p   = np.diff(dp) / np.diff(t_dp)
    t_d2p = t_dp[1:]

    # ------------------------------------------------------------------ #
    #  2. Cutoff 1 — end of Quantum Zeno plateau (index on t_orig)       #
    # ------------------------------------------------------------------ #
    if zeno_method == "inflection":
        # First neg→pos sign change in d²P: where the plateau ends and the
        # exponential decay begins. More conservative than min_derivative.
        sign_changes = np.where(np.diff(np.sign(d2p)) > 0)[0]
        t_zeno_end = t_d2p[sign_changes[0]] if len(sign_changes) > 0 else t_orig[0]
    else:
        # Original behaviour: point of maximum slope magnitude.
        t_zeno_end = t_dp[np.argmin(dp)]

    cut_off_times["T_Zeno_End"] = t_zeno_end

    # ------------------------------------------------------------------ #
    #  3. Cutoff 2 — end of Schwinger regime (first local minimum)       #
    # ------------------------------------------------------------------ #
    minima = argrelmin(p_smooth, order=5)[0]
    t_schwinger_end = t_orig[minima[0]] if len(minima) > 0 else t_orig[-1]
    cut_off_times["T_Schwinger_End"] = t_schwinger_end

    # ------------------------------------------------------------------ #
    #  4. Cutoff 3 — end of pure-exponential sub-regime (interference)   #
    #     Computed on the Zeno→Schwinger window of p_smooth              #
    # ------------------------------------------------------------------ #
    sw_mask = (t_orig >= t_zeno_end) & (t_orig <= t_schwinger_end)
    t_sw    = t_orig[sw_mask]
    p_sw    = p_smooth[sw_mask]

    if interference_method == "log_derivative":
        # ln(P) is linear in the pure-exponential region; when its slope
        # becomes significantly less negative, revivals are contaminating.
        valid  = p_sw > 1e-12
        t_log  = t_sw[valid]
        d_log  = np.diff(np.log(p_sw[valid])) / np.diff(t_log)
        t_dlog = t_log[1:]

        window    = max(3, len(d_log) // 10)
        baseline  = np.mean(d_log[:window])        # negative number
        threshold = baseline * 0.75                # less negative (closer to 0)

        deviations = np.where(d_log[window:] > threshold)[0]
        if len(deviations) > 0:
            cut_off_times["T_Interference_End"] = t_schwinger_end
            t_schwinger_end = t_dlog[deviations[0] + window]
            cut_off_times["T_Schwinger_End"] = t_schwinger_end

    elif interference_method == "concavity_change":
        # First negative value of d²P inside the Schwinger window signals
        # the onset of the revival-driven concavity change.
        sw_d2p_mask = (t_d2p >= t_zeno_end) & (t_d2p <= t_schwinger_end)
        d2p_sw      = d2p[sw_d2p_mask]
        t_d2p_sw    = t_d2p[sw_d2p_mask]

        neg_indices = np.where(d2p_sw < 0)[0]
        if len(neg_indices) > 0:
            cut_off_times["T_Interference_End"] = t_schwinger_end
            t_schwinger_end = t_d2p_sw[neg_indices[0]]
            cut_off_times["T_Schwinger_End"] = t_schwinger_end

    # ------------------------------------------------------------------ #
    #  5. Single final mask — applied once to all arrays                  #
    # ------------------------------------------------------------------ #
    final_mask = (t_orig >= t_zeno_end) & (t_orig <= t_schwinger_end)

    t_fit     = t_orig[final_mask]
    p_fit     = p_orig[final_mask]
    sigma_fit = sigma[final_mask] if sigma is not None else None

    # Normalise amplitude to 1 at the start of the fit window
    p0_val = p_fit[0]
    p_fit  = p_fit / p0_val
    if sigma_fit is not None:
        sigma_fit = sigma_fit / p0_val

    # Shift time so that t=0 at the start of the fit window
    t_fit = t_fit - t_fit[0]

    # ------------------------------------------------------------------ #
    #  6. Fit                                                             #
    # ------------------------------------------------------------------ #
    if not use_offset:
        fit_model   = decay_model
        p0_init     = [0.5, 0.85]
        bounds_init = ([0, 0.5], [50, 1.05])
    else:
        fit_model   = decay_model_offset
        c_guess     = float(np.min(p_fit))
        a_guess     = float(p_fit[0]) - c_guess
        p0_init     = [0.5, a_guess, c_guess]
        bounds_init = ([0.0, 0.0, 0.0], [50.0, 1.05, 1.0])

    curve_fit_kwargs = dict(p0=p0_init, bounds=bounds_init)
    if sigma_fit is not None:
        curve_fit_kwargs["sigma"]          = sigma_fit
        curve_fit_kwargs["absolute_sigma"] = True

    popt, pcov      = curve_fit(fit_model, t_fit, p_fit, **curve_fit_kwargs)
    simulated_gamma = popt[0]
    gamma_err       = float(np.sqrt(np.diag(pcov))[0])

    y_fit = fit_model(t_fit, simulated_gamma, popt[1])
    stats = calculate_fit_quality(p_fit, y_fit, sigma_fit)
    stats = calculate_fit_quality(np.log(p_fit), np.log(y_fit), sigma_fit/p_fit)


    # ------------------------------------------------------------------ #
    #  7. Analytical comparison (requires initial_state)                  #
    # ------------------------------------------------------------------ #
    gamma_analytical = None
    E_array          = None

    if initial_state is not None or evolution_error is not None:
        L   = config["Hamiltonian"]["Parameters"]["L"]
        a   = config["Hamiltonian"]["Parameters"]["a"]
        m   = config["Hamiltonian"]["Parameters"]["m"]
        e0  = config["Temporal Evolution"]["Quench"]["Parameters_to_Change"]["e0"]

        if evolution_error is not None:
            # Compute the error in the Schwinger decay rate due to uncertainties in the electric field values
            E_field_columns = [col for col in evolution_error.columns if col.startswith("E_link")]
            E_array = np.array([evolution_data.iloc[0][col] for col in E_field_columns]) # Assuming error is constant
            delta_E_array = np.array([evolution_error.iloc[0][col] for col in E_field_columns]) # Assuming error is constant
            gamma_func = lambda E: a * (abs(E) / (2 * np.pi)) * np.exp(-(np.pi * m**2) / abs(E))
            gamma_analytical = sum(gamma_func(E) for E in E_array if abs(E) > 1e-6)
            gamma_analytical_err = error_gamma_schwinger(E_array, delta_E_array, gamma_func)
        else:
            state   = Statevector.from_instruction(initial_state)
            E_observable = ElectricFieldObservable(qubits_num=L, e0=e0)
            E_array, _ = E_observable.calculate_exact(state)

            gamma_analytical = sum(
                a * (abs(E) / (2 * np.pi)) * np.exp(-(np.pi * m**2) / abs(E))
                for E in E_array if abs(E) > 1e-6
            )
            gamma_analytical_err = 0.0

        if print_info:
            check_regime(L, a, m, e0)
            print(f"{getTimer()} INFO: Γ (simulated):  {simulated_gamma:.4f} ± {gamma_err:.4f}")
            print(f"{getTimer()} INFO: Γ (analytical): {gamma_analytical:.4f}")
            deviation = abs(simulated_gamma - gamma_analytical) / gamma_analytical * 100
            print(f"{getTimer()} INFO: Deviation: {deviation:.1f}%")
            # Log-linear slope and R² as additional validation diagnostics
            log_slope = np.polyfit(t_fit, np.log(np.clip(p_fit, 1e-12, None)), 1)[0]
            ss_res    = np.sum((p_fit - fit_model(t_fit, *popt)) ** 2)
            ss_tot    = np.sum((p_fit - np.mean(p_fit)) ** 2)
            r2        = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
            print(f"{getTimer()} INFO: log-linear slope: {log_slope:.4f}")
            print(f"{getTimer()} INFO: R² of linear fit: {r2:.4f}")
    else:
        if print_info:
            print(f"{getTimer()} INFO: No initial state provided, skipping analytical comparison.")

    # ------------------------------------------------------------------ #
    #  8. Return                                                         #
    # ------------------------------------------------------------------ #
    if return_plot:
        plot_params = parseDictToPlot(
            {
                **config["Hamiltonian"]["Parameters"],
                **config["Temporal Evolution"]["Quench"]["Parameters_to_Change"],
            },
            remove_keys=[],
            rename_keys={"e0": "$\\varepsilon_0$"},
        )
        from .Plots import plot_simulated_vs_analytical
        fig, axes = plot_simulated_vs_analytical(
            decay_model,
            p_fit,
            t_fit,
            simulated_gamma,
            popt[1:],
            gamma_analytical,
            gamma_err,
            gamma_analytical_err=gamma_analytical_err,
            params=plot_params,
            time_offset=cut_off_times["T_Zeno_End"],
            sigma=sigma_fit,
        )
        return fig, axes
    else:
        return simulated_gamma, gamma_analytical, gamma_err, gamma_analytical_err, E_array, cut_off_times

def error_gamma_schwinger(
    E_array: np.ndarray,
    delta_E_array: np.ndarray,
    gamma_func: Callable,
    h:float=1e-5
):
    """
    Calculate the error in the Schwinger decay rate (gamma) due to uncertainties in the electric field values.

    Parameters
    ----------
    E_array : np.ndarray
        Array of electric field values.
    delta_E_array : np.ndarray
        Array of uncertainties in the electric field values.
    gamma_func : Callable
        Function to calculate the decay rate.
    h : float, optional
        Step size for finite difference, by default 1e-5.

    Returns
    -------
    float
        Error in the Schwinger decay rate.
    """
    E_array = np.array(E_array, dtype=float)
    delta_E_array = np.array(delta_E_array, dtype=float)
    
    # Vector para guardar las derivadas parciales
    gradiente = np.zeros_like(E_array)
    
    # Calcular la derivada parcial para cada componente i
    for i in range(len(E_array)):
        # Hacemos copias para no pisar el array original
        E_plus = E_array.copy()
        E_minus = E_array.copy()
        
        # Perturbamos SOLO la componente i
        E_plus[i] += h
        E_minus[i] -= h

        gamma_plus = gamma_func(E_plus[i])
        gamma_minus = gamma_func(E_minus[i])

        # Diferencia finita central
        gradiente[i] = (gamma_plus - gamma_minus) / (2 * h)
        
    # Propagación de errores en cuadratura
    error_gamma = np.sqrt(np.sum((gradiente * delta_E_array)**2))
    
    return error_gamma


def evaluate_energy_statevector(circuit, parameters, param_values, hamiltonian_matrix):
    """
    Evaluate exact energy using Statevector
    """
    bound_circuit = circuit.assign_parameters(dict(zip(parameters, param_values)))
    state = Statevector(bound_circuit).data
    # <psi | H | psi>
    energy = np.real(np.vdot(state, hamiltonian_matrix.dot(state)))
    return energy


def calculate_global_gradient_variance(
    ansatz_circuit, hamiltonian_sparse, num_samples=100
):
    """
    Calculate mean variance of the gradient of all parameters inside a circuit.
    """
    params = ansatz_circuit.parameters
    num_params = len(params)
    if num_params == 0:
        return 0.0

    # Matrix to save all gradients: (num_samples, num_params)
    all_gradients = np.zeros((num_samples, num_params))

    for s in range(num_samples):
        # 1. Random point in the energy landscape
        theta_random = np.random.uniform(0, 2 * np.pi, num_params)

        # 2. Calculate gradient for each parameter
        for i in range(num_params):
            # Forward shift (+ pi/2)
            theta_plus = theta_random.copy()
            theta_plus[i] += np.pi / 2
            e_plus = evaluate_energy_statevector(
                ansatz_circuit, params, theta_plus, hamiltonian_sparse
            )

            # Backward shift (+ pi/2)
            theta_minus = theta_random.copy()
            theta_minus[i] -= np.pi / 2
            e_minus = evaluate_energy_statevector(
                ansatz_circuit, params, theta_minus, hamiltonian_sparse
            )

            # Gradient dH/d\theta_i
            all_gradients[s, i] = 0.5 * (e_plus - e_minus)

    # 3. Calculate variance of each parameter
    variances = np.var(all_gradients, axis=0)

    # 4. Mean of all variances
    return np.mean(variances)


def calculate_gradient_variance(
    ansatz_circuit, hamiltonian_sparse, num_samples=100, target_param_idx=0
):
    """
    Calculate gradient variance respect to the given target_param_idx parameter
    """
    params = ansatz_circuit.parameters
    if len(params) == 0:
        return 0.0

    gradients = []

    for _ in range(num_samples):
        # 1. Random point in the energy landscape
        theta_random = np.random.uniform(0, 2 * np.pi, len(params))

        # 2. Forward shift (+ pi/2)
        theta_plus = theta_random.copy()
        theta_plus[target_param_idx] += np.pi / 2
        e_plus = evaluate_energy_statevector(
            ansatz_circuit, params, theta_plus, hamiltonian_sparse
        )

        # 3. Backward shift (- pi/2)
        theta_minus = theta_random.copy()
        theta_minus[target_param_idx] -= np.pi / 2
        e_minus = evaluate_energy_statevector(
            ansatz_circuit, params, theta_minus, hamiltonian_sparse
        )

        # 4. Gradient calculation
        grad = 0.5 * (e_plus - e_minus)
        gradients.append(grad)

    return np.var(gradients)


def calculate_fit_quality(
        y: np.ndarray,
        y_fit: np.ndarray,
        y_err: np.ndarray,
        num_params: int=2,
        fit: str = "poly"
    ):
    """
    Calculate the reduced chi-squared and weighted R² for a linear fit of y vs x with uncertainties y_err.

    Parameters
    ----------
    y : np.ndarray
        Dependent variable data points.
    y_fit : np.ndarray
        Fitted values of the dependent variable.
    y_err : np.ndarray
        Uncertainties in the dependent variable data points.
    num_params : int
        Number of parameters in the fit model.
        
    Returns
    -------
    reduced_chi_sq : float
        The reduced chi-squared value of the fit.
    r2_weighted : float
        The weighted R² value of the fit.
    """
    # Reduced Chi-Squared
    chi_squared = np.sum(((y - y_fit) / y_err) ** 2)
    degrees_of_freedom = len(y) - num_params
    reduced_chi_sq = chi_squared / degrees_of_freedom

    # Weighted R² (Standard statistical weights are 1/Variance)
    w = 1.0 / (y_err ** 2)
    y_mean_w = np.sum(w * y) / np.sum(w)
    ss_tot_w = np.sum(w * (y - y_mean_w) ** 2)
    ss_res_w = np.sum(w * (y - y_fit) ** 2)
    
    r2_weighted = 1.0 - (ss_res_w / ss_tot_w) if ss_tot_w != 0 else 0.0

    return reduced_chi_sq, r2_weighted