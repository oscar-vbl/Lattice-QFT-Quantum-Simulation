"""
Analysis utilities for Schwinger simulation calculations.

This module provides tools to fit simulation results and validate theoretical formulas.
"""

import sys
from pathlib import Path
sys.path.append(Path(__file__).parent.as_posix())
from Plots import plot_simulated_vs_analytical
from Utils import getTimer, parseDictToPlot
import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import argrelmin
from Operators import measure_electric_field
from qiskit.quantum_info import Statevector

def check_regime(L, a, m, e0):
    '''
    Print parameters of the hamiltonian to check if we are in the regime where Schwinger formula is valid and non-trivial.
    '''
    w = 1 / (2*a)
    J = a / 2
    L_phys = (L-1) * a
    eE = e0
    M_S = 1/np.sqrt(np.pi)  # Boson mass (e=1, m=0 approx)
    
    print(f"{getTimer()} INFO: Parameters of the hamiltonian to check regime:")
    print(f"    w = {w:.2f}  (hopping)")
    print(f"    J = {J:.4f}  (gauge coupling)")
    print(f"    w/J = {w/J:.1f}  (must be O(1) for non trivial physics)")
    print(f"    eE/πm² = {eE/(np.pi*m**2):.2f}  (must be >> 1)")
    print(f"    L_phys·M_S = {L_phys*M_S:.2f}  (must be >> 1)")
    print(f"    eE·a = {eE*a:.3f}  (must be << 1 for continous limit)")


def fit_persistence(evolution_data, config,
                    initial_state=None,
                    use_offset=False,
                    return_plot=False,
                    print_info=True):
    '''
    Fit the persistence curve to an exponential decay, with a possible offset.
    The offset can be used to account for the non-zero value of the persistence at the end of the simulation, which is due to the finite size of the system and the revivals.
    The function also applies cut-offs to discard the initial Zeno region and the final revivals, and focuses on the Schwinger region where the decay is expected to be exponential.
    The cut-offs are determined by analyzing the derivative of the persistence curve.
    
    Per default, if return_plot=False, the function returns the fitted parameters.
    If return_plot=True, the function returns the plot object, showing the fit.
    '''
    cut_off_times = {
        "T_Zeno_End":         None,
        "T_Schwinger_End":    None,
        "T_Interference_End": None,
        "T_Revivals_End":     None,
    }
    # Robust exponential fit
    def decay_model(t, gamma, A):
        return A * np.exp(-gamma * t)
    def decay_model_offset(t, gamma, A, C):
        return A * np.exp(-gamma * t) + C

    persistence_orig = evolution_data["Persistence"].values
    t_values_orig    = evolution_data.index.values

    cut_off_times["T_Revivals_End"] = t_values_orig[-1]

    # 1. Take only first monotonic decay (until first local minima)
    minima = argrelmin(np.array(persistence_orig), order=3)[0]
    if len(minima) > 0:
        t_fit_end = t_values_orig[minima[0]]
    else:
        t_fit_end = t_values_orig[-1]  # fallback if no minima found
    mask = t_values_orig <= t_fit_end
    
    t_values    = t_values_orig[mask]
    persistence = persistence_orig[mask]
    cut_off_times["T_Schwinger_End"] = t_fit_end

    # 2. Find derivative change point (to discard initial oscillation)
    t_step = t_values[1] - t_values[0]
    persistence_deriv = np.array([(persistence[i+1] - persistence[i-1])/t_step for i in range(1, len(persistence)-1)])
    persistence_deriv = np.diff(persistence) / np.diff(t_values)
    if persistence[1] - persistence[0] < 0:
        # Discard initial oscillation
        min_pers_der = min(persistence_deriv)
        cut_off_zeno_t = t_values[1:-1][list(persistence_deriv).index(min_pers_der)]
    elif persistence[1] - persistence[0] > 0:
        # Discard initial oscillation
        min_pers_der = min(persistence_deriv)
        cut_off_zeno_t = t_values[1:-1][list(persistence_deriv).index(min_pers_der)]
    
    # Apply cut-off to discard relaxation
    mask2       = t_values >= cut_off_zeno_t
    t_values    = t_values[mask2]
    persistence = persistence[mask2]
    persistence_deriv = persistence_deriv[mask2[1:]]
    cut_off_times["T_Zeno_End"] = cut_off_zeno_t

    # 3. Early stop for step: 
    # Look if curve gets plain or changes concavity
    # 1st der close to 0 or positive → not exponential anymore, stop before
    # 2nd der changes sign → inflection point, stop before
    sec_persistence_deriv = np.diff(persistence_deriv) / np.diff(t_values)
    
    # If sec derivative changes sign, it indicates an inflection point.
    sec_der_negs = np.where(sec_persistence_deriv < 0)[0]
    if len(sec_der_negs) > 0:
        concavity_change_index = sec_der_negs[0] 
        t_concavity_change = t_values[concavity_change_index]
        # Concavity change found, we must redefine Schwinger cutoff
        # There is a region of interference between Schwinger and revivals
        cut_off_times["T_Interference_End"] = cut_off_times["T_Schwinger_End"]
        cut_off_times["T_Schwinger_End"]    = t_concavity_change
    else:
        concavity_change_index = len(persistence) - 1
        t_concavity_change = t_values[concavity_change_index]
        # No concavity change found
        # Schwinger cutoff is at the local minima defined before
        # Interference cutoff is null
    
    # Apply cut-off to discard non-exponential tail
    mask3       = t_values <= t_concavity_change
    t_values    = t_values[mask3]
    persistence = persistence[mask3]
    
    if not use_offset:
        # Normalize to 1 at cut-off point
        persistence = persistence / persistence[0]
        fit_model = decay_model
        p0_init     = [0.5, 0.85]     # Found by inspection
        bounds_init = ([0, 0.5], [50, 1.05])
    else:
        fit_model = decay_model_offset
        # Dynamic estimation (Data-driven p0)
        # 1. C (offset) is, at most, the minimum of persistence
        c_guess = min(persistence) 
        
        # 2. P(0) = A + C
        # So, A = P(0) - C
        a_guess = persistence[0] - c_guess
        
        # 3. Gamma_guess: generic
        gamma_guess = 0.5
        
        # [gamma, A, C]
        p0_init     = [gamma_guess, a_guess, c_guess] 
        
        # Limits: ([gamma_min, A_min, C_min], [gamma_max, A_max, C_max])
        bounds_init = (
            [0.0, 0.0, 0.0],       # Neither decaiment, amplitude nor offset are negative
            [50.0, 1.05, 1.0]      # Upper limit for gamma, A and C (cannot exceed 1)
        )
    # Redefine t=0 where fit starts
    t_values_rel = t_values - t_values[0]
    t_values = t_values_rel

    # Fit to decay_model
    popt, pcov = curve_fit(
        fit_model,
        t_values,           
        persistence,        
        p0=p0_init,     # Found by inspection
        bounds=bounds_init
    )
    simulated_gamma = popt[0]
    gamma_err = np.sqrt(np.diag(pcov))[0]

    if initial_state is not None:
        L = config["Hamiltonian"]["Parameters"]["L"]
        a = config["Hamiltonian"]["Parameters"]["a"]
        m = config["Hamiltonian"]["Parameters"]["m"]
        e0 = config["Temporal Evolution"]["Quench"]["Parameters_to_Change"]["e0"] # Campo eléctrico efectivo (ajustado por la configuración)
        state = Statevector.from_instruction(initial_state)
        # Physical electric field E_n
        eE_evol = measure_electric_field(state, L, e0) 

        # Analytical value of gamma from config
        gamma_analytical = sum([a * (abs(eE_evol_) / (2 * np.pi)) * 
                            np.exp(-(np.pi * m**2) / abs(eE_evol_)) 
                            for eE_evol_ in eE_evol if abs(eE_evol_) > 1e-6])  # Evitar división por cero
        
        if print_info:
            # Check regime for validity of Schwinger formula
            check_regime(L, a, m, e0)
            print(f"{getTimer()} INFO: Γ (simulated):  {simulated_gamma:.4f} ± {gamma_err:.4f}")
            print(f"{getTimer()} INFO: Γ (analytical): {gamma_analytical:.4f}")
            print(f"{getTimer()} INFO: Deviation: {abs(simulated_gamma - gamma_analytical) / gamma_analytical * 100:.1f}%")
    else:
        print(f"{getTimer()} INFO: No initial state provided, skipping analytical comparison.")
        gamma_analytical, gamma_err, eE_evol = None, None, None

    if return_plot:
        plot_params = parseDictToPlot(
            {**config["Hamiltonian"]["Parameters"], **config["Temporal Evolution"]["Quench"]["Parameters_to_Change"]},
            remove_keys=[],
            rename_keys={"e0": "$\\varepsilon_0$"})
        fig, axes = plot_simulated_vs_analytical(decay_model, persistence, t_values, simulated_gamma, popt[1], gamma_analytical, params=plot_params)
        return fig, axes
    else:
        return simulated_gamma, gamma_analytical, gamma_err, eE_evol, cut_off_times
