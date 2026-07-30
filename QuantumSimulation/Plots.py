import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from scipy.odr import ODR, Model, RealData
from .Utils import getTimer
from .ResultsAnalysis import calculate_fit_quality
from qiskit.circuit import QuantumCircuit, Parameter, QuantumRegister
import copy

def set_plot_style(
        style: str = "academic"
) -> None:
    """
    Set the style of the plots according to the specified style.
    """
    if style == "academic":
        # Academic style
        plt.rcParams.update(
            {
                "font.family": "serif",
                "mathtext.fontset": "dejavuserif",    # Makes math Latex ($$) serif
                "axes.labelsize": 12,
                "font.size": 11,
                "legend.fontsize": 10,
                "xtick.labelsize": 10,
                "ytick.labelsize": 10,
            }
        )
    else:
        print(f"{getTimer()} WARNING: Unknown plot style '{style}'. Using default style.")
        # Default style
        plt.rcParams.update(plt.rcParamsDefault)

def simplePlot(x, y, title="", xlabel="", ylabel="", savePath=None):
    """
    Basic plot function for testing
    """
    fig, ax = plt.subplots()
    ax.plot(x, y)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if savePath:
        plt.savefig(savePath)
    return fig, ax


def plotData(plot_config):
    """
    Plot data according to the provided configuration.
    """
    # Define initial plot parameters
    # plt.subplots(a,b) a (vertical) and b (horizontal) number of subplots
    num_vertical = plot_config.get("Num_Vertical_Subplots", 1)
    num_horizontal = plot_config.get("Num_Horizontal_Subplots", 1)
    total_plots = num_vertical * num_horizontal
    subplots_kwargs = plot_config.get("Subplots_Kwargs", {})
    fig, axes = plt.subplots(num_vertical, num_horizontal, **subplots_kwargs)

    for plot in plot_config.get("Plots", []):
        # Extract plot parameters
        x_loc = plot.get("X_Loc", None)
        y_loc = plot.get("Y_Loc", None)
        if total_plots > 1:
            if x_loc is not None:
                if y_loc is not None:
                    ax = axes[y_loc, x_loc]
                else:
                    ax = axes[x_loc]
            elif y_loc is not None:
                ax = axes[y_loc]
        else:
            ax = axes

        x_data = plot.get("X_Data", [])
        y_data = plot.get("Y_Data", np.arange(len(x_data)))
        title = plot.get("Title", "")
        xlabel = plot.get("X_Label", "")
        ylabel = plot.get("Y_Label", "")
        save_path = plot.get("Save_Path", None)

        ax.plot(x_data, y_data)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        legend = plot.get("Legend", {})
        if legend:
            ax.legend(**legend)
        if plot.get("Grid", True):
            ax.grid(True)
    title = plot_config.get("Title", "")
    xlabel = plot_config.get("X_Label", "")
    ylabel = plot_config.get("Y_Label", "")
    save_path = plot_config.get("Save_Path", None)
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

    return save_path


def plot_gamma_vs_qubitNum(fit_results_df, params=None):
    """
    Figure: decay_rate_qubits_num
    """
    set_plot_style()
    fig, ax = plt.subplots()
    if "Gamma_Analytical_Err" not in fit_results_df.columns: 
        fit_results_df["Gamma_Analytical_Err"] = 0
    if "Gamma_Err" not in fit_results_df.columns:
        fit_results_df["Gamma_Err"] = 0
    ax.errorbar(
        fit_results_df.index,
        fit_results_df["Gamma_Simulated"],
        yerr=fit_results_df["Gamma_Err"], # Error array
        fmt="bs",          # Blue square
        markersize=5,      # ms=5
        capsize=3,         # Horizontal bands in error bars
        elinewidth=1,      # Error bar width
        ecolor="blue",     # Error bar color
        alpha=0.7,         # A little transparency
        label="$\\Gamma$ (Simulated)",
    )
    ax.errorbar(
        fit_results_df.index,
        fit_results_df["Gamma_Analytical"],
        yerr=fit_results_df["Gamma_Analytical_Err"], # Error array
        fmt="ro",          # Red circle
        markersize=5,      # ms=5
        capsize=3,         # Horizontal bands in error bars
        elinewidth=1,      # Error bar width
        ecolor="red",     # Error bar color
        alpha=0.7,         # A little transparency
        label="$\\Gamma$ (Schwinger)",
    )
    ax.set_xlabel("Number of Qubits")
    ax.set_ylabel("$\\Gamma$")
    plt.suptitle("Vacuum Persistence $\\Gamma$ vs Number of Qubits")
    if params:
        ax.set_title(f"Parameters: {params}", fontsize=10)
    ax.legend()
    plt.grid(True)
    return fig, ax


def plot_simulated_vs_analytical(
    decay_model,
    persistence,
    t_values,
    gamma_simulated,
    model_args,
    gamma_analytical,
    gamma_err=None,
    gamma_analytical_err=None,
    params=None,
    time_offset=0,
    sigma=None,
):
    """
    Figure: best_{best_qubit_num}_decay_rate
    """
    set_plot_style()
    # Add time offset to make plot start from the end of the Zeno regime (T_Zeno_End) instead of t=0
    if time_offset is None: time_offset = 0
    t_plot = t_values + time_offset
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6))

    # Left: linear scale
    ax1.plot(t_plot, persistence, "b-", lw=2, label="Simulation")
    if gamma_err is not None:
        g_sim_label = rf"Fit: $\Gamma={gamma_simulated:.3f} \pm {gamma_err:.3f}$"
    else:
        g_sim_label = rf"Fit: $\Gamma={gamma_simulated:.3f}$"
    ax1.plot(
        t_plot,
        decay_model(t_values, gamma_simulated, *model_args),
        "r--",
        lw=1.5,
        label=g_sim_label,
    )
    if gamma_analytical_err is not None:
        g_analytical_label = rf"Schwinger: $\Gamma={gamma_analytical:.3f} \pm {gamma_analytical_err:.3f}$"
    else:
        g_analytical_label = rf"Schwinger: $\Gamma={gamma_analytical:.3f}$"
    ax1.plot(
        t_plot,
        np.exp(-gamma_analytical * t_values),
        "k:",
        lw=1.5,
        label=g_analytical_label,
    )
    ax1.set_xlabel("Time")
    ax1.set_ylabel("$G(t)$")
    ax1.set_title("Vacuum Persistence Probability $G(t)$ vs Time")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Right: logarithmic scale
    mask_pos = np.array(persistence) > 1e-4
    t_values_masked = t_plot[mask_pos]
    p_values_masked = np.array(persistence)[mask_pos]
    y_err_masked = np.array(sigma)[mask_pos] if sigma is not None else None

    # Plot with errorbar
    ax2.errorbar(
        t_values_masked,
        p_values_masked,
        yerr=y_err_masked, # Error array
        fmt="b.",          # Blue dot
        markersize=4,      # ms=4
        capsize=3,         # Horizontal bands in error bars
        elinewidth=1,      # Error bar width
        ecolor="blue",     # Error bar color
        alpha=0.7,         # A little transparency
        label="Simulation",
    )
    
    # Verifies that the decay is a pure exponential (not relaxation)
    # In log scale, the points should be a straight line
    # If there is curvature at the beginning (first 5-10 points), it is relaxation
    # If the straight line starts from t=0, it is a pure Schwinger decay

    log_p = np.log(persistence)
    x = t_values
    y = log_p
    if sigma is not None:
        y_err = np.array(sigma) / np.array(persistence) if sigma is not None else None
        # Evaluate fit on real points to calculate stats
        y_fit_points = decay_model(x, gamma_simulated, *model_args)

        reduced_chi_sq, r2 = calculate_fit_quality(y, np.log(y_fit_points), y_err)

        stats_text = (
            f"Slope: ${gamma_simulated:.4f} \\pm {gamma_err:.4f}$\n"
            f"$R^2 = {r2:.4f}$\n"
            f"$\\chi_\\nu^2 = {reduced_chi_sq:.4f}$"
        )
    else:
        r2 = np.corrcoef(x, y)[0, 1] ** 2
        stats_text = (
            f"Slope: ${gamma_simulated:.4f} \\pm {gamma_err:.4f}$\n"
            f"$R^2 = {r2:.4f}$"
        )

    print(f"{getTimer()} INFO: log-linear slope: {-gamma_simulated:.4f}")
    print(
        f"{getTimer()} INFO: R² of linear fit: {r2:.4f}"
    )
    # Force logarithmic scale because of using errorbar
    ax2.set_yscale('log')
    ax2.plot(
        t_plot,
        decay_model(x, gamma_simulated, *model_args), # Pass on np.exp because axis is in log scale
        "r--",
        lw=1.5,
        label=f"Fit"
    )
    ax2.plot(
        t_plot,
        np.exp(-gamma_analytical * t_values),
        "k:",
        lw=1.5,
        label="Schwinger Prediction",
    )
    ax2.set_xlabel("Time")
    ax2.set_ylabel("$\\log\\left(G(t)\\right)$")
    ax2.set_title("Rate validation (log scale)")
    ax2.legend(loc="lower left")
    ax2.grid(True, which="both", alpha=0.3)

    # Add box with stats
    props = dict(boxstyle="round", facecolor="white", alpha=0.8, edgecolor="lightgray")
    ax2.text(0.95, 0.95, stats_text, transform=ax2.transAxes, fontsize=10,
             verticalalignment="top", horizontalalignment="right", bbox=props)
    # R² > 0.99 → pure exponential decay (Schwinger)
    # R² < 0.95 → there is curvature, a mixture of effects


    # Add error bands
    # Done here because they move with the offset
    if time_offset is None: time_offset = 0
    if sigma is not None:
        ax1.fill_between(
            t_plot,
            persistence - sigma,
            persistence + sigma,
            color='blue', alpha=0.1, label=f'Simulation Uncertainty (±$\\sigma$)'
        )
    if gamma_err is not None:
        ax1.fill_between(
            t_plot,
            decay_model(t_values, gamma_simulated - gamma_err, *model_args),
            decay_model(t_values, gamma_simulated + gamma_err, *model_args),
            color='red', alpha=0.1, label=f'Fit Uncertainty (±$\\sigma$)'
        )

    plt.suptitle("Exponential Decay Fit and Schwinger Prediction")
    if params:
        fig.text(
            0.5, 0.945, f"Parameters: {params}", fontsize=10, ha="center", va="top"
        )
    plt.tight_layout()

    return fig, (ax1, ax2)


def plot_persistenece_vs_time_regimes(evolution_data,
                                      cut_off_times,
                                      params=None,
                                      sigma=None,
                                      ideal_evolution_data=None):
    """
    Figure: best_{best_qubit_num}_persistenece_vs_time
    """
    set_plot_style()
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(
        evolution_data.index.values,
        evolution_data["Persistence"].values,
        label="Simulated Evolution",
    )
    if ideal_evolution_data is not None:
        ax.plot(
            ideal_evolution_data.index.values,
            ideal_evolution_data["Persistence"].values,
            "r--",
            label="Ideal Evolution",
        )
    ax.axvspan(
        0, cut_off_times["T_Zeno_End"], alpha=0.2, color="gray", label="Zeno Effect"
    )
    ax.axvspan(
        cut_off_times["T_Zeno_End"],
        cut_off_times["T_Schwinger_End"],
        alpha=0.2,
        color="blue",
        label="Schwinger Regime",
    )
    if cut_off_times["T_Interference_End"] is not None:
        ax.axvspan(
            cut_off_times["T_Schwinger_End"],
            cut_off_times["T_Interference_End"],
            alpha=0.2,
            color="orange",
            label="Interference",
        )
        ax.axvspan(
            cut_off_times["T_Interference_End"],
            cut_off_times["T_Revivals_End"],
            alpha=0.2,
            color="red",
            label="Revivals",
        )
    else:
        ax.axvspan(
            cut_off_times["T_Schwinger_End"],
            cut_off_times["T_Revivals_End"],
            alpha=0.2,
            color="red",
            label="Revivals",
        )

    if sigma is not None:
        ax.fill_between(
            evolution_data.index.values,
            evolution_data["Persistence"].values - sigma,
            evolution_data["Persistence"].values + sigma,
            color='blue', alpha=0.1, label='Fit Uncertainty ($\\sigma$)'
        )

    ax.legend()
    ax.grid(True)
    ax.set_xlabel("Time")
    ax.set_ylabel("$G(t)$")
    plt.suptitle("Vacuum Persistence Probability $G(t)$ vs Time and Regimes")
    if params:
        ax.set_title(f"Parameters: {params}", fontsize=10)
    plt.tight_layout()
    return fig, ax


def plot_gamma_vs_e0(fit_results_df, params=None, title_suffix=""):
    """
    Figure: e0_quench_{qubits_num}qubits
    """
    set_plot_style()
    fig, ax = plt.subplots()
    if "Gamma_Analytical_Err" not in fit_results_df.columns: 
        fit_results_df["Gamma_Analytical_Err"] = 0
    if "Gamma_Err" not in fit_results_df.columns:
        fit_results_df["Gamma_Err"] = 0
    ax.errorbar(
        fit_results_df.index,
        fit_results_df["Gamma_Simulated"],
        yerr=fit_results_df["Gamma_Err"], # Error array
        fmt="bs",          # Blue square
        markersize=3,      # ms=5
        capsize=3,         # Horizontal bands in error bars
        elinewidth=1,      # Error bar width
        ecolor="blue",     # Error bar color
        alpha=0.7,         # A little transparency
        label="$\\Gamma$ (Simulated)",
    )
    ax.errorbar(
        fit_results_df.index,
        fit_results_df["Gamma_Analytical"],
        yerr=fit_results_df["Gamma_Analytical_Err"], # Error array
        fmt="ro",          # Red circle
        markersize=3,      # ms=5
        capsize=3,         # Horizontal bands in error bars
        elinewidth=1,      # Error bar width
        ecolor="red",     # Error bar color
        alpha=0.7,         # A little transparency
        label="$\\Gamma$ (Schwinger)",
    )
    ax.set_xlabel("$\\varepsilon_0$")
    ax.set_ylabel("$\\Gamma$")
    plt.suptitle(
        f"Vacuum Persistence $\\Gamma$ vs Background Field $\\varepsilon_0${title_suffix}"
    )
    if params:
        ax.set_title(f"Parameters: {params}", fontsize=10)
    ax.legend()
    plt.grid(True)
    return fig, ax


def ind_plot_gamma_electricField(
    ax, gamma_simulated, field_values,
    gamma_err, field_err = None,
    field_tag="\\varepsilon_0"
):
    """
    Individual subplots for function plot_gamma_vs_electricField
    """
    log_gamma_div_e0 = np.log(gamma_simulated / field_values)
    x, y = 1 / field_values, log_gamma_div_e0
    if field_err is not None:
        x_err = field_err / (field_values ** 2)
        y_err = np.sqrt(
            (gamma_err / gamma_simulated) ** 2 + \
            (field_err / field_values) ** 2
        )
    else:
        x_err = None
        #y_err = gamma_err / field_values
        y_err = gamma_err / gamma_simulated
    #ax.plot(x, y, "ro", label="$\\log(\\Gamma / E)$", markersize=5)
    ax.errorbar(
        x, y, xerr=x_err, yerr=y_err,
        fmt="ro", markersize=5, capsize=3, elinewidth=1, ecolor="red", alpha=0.7,
        label="$\\log(\\Gamma / E)$"
    )
    num = "{" + "1" + "}"
    field = "{" + field_tag + "}"
    #ax.set_xlabel(rf"$\frac{num}{field}$")
    ax.set_xlabel(rf"$1/{field_tag}$")
    gamma = "{" + "\\Gamma" + "}"
    field = "{" + field_tag + "}"
    #ax.set_ylabel(rf"$\log\left(\frac{gamma}{field}\right)$")
    ax.set_ylabel(rf"$\log\left(\Gamma/{field_tag}\right)$")

    # Fit to a line to check if it's linear in log scale
    def linear_model(x, a, b):
        return a * x + b
    
    # Preadjustment for preliminar slope 'a'
    # If field error is provided, we can use it to calculate a more accurate slope and error
    if field_err is not None:
        popt_pre, _ = curve_fit(linear_model, x, y, sigma=y_err, absolute_sigma=True)
        a_pre = popt_pre[0]

        # Effective correlated variance
        # we have x=1/E, y=log(Gamma/E)
    
        sigma_eff = np.sqrt(
            (gamma_err / gamma_simulated)**2 + 
            ((field_err / field_values) * (1.0 - np.abs(a_pre) * x))**2
        )

        # ODR Fit (Orthogonal Distance Regression) ---
        # scipy.odr needs f(beta, x)
        def odr_linear(B, x_val):
            return B[0] * x_val + B[1]

        # ODR needs initial guess. A simple polyfit is enough.
        beta0 = np.polyfit(x, y, 1)

        # RealData assumes that x_err and y_err are standard deviations (sigma)
        data = RealData(x, y, sx=x_err, sy=y_err)
        model = Model(odr_linear)

        # Run ODR
        odr_obj = ODR(data, model, beta0=beta0)
        output = odr_obj.run()

        popt         = output.beta
        a_fit, b_fit = popt
        a_err, b_err = output.sd_beta
        
        # ODR calculates Chi-Squared on 'res_var'
        reduced_chi_sq = output.res_var 
        
        # Add line in the plot
        x_fit = np.linspace(x.min(), x.max(), 100)
        y_fit = linear_model(x_fit, a_fit, b_fit)
        y_fit_points = linear_model(x, a_fit, b_fit)
        _, r2_weighted = calculate_fit_quality(y, y_fit_points, y_err)
    else:
        sigma_eff = y_err

        # Final adjustment
        popt, pcov = curve_fit(linear_model, x, y, sigma=sigma_eff, absolute_sigma=True)
        a_fit, b_fit = popt
        a_err, b_err = np.sqrt(np.diag(pcov))

        # Add line in the plot
        x_fit = np.linspace(x.min(), x.max(), 100)
        y_fit = linear_model(x_fit, a_fit, b_fit)

        # Evaluate fit on real points to calculate stats
        y_fit_points = linear_model(x, a_fit, b_fit)
        reduced_chi_sq, r2_weighted = calculate_fit_quality(y, y_fit_points, y_err)

    r_squared = np.corrcoef(x, y)[0, 1] ** 2
    ax.plot(
        x_fit,
        y_fit,
        "b--",
        #label=f"Fit: $a={a_fit:.4f}, b={b_fit:.4f}$\n$R^2={r_squared:.4f}$\n$\chi_v^2={reduced_chi_sq:.4f}$",
        label=f"Linear Fit: $y=a \\cdot x + b$",
    )
    ax.legend(loc="lower left")

    # Add text box with fit parameters and statistics
    stats_text = (
        f"$a = {a_fit:.4f} \\pm {a_err:.4f}$\n"
        f"$b = {b_fit:.4f} \\pm {b_err:.4f}$\n"
        f"$R^2 = {r2_weighted:.4f}$\n"
        f"$\\chi_\\nu^2 = {reduced_chi_sq:.4f}$"
    )
    
    # Set box up right (outside legend)
    props = dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='lightgray')
    ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='right', bbox=props)    
    ax.grid(True)
    return ax, popt


def plot_gamma_vs_electricField(gamma_simulated, e0_values, field_values,
                                gamma_err, field_err=None, params=None):
    """
    Figure: best_{best_qubit_num}_logPersistence_vs_electricField
    """
    fit_params = {}
    set_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    ax, popt = ind_plot_gamma_electricField(
        axes[0], gamma_simulated, e0_values,
        gamma_err, field_err=None,
        field_tag="\\varepsilon_0"
    )
    fit_params["e0"] = popt
    ax.set_title("Persistence Log vs Background Field $\\varepsilon_0$")
    ax, popt = ind_plot_gamma_electricField(
        axes[1], gamma_simulated, field_values,
        gamma_err, field_err=field_err,
        field_tag="\\langle E_n \\rangle"
    )
    fit_params["E_n_Mean"] = popt
    ax.set_title("Persistence Log vs Mean Electric Field $\\langle E_n \\rangle$")
    plt.suptitle("Persistence Log Fit")
    if params:
        fig.text(
            0.5, 0.945, f"Parameters: {params}", fontsize=10, ha="center", va="top"
        )

    return fig, axes, fit_params


def plot_simple_hva_circuit(reps=1):
    qc = QuantumCircuit(4)

    for rep in range(reps):
        theta_Z = Parameter(rf"$\theta_Z^{{{rep}}}$")
        theta_E = Parameter(rf"$\theta_E^{{{rep}}}$")
        theta_O = Parameter(rf"$\theta_O^{{{rep}}}$")
        # 1. Theta_Z Layer (Mass and Electric Field)
        for i in range(4):
            qc.rz(theta_Z, i)  # Mass
        for i in range(3):
            qc.rzz(theta_Z, i, i + 1)  # ZZ Electric Field

        qc.barrier()

        # 2. Theta_E Even Hopping Layer
        for i in [0, 2]:
            qc.rxx(theta_E, i, i + 1)
            qc.ryy(theta_E, i, i + 1)

        qc.barrier()

        # 3. Theta_O Odd Hopping Layer
        for i in [1]:
            qc.rxx(theta_O, i, i + 1)
            qc.ryy(theta_O, i, i + 1)

        qc.barrier()

    return qc


def plot_ansatz_circuit(base_config, L, reps, ansatz):
    """
    Figure: 00_01_Circuit_{ansatz}
    """
    from .SchwingerSimulation import SchwingerSimulation

    ansatz_config = copy.deepcopy(base_config)
    ansatz_config["Ansatz"]["Type"] = ansatz
    ansatz_config["Ansatz"]["Reps"] = reps
    ansatz_config["QubitsNumber"] = L
    ansatz_config["Hamiltonian"]["Parameters"]["L"] = L
    sim = SchwingerSimulation(copy.deepcopy(ansatz_config))
    sim.hamiltonian_prep = sim.get_hamiltonian()
    ansatz_circuit = sim.get_ansatz()
    # Add to show better qubits at the begining
    qr = QuantumRegister(L, name="q")
    clean_circuit = QuantumCircuit(qr)
    clean_circuit.compose(ansatz_circuit, inplace=True)
    fig = clean_circuit.draw(
        output="mpl",
        fold=-1,  # Delete empty wires
        scale=1.1,  # Scale the image
        initial_state=True,
        style={
            "name": "bw",  # Black and White
            "fontsize": 9,
            "fontname": "serif",  # Smaller
            "subfontsize": 7,
        },
    )
    fig.tight_layout()
    return fig


def plot_validation_ansatzes(
    qubits_nums, ansatz_comparison, dither=0.1, x_label="System Size (L)"
):
    """
    Figures: 00_02_Overlap_Vs_Reps_Errors and 00_03_Overlap_Vs_Qubits_Errors
    """
    # ==========================================
    # PLOT CONFIG
    # ==========================================
    # Academic style
    plt.rcParams.update(
        {
            "font.family": "serif",
            "axes.labelsize": 12,
            "font.size": 10,
            "legend.fontsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
        }
    )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # --- Plot 1: Infidelity ---
    ax1.set_yscale("log")
    ax1.set_xlabel(x_label)
    ax1.set_ylabel(
        r"State Infidelity $(1 - |\langle \psi_{exact} | \psi_{VQE} \rangle|^2)$"
    )
    ax1.set_title("Ansatz Expressibility (Vacuum State)")
    ax1.set_xticks(qubits_nums)
    ax1.grid(True, which="both", ls="--", alpha=0.4)

    # --- Plot 2: Relative energy error ---
    ax2.set_yscale("log")
    ax2.set_xlabel(x_label)
    ax2.set_ylabel(r"Relative Energy Error $\frac{|E_{VQE} - E_{exact}|}{|E_{exact}|}$")
    ax2.set_title("Ground State Energy Convergence")
    ax2.set_xticks(qubits_nums)
    ax2.grid(True, which="both", ls="--", alpha=0.4)

    # ==========================================
    # CALCULATIONS
    # ==========================================
    for ansatz in ansatz_comparison:
        if ansatz == list(ansatz_comparison.keys())[0]:
            dither_add = -np.array([dither] * len(qubits_nums))
            color = "#d62728"
            error_color = "#841414"
        elif ansatz == list(ansatz_comparison.keys())[-1]:
            dither_add = +np.array([dither] * len(qubits_nums))
            color = "#1f77b4"
            error_color = "#151C90"
        else:
            dither_add = 0
            color = "#157931"
            error_color = "#044626"
        comparisons = ansatz_comparison[ansatz]
        # Overlap data
        mean_overlaps = np.array([comparisons[q]["Overlap_Mean"] for q in comparisons])
        std_overlaps = np.array([comparisons[q]["Overlap_Error"] for q in comparisons])

        # Energy data (Mean and std from VQE)
        exact_energies = np.array(
            [comparisons[q]["Num_Energy_Value"] for q in comparisons]
        )  # Numpy exact
        vqe_energies = np.array(
            [comparisons[q]["Sim_Energy_Mean"] for q in comparisons]
        )  # VQE mean
        std_energies = np.array(
            [comparisons[q]["Sim_Energy_Error"] for q in comparisons]
        )  # Error estadístico/algorítmico del VQE

        # Infidelity = 1 - Overlap
        infidelities = 1.0 - mean_overlaps
        # El error de la infidelidad es exactamente igual al error del overlap
        error_infidelities = std_overlaps

        # Avoid zeros for log scale (add a small epsilon)
        infidelities = np.clip(infidelities, 1e-10, None)

        # Energy relative error: |E_vqe - E_exact| / |E_exact|
        rel_energy_error = np.abs(vqe_energies - exact_energies) / np.abs(
            exact_energies
        )

        # Propagation of error for the relative error: std_E / |E_exact|
        rel_energy_error_std = std_energies / np.abs(exact_energies)

        ax1.errorbar(
            np.array(qubits_nums) + dither_add,
            infidelities,
            yerr=error_infidelities,
            fmt="-o",
            ecolor=error_color,
            capsize=4,
            color=color,
            markersize=6,
            linewidth=1.5,
            label=f"{ansatz} VQE",
        )

        ax2.errorbar(
            np.array(qubits_nums) + dither_add,
            rel_energy_error,
            yerr=rel_energy_error_std,
            fmt="-s",
            ecolor=error_color,
            capsize=4,
            color=color,
            markersize=6,
            linewidth=1.5,
            label=f"{ansatz} VQE",
        )
        ax1.set_xticks(qubits_nums)
        # Force X-axis to show exact numbers
        ax1.set_xticklabels([str(l) for l in qubits_nums])
        ax2.set_xticks(qubits_nums)
        # Force X-axis to show exact numbers
        ax2.set_xticklabels([str(l) for l in qubits_nums])

    ax1.legend(title="Ansatz")
    ax2.legend(title="Ansatz")

    plt.suptitle("VQE Vacuum Preparation: Infidelity and Relative Energy Error")

    plt.tight_layout()

    return fig, (ax1, ax2)


def plot_duration_ansatzes(
    qubits_nums, ansatz_comparison, dither=0.1, x_label="System Size (L)"
):
    """
    Figures: 00_02_Overlap_Vs_Reps_Duration and 00_03_Overlap_Vs_Qubits_Duration
    """
    # ==========================================
    # PLOT CONFIG
    # ==========================================
    # Academic style
    plt.rcParams.update(
        {
            "font.family": "serif",
            "axes.labelsize": 12,
            "font.size": 10,
            "legend.fontsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
        }
    )

    fig, ax = plt.subplots(figsize=(10, 5))

    # --- Plot 1: Infidelity ---
    ax.set_yscale("log")
    ax.set_xlabel(x_label)
    ax.set_ylabel("Execution Time (seconds)")
    ax.set_title("VQE Convergence Time Comparison")
    ax.set_xticks(qubits_nums)
    ax.grid(True, ls="--", alpha=0.4)

    # ==========================================
    # CALCULATIONS
    # ==========================================
    for ansatz in ansatz_comparison:
        if ansatz == list(ansatz_comparison.keys())[0]:
            dither_add = -np.array([dither] * len(qubits_nums))
            color = "#d62728"
            error_color = "#841414"
        elif ansatz == list(ansatz_comparison.keys())[-1]:
            dither_add = +np.array([dither] * len(qubits_nums))
            color = "#1f77b4"
            error_color = "#151C90"
        else:
            dither_add = 0
            color = "#157931"
            error_color = "#044626"
        comparisons = ansatz_comparison[ansatz]
        # Overlap data
        mean_times = np.array([comparisons[q]["Time_Mean"] for q in comparisons])
        std_times = np.array([comparisons[q]["Time_Error"] for q in comparisons])

        ax.errorbar(
            np.array(qubits_nums) + dither_add,
            mean_times,
            yerr=std_times,
            fmt="-o",
            ecolor=error_color,
            capsize=4,
            color=color,
            markersize=6,
            linewidth=1.5,
            label=f"{ansatz} VQE",
        )

        ax.set_xticks(qubits_nums)
        # Force X-axis show exact numbers
        ax.set_xticklabels([str(l) for l in qubits_nums])

    ax.legend(title="Ansatz")

    plt.tight_layout()

    return fig, ax


def plot_computational_resources(ansatzes, L_vals, depths, cnots):
    """
    Figure: 00_04_ComputationalResources
    """
    # Plot config
    plt.rcParams.update({"font.family": "serif", "font.size": 11})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Plot CNOTs
    for ansatz in ansatzes:
        ax1.plot(L_vals, cnots[ansatz].values(), label=ansatz, linewidth=2)

    ax1.set_xlabel("System Size (L)")
    ax1.set_ylabel("Number of CNOT gates")
    ax1.set_title("Entanglement Overhead")
    ax1.grid(True, ls="--", alpha=0.4)
    ax1.legend()

    # Plot Depth
    for ansatz in ansatzes:
        ax2.plot(L_vals, depths[ansatz].values(), label=ansatz, linewidth=2)

    ax2.set_xlabel("System Size (L)")
    ax2.set_ylabel("Transpiled Circuit Depth")
    ax2.set_title("Circuit Depth Scaling")
    ax2.grid(True, ls="--", alpha=0.4)
    ax2.legend()

    plt.suptitle("Computational Resources")
    plt.tight_layout()

    return fig, (ax1, ax2)


def plot_variance_decay(ansatzes, L_vals, var_grads):
    """
    Figure: 00_05_BarrenPlateaus
    """
    fig, ax = plt.subplots(figsize=(7, 5))

    # Logarithmic Y to show exponential decay as a straight line
    for ansatz in ansatzes:
        ax.plot(L_vals, var_grads[ansatz].values(), label=ansatz, linewidth=2)

    ax.set_yscale("log")
    ax.set_xlabel("System Size (L)")
    ax.set_ylabel(
        r"Gradient Variance $\sum_i\left(\text{Var}[\partial\langle H\rangle/\partial\theta_i]\right)/N$"
    )
    ax.set_title("Barren Plateau Diagnosis")
    ax.grid(True, which="both", ls="--", alpha=0.4)
    ax.set_xticks(L_vals)
    ax.legend()

    plt.tight_layout()

    return fig, ax
