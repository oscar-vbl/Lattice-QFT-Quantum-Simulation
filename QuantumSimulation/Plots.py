import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from Utils import getTimer

def simplePlot(x, y, title="", xlabel="", ylabel="", savePath=None):
    '''
    Basic plot function for testing
    '''
    fig, ax = plt.subplots()
    ax.plot(x, y)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if savePath:
        plt.savefig(savePath)
    return fig, ax

def plotData(plot_config):
    '''
    Plot data according to the provided configuration.
    '''
    # Define initial plot parameters
    # plt.subplots(a,b) a (vertical) and b (horizontal) number of subplots
    num_vertical    = plot_config.get("Num_Vertical_Subplots", 1)
    num_horizontal  = plot_config.get("Num_Horizontal_Subplots", 1)
    total_plots     = num_vertical * num_horizontal
    subplots_kwargs = plot_config.get("Subplots_Kwargs", {})
    fig, axes = plt.subplots(num_vertical, num_horizontal, **subplots_kwargs)

    for plot in plot_config.get("Plots", []):
        # Extract plot parameters
        x_loc  = plot.get("X_Loc", None)
        y_loc  = plot.get("Y_Loc", None)
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

        x_data    = plot.get("X_Data",    [] )
        y_data    = plot.get("Y_Data",    np.arange(len(x_data)) )
        title     = plot.get("Title",     "")
        xlabel    = plot.get("X_Label",   "")
        ylabel    = plot.get("Y_Label",   "")
        save_path = plot.get("Save_Path", None)
        
        ax.plot(x_data, y_data)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        legend = plot.get("Legend", {})
        if legend:                 ax.legend(**legend)
        if plot.get("Grid", True): ax.grid(True)
    title  = plot_config.get("Title", "")
    xlabel = plot_config.get("X_Label", "")
    ylabel = plot_config.get("Y_Label", "")
    save_path = plot_config.get("Save_Path", None)
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

    return save_path

def plot_gamma_vs_qubitNum(fit_results_df, params=None):
    '''
    Figure: decay_rate_qubits_num
    '''
    fig, ax = plt.subplots()
    ax.plot(fit_results_df.index, fit_results_df["Gamma_Analytical"], "ro", label="$\\Gamma$ (Analytical)", markersize=5)
    ax.plot(fit_results_df.index, fit_results_df["Gamma_Simulated"], "bs",  label="$\\Gamma$ (Simulated)",  markersize=5)
    ax.set_xlabel("Number of Qubits")
    ax.set_ylabel("$\\Gamma$")
    plt.suptitle("Vacuum Persistence $\\Gamma$ vs Number of Qubits")
    if params:
        ax.set_title(f"Parameters: {params}", fontsize=10)
    ax.legend()
    plt.grid(True)
    return fig, ax

def plot_simulated_vs_analytical(decay_model, persistence, t_values, gamma_simulada,
                                 A_fit, gamma_analitica, params=None):
    '''
    Figure: best_{best_qubit_num}_decay_rate
    '''
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6))

    # Left: linear scale
    ax1.plot(t_values, persistence, 'b-', lw=2, label='Simulation')
    ax1.plot(t_values, decay_model(t_values, gamma_simulada, A_fit), 
            'r--', lw=1.5, label=f'Fit: Γ={gamma_simulada:.3f}')
    ax1.plot(t_values, np.exp(-gamma_analitica * t_values), 
            'k:', lw=1.5, label=f'Schwinger: Γ={gamma_analitica:.3f}')
    ax1.set_xlabel('Time'); ax1.set_ylabel('$\\Gamma$')
    ax1.set_title('Vacuum Persistence $\\Gamma$ vs Time'); ax1.legend(); ax1.grid(True, alpha=0.3)

    # Right: logarithmic scale
    mask_pos = np.array(persistence) > 1e-4
    ax2.semilogy(t_values[mask_pos], np.array(persistence)[mask_pos], 'b.', ms=4, label='Simulation')
    ax2.semilogy(t_values, np.exp(-gamma_analitica * t_values), 'r--', lw=2, label='Schwinger Prediction')
    ax2.set_xlabel('Time'); ax2.set_ylabel('$log(\\Gamma)$')
    ax2.set_title('Rate validation (log scale)'); ax2.legend(); ax2.grid(True, which='both', alpha=0.3)

    # Verifies that the decay is a pure exponential (not relaxation)
    # In log scale, the points should be a straight line
    # If there is curvature at the beginning (first 5-10 points), it is relaxation
    # If the straight line starts from t=0, it is a pure Schwinger decay

    log_p = np.log(persistence)
    # Linear fit to log(persistence) to check if it is a straight line (pure exponential decay)
    slope_lineal = np.polyfit(t_values, log_p, 1)
    print(f"{getTimer()} INFO: log-linear slope: {-slope_lineal[0]:.4f}")
    print(f"{getTimer()} INFO: R² of linear fit: {np.corrcoef(t_values, log_p)[0,1]**2:.4f}")
    ax2.annotate(f"log-linear slope: {slope_lineal[0]:.4f}\nR²: {np.corrcoef(t_values, log_p)[0,1]**2:.4f}",
                xy=(0.95, 0.8), xycoords='axes fraction', fontsize=10, horizontalalignment='right', verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5)) 
    # R² > 0.99 → pure exponential decay (Schwinger)
    # R² < 0.95 → there is curvature, a mixture of effects

    plt.suptitle("Exponential Decay Fit and Schwinger Prediction")
    if params:
        fig.text(0.5, 0.945, f"Parameters: {params}",
            fontsize=10, ha='center', va='top')
    plt.tight_layout()

    return fig, (ax1, ax2)

def plot_persistenece_vs_time_regimes(evolution_data, cut_off_times, params=None):
    '''
    Figure: best_{best_qubit_num}_persistenece_vs_time
    '''
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(evolution_data.index.values, evolution_data["Persistence"].values)
    ax.axvspan(0, cut_off_times["T_Zeno_End"], alpha=0.2, color='gray', label='Zeno Effect')
    ax.axvspan(cut_off_times["T_Zeno_End"], cut_off_times["T_Schwinger_End"], alpha=0.2, color='blue', label='Schwinger Regime')
    if cut_off_times["T_Interference_End"] is not None:
        ax.axvspan(cut_off_times["T_Schwinger_End"], cut_off_times["T_Interference_End"], alpha=0.2, color='orange', label='Interference')
        ax.axvspan(cut_off_times["T_Interference_End"], cut_off_times["T_Revivals_End"], alpha=0.2, color='red', label='Revivals')        
    else:
        ax.axvspan(cut_off_times["T_Schwinger_End"], cut_off_times["T_Revivals_End"], alpha=0.2, color='red', label='Revivals')        
    ax.legend()
    ax.grid(True)
    ax.set_xlabel("Time")
    ax.set_ylabel("$\\Gamma$")
    plt.suptitle(f"Vacuum Persistence $\\Gamma$ vs Time and Regimes")
    if params:
        ax.set_title(f"Parameters: {params}", fontsize=10)
    plt.tight_layout()
    return fig, ax

def plot_gamma_vs_e0(fit_results_df, params=None):
    '''
    Figure: e0_quench_{qubits_num}qubits
    '''
    fig, ax = plt.subplots()
    ax.plot(fit_results_df.index, fit_results_df["Gamma_Analytical"], "ro", label="$\\Gamma$ (Analytical)", markersize=5)
    ax.plot(fit_results_df.index, fit_results_df["Gamma_Simulated"],  "bs",  label="$\\Gamma$ (Simulated)",  markersize=5)
    ax.set_xlabel("$\\varepsilon_0$")
    ax.set_ylabel("$\\Gamma$")
    plt.suptitle("Vacuum Persistence $\\Gamma$ vs Background Field $\\varepsilon_0$")
    if params:
        ax.set_title(f"Parameters: {params}", fontsize=10)
    ax.legend()
    plt.grid(True)
    return fig, ax

def ind_plot_gamma_electricField(ax,
                             gamma_simulated, field_values,
                             field_tag="\\varepsilon_0"):

    '''
    Individual subplots for function plot_gamma_vs_electricField
    '''
    log_gamma_div_e0 = np.log(gamma_simulated / field_values)
    x, y = 1/field_values, log_gamma_div_e0
    ax.plot(x, y, "ro", label="$\\log(\\Gamma / E)$", markersize=5)
    num   = "{" + "1" + "}"
    field = "{" + field_tag + "}"
    ax.set_xlabel(rf"$\frac{num}{field}$")
    gamma = "{" + "\\Gamma" + "}"
    field = "{" + field_tag + "}"
    ax.set_ylabel(rf"$\log\left(\frac{gamma}{field}\right)$")
    # Fit to a line to check if it's linear in log scale
    def linear_model(x, a, b):
        return a * x + b
    popt, pcov = curve_fit(linear_model, x, y)
    a_fit, b_fit = popt
    # Add line in the plot
    x_fit = np.linspace(x.min(), x.max(), 100)
    y_fit = linear_model(x_fit, a_fit, b_fit)
    ax.plot(x_fit, y_fit, "b--", label=f"Fit: a={a_fit:.4f}, b={b_fit:.4f}")
    ax.legend()
    ax.grid(True)
    return ax, popt

def plot_gamma_vs_electricField(gamma_simulated, e0_values, field_values,
                                params=None):
    '''
    Figure: best_{best_qubit_num}_logPersistenece_vs_electricField
    '''
    fit_params = {}
    fig, axes = plt.subplots(1,2, figsize=(12,6))
    ax, popt = ind_plot_gamma_electricField(axes[0], gamma_simulated, e0_values,
                                field_tag="\\varepsilon_0")
    fit_params["e0"] = popt
    ax.set_title("Persistence Log vs Background Field $\\varepsilon_0$")
    ax, popt = ind_plot_gamma_electricField(axes[1], gamma_simulated, field_values,
                                field_tag="\\langle E_n \\rangle")
    fit_params["E_n_Mean"] = popt
    ax.set_title("Persistence Log vs Mean Electric Field $\\langle E_n \\rangle$")
    plt.suptitle("Persistence Log Fit")
    if params:
        fig.text(0.5, 0.945, f"Parameters: {params}",
            fontsize=10, ha='center', va='top')

    return fig, axes, fit_params
