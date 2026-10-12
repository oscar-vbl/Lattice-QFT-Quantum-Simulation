# QuantumSimulation/processing/dataframes.py

from __future__ import annotations

import pandas as pd

from QuantumSimulation.core.types import TrajectoryResult

def extract_trajectory_dataframes(trajectory: TrajectoryResult) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Converts a strictly typed TrajectoryResult into Pandas DataFrames.
    Dynamically unpacks multi-dimensional observables using their component_labels,
    completely agnostic to the underlying physics.
    """
    data_dict = {}
    error_dict = {}

    for obs_name, series in trajectory.observables.items():
        # 1D Series (Scalars like Energy or Persistence)
        if series.values.ndim == 1:
            data_dict[obs_name] = series.values
            error_dict[obs_name] = series.uncertainties
            
        # 2D Series (Arrays like Electric Field or Pair Creation)
        elif series.values.ndim == 2:
            num_components = series.values.shape[1]
            labels = series.component_labels
            
            # Fallback dynamic naming if labels are missing
            if not labels or len(labels) != num_components:
                labels = [f"component_{i}" for i in range(num_components)]
                
            for i, label in enumerate(labels):
                # Cleanly combine observable name and component label (e.g., 'Pair_Creation_electrons')
                col_name = f"{obs_name}_{label}"
                
                data_dict[col_name] = series.values[:, i]
                error_dict[col_name] = series.uncertainties[:, i]

    # Build DataFrames using the uniform time grid
    df_values = pd.DataFrame(data_dict, index=trajectory.times)
    df_errors = pd.DataFrame(error_dict, index=trajectory.times)
    
    df_values.index.name = "Time"
    df_errors.index.name = "Time"

    return df_values, df_errors