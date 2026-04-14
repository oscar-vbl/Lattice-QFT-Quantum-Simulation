import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
MODULES_ROOT = Path(__file__).parent
CONFIGS_FOLDER = os.path.join(MODULES_ROOT, "Configs")
PLOTS_FOLDER   = os.path.join(PROJECT_ROOT, "Plots")
DATA_FOLDER    = os.path.join(PROJECT_ROOT, "Data")
os.makedirs(CONFIGS_FOLDER, exist_ok=True)
os.makedirs(PLOTS_FOLDER,   exist_ok=True)
os.makedirs(DATA_FOLDER,    exist_ok=True)
