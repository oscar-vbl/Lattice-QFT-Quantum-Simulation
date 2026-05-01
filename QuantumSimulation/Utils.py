import os
import datetime as dt
import json
from qiskit import qpy
import pandas as pd
from typing import Callable, Mapping, Any, Iterable
import pickle

from _config import PROJECT_ROOT, MODULES_ROOT, CONFIGS_FOLDER, PLOTS_FOLDER, DATA_FOLDER

def getTimer():
    return dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def parseDictToPlot(
        d: Mapping[str, Any],
        remove_keys: Iterable=[],
        rename_keys: Mapping[str, str]={}):
    params = {rename_keys.get(k, k): v for k, v in d.items() if k not in remove_keys}
    return ", ".join(f"{k}={v:.2f}" if isinstance(v, float) else f"{k}={v}" for k, v in params.items())

def saveJsonConfig(config, saveName="config", saveFolder=None):
    if not saveFolder:
        saveFolder = CONFIGS_FOLDER
    os.makedirs(saveFolder, exist_ok=True)
    savePath = os.path.join(saveFolder, saveName)
    with open(savePath, "w") as f:
        json.dump(config, f, indent=4)

def loadJsonConfig(fileName, folderName=None):
    if not folderName:
        saveFolder = CONFIGS_FOLDER
    os.makedirs(saveFolder, exist_ok=True)
    savePath = os.path.join(saveFolder, fileName)
    with open(savePath, "r") as f:
        return json.load(f)

def load_initial_state(folderName, fileName, rootPath=None):
    if rootPath is None: rootPath = DATA_FOLDER
    save_folder = os.path.join(rootPath, folderName)
    initial_state_path = os.path.join(save_folder, fileName)
    with open(initial_state_path, "rb") as handle:
        initial_state = qpy.load(handle)[0]
    return initial_state

def load_data(folderName, fileName, rootPath=None, indexSet=None):
    if rootPath is None: rootPath = DATA_FOLDER
    save_folder = os.path.join(rootPath, folderName)
    data_path   = os.path.join(save_folder, fileName)
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"File {data_path} not found.")
    if fileName.endswith(".csv"):
        data = pd.read_csv(data_path)
        if indexSet: data.set_index(indexSet, inplace=True)
    elif fileName.endswith(".qpy"):
        with open(data_path, "rb") as handle:
           data = qpy.load(handle)[0]
    elif fileName.endswith(".pkl"):
         with open(data_path, "rb") as handle:
            data = pickle.load(handle)
    else:
        raise ValueError(f"Unsupported file format: {os.path.splitext(fileName)[1]}")
    return data

def save_data(data, folderName, fileName, rootPath=None, overWrite=True, **kwargs):
    if rootPath is None: rootPath = DATA_FOLDER
    save_folder = os.path.join(rootPath, folderName)
    os.makedirs(save_folder, exist_ok=True)
    data_path   = os.path.join(save_folder, fileName)
    if not overWrite and os.path.exists(data_path):
        data_path = getValidFileName(data_path)
    if fileName.endswith(".csv"):
        data.to_csv(data_path)
    elif fileName.endswith(".qpy"):
        with open(data_path, "wb") as file:
            qpy.dump(data, file)
    elif fileName.endswith(".pkl"):
        with open(data_path, "wb") as file:
            pickle.dump(data, file)
    elif fileName.endswith(".png") or \
            fileName.endswith(".jpg") or \
            fileName.endswith(".jpeg") or \
            fileName.endswith(".pdf") or \
            fileName.endswith(".svg"):
        data.savefig(data_path, **kwargs)
    else:
        raise ValueError(f"Unsupported file format: {os.path.splitext(fileName)[1]}")
    return data

def getValidFileName(save_path):
    base_name, ext = os.path.splitext(save_path)
    counter = 1
    new_save_path = save_path
    while os.path.exists(new_save_path):
        new_save_path = f"{base_name} ({counter}){ext}"
        counter += 1
    return new_save_path

def sortEigenstates(eigVals, eigVecs):
    idx = eigVals.argsort()
    eigVals = eigVals[idx]
    eigVecs = eigVecs[:,idx]
    return eigVals, eigVecs

def drawCircuitLatex(circuit, saveName="circuit"):
    text = circuit.draw("latex_source")
    saveFolder = os.path.join(PROJECT_ROOT, "Circuits")
    os.makedirs(saveFolder, exist_ok=True)
    savePath = os.path.join(saveFolder, f"{saveName}.txt")
    with open(savePath, "w") as f:
        f.write(text)

def func_return(
        func: Callable,
        params: Mapping = {},
        default=None,
        expect_type: type | None = None) -> Any:
    '''try return a function given its parameters, else return default'''
    try:
        result = func(**params)
        if expect_type is not None:
            assert isinstance(result, expect_type), f"WARNING: Function {func.__name__} did not return expected type {expect_type}, got {type(result)}. Review parameters..."
        return result
    except Exception as e:
        print(f"{getTimer()} WARNING: Exception {e} raised, review parameters...")
        return default

