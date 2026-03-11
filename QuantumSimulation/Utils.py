import os
import datetime as dt
import json

CONFIGS_FOLDER = os.path.join(os.path.dirname(__file__), "Configs")
PLOTS_FOLDER   = os.path.join(os.environ["USERPROFILE"], "Documents", "uned", "TFM-Schwinger", "Plots")
os.makedirs(CONFIGS_FOLDER, exist_ok=True)
os.makedirs(PLOTS_FOLDER,   exist_ok=True)

def getTimer():
    return dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

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
    saveFolder = os.path.join(os.environ["USERPROFILE"], "Documents", "uned", "TFM-Schwinger", "Circuits")
    os.makedirs(saveFolder, exist_ok=True)
    savePath = os.path.join(saveFolder, f"{saveName}.txt")
    with open(savePath, "w") as f:
        f.write(text)
