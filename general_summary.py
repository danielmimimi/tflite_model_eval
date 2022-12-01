import os
import numpy as np
import pandas as pd
class GeneralSummary(object):
    def __init__(self, path_to_saves:str):
        self.path_to_saves = path_to_saves
        
    
    def create_summary(self):
        precision_recall = []
        for root, dirs, files in os.walk(self.path_to_saves):      
            for dir in dirs:
                for root, dirs, files in os.walk(self.path_to_saves.joinpath(dir)):
                    d = {"Recall":[],"Precision":[],"ResultMatrix":[], "InferenceTime":[]}
                    for file in files:                
                        if "recalls.npy" in file:
                            d["Recall"]= self.path_to_saves.joinpath(dir).joinpath(file)
                        if "precisions.npy" in file:
                            d["Precision"]= self.path_to_saves.joinpath(dir).joinpath(file)
                        if "detectionResultMatrix.csv" in file:
                            d["ResultMatrix"] = self.path_to_saves.joinpath(dir).joinpath(file)
                        if "inferenceTime.npy" in file:
                            d["InferenceTime"] = self.path_to_saves.joinpath(dir).joinpath(file)
                    try:
                        if d['Recall'].exists():
                            precision_recall.append(d)
                    except:
                        d = {"Recall":[],"Precision":[],"ResultMatrix":[], "InferenceTime":[]}
        data = pd.DataFrame(precision_recall)
        for files in precision_recall:
            print(files)
        
                    
        