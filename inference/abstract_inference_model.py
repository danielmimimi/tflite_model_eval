from abc import ABC,abstractmethod
import numpy as np
import time
import os

class AbstractInferenceModel(ABC):
    def __init__(self, path_to_model:str):
        self.path_to_model = path_to_model
        self.inference_measurements = []
        
    @abstractmethod
    def get_image_size(self):
        pass
      
    @abstractmethod
    def predict(self,image:np.array):
        pass
    
    @abstractmethod
    def get_model_name(self)->str:
        pass
    
    @abstractmethod
    def get_model_dir(self)->str:
        pass
    
    def inference(self):
        self.stop_watch(True)
        self.interpreter.invoke()
        self.stop_watch(False)
    
    def stop_watch(self,start:bool):
        if start :
            self.start = time.time()
        else:
            self.end = time.time()
            self.inference_measurements.append(self.end - self.start)
            
    def save_inference_time(self,path_to_save:str):
        model_name = self.get_model_name()
        model_dir = self.get_model_dir()
        folder_to_save_path = os.path.join(model_dir,"saves")
        file_name = model_name+"_"+path_to_save.stem+"_inferenceTime.npy"
        np.save(os.path.join(folder_to_save_path,file_name),self.inference_measurements)
            
        
