from abc import ABC,abstractmethod
import numpy as np
import time

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
        np.save(path_to_save.joinpath("inferenceTime.npy"),self.inference_measurements)
            
        
