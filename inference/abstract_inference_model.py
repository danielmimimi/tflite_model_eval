from abc import ABC,abstractmethod
import numpy as np

class AbstractInferenceModel(ABC):
    def __init__(self, path_to_model:str):
        self.path_to_model = path_to_model
        
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