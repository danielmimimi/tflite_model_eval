from pathlib import Path
from inference.abstract_inference_model import AbstractInferenceModel
from tflite_runtime.interpreter import load_delegate
from tflite_runtime.interpreter import Interpreter
import cv2 
import numpy as np

class InferenceTfliteImx8EfficientNpumodel(AbstractInferenceModel):
    def __init__(self,path_to_model:str):
        super().__init__(path_to_model)
        delegate = [load_delegate("libedgetpu.so.1")]
        self.interpreter = Interpreter(model_path=self.path_to_model,experimental_delegates=delegate)
        self.interpreter.allocate_tensors()
        self.model_input_details = self.interpreter.get_input_details()
        self.model_output_details = self.interpreter.get_output_details()
        self.input_image_size = self.model_input_details[0]['shape']

    def predict(self,image:np.array):
        """Bilder in Ordner speichern, returns [BBOX(tl,br)]. box format: [x0, y0, x1, y1]"""
        if self.model_input_details[0]['dtype'] == np.float32 :
            if len(np.array(image).shape) == 2 :
                image = cv2.cvtColor(np.array(image),cv2.COLOR_GRAY2BGR)
            self.interpreter.set_tensor(self.model_input_details[0]["index"],np.expand_dims(np.array(image),0).astype(np.float32))
        elif self.model_input_details[0]['dtype'] == np.uint8 :
            if len(np.array(image).shape) == 2 :
                image = cv2.cvtColor(np.array(image),cv2.COLOR_GRAY2BGR)
            self.interpreter.set_tensor(self.model_input_details[0]["index"],np.expand_dims(np.array(image),0))
        else:
            raise Exception("No such Datatype in models")
        self.interpreter.invoke()
        hight,width = self.input_image_size[1:3]
        # get realtive bbox coords
        detected_class_labels =  np.squeeze(self.interpreter.get_tensor(self.model_output_details[3]['index']).astype(int))
        detected_boxes =  np.squeeze(self.interpreter.get_tensor( self.model_output_details[1]['index']))
        detected_scores =  np.squeeze(self.interpreter.get_tensor( self.model_output_details[0]['index']))

        detected_boxes = detected_boxes[:, (1,0,3,2)]
        detected_boxes[:] *= width,hight,width,hight
        personIndex = 0
        return detected_boxes[detected_class_labels == personIndex].tolist(), detected_scores[detected_class_labels == personIndex].tolist()
    
    def get_image_size(self):
        return self.input_image_size
     
    def get_model_name(self)->str:
        model_path = Path(self.path_to_model)
        return model_path.stem
    
    def get_model_dir(self)->str:
        model_path = Path(self.path_to_model)
        return model_path.parent
    
