
from abc import ABC,abstractmethod
import os
import numpy as np
from PIL import Image
import xmltodict

class AbstractDatasetReader(ABC):
    def __init__(self, file_path,image_input_size):
        self.file_path = file_path
        self.image_input_size = image_input_size

    @abstractmethod
    def load_dataset(self):
        pass
      
    @abstractmethod
    def read_next_sample(self): 
        pass
    
    @abstractmethod
    def get_number_of_images(self) -> int: 
        pass
    
    
class TfRecordReader(AbstractDatasetReader):
    def __init__(self,file_path,image_input_size):
        super().__init__(file_path,image_input_size)
        
    def load_dataset(self):
        pass
    
    @abstractmethod
    def read_next_sample(self):
        pass
    
    @abstractmethod
    def get_number_of_images(self) -> int: 
        pass
    

class ImageAnnotationReader(AbstractDatasetReader):
    def __init__(self,file_path,image_input_size):
        super().__init__(file_path,image_input_size)
        self.annotation_path = os.path.join(self.file_path,"annotations")
        self.image_path = os.path.join(self.file_path,"images")
        self.counter = 0
        
    def load_dataset(self):
        self.images = os.listdir(self.image_path)
        self.annotations = os.listdir(self.annotation_path)

    def read_next_sample(self):
        image_identifier = self.images[self.counter].split(" ")[0]
        image = Image.open(os.path.join(self.image_path,self.images[self.counter]))
        base_width, base_height = image.size
        bounding_boxes = self._read_xml_annotation(os.path.join(self.annotation_path,self.annotations[self.counter]))
        
        # RESIZE IMAGE
        resized_image = image.resize(self.image_input_size[1:3])
        new_width, new_height = resized_image.size
        # RESIZE BOUNDING BOX
        resized_bounding_box = []
        for unresized_bounding_boxes in bounding_boxes:
            xmin,ymin,xmax,ymax = unresized_bounding_boxes
            xmin_rel,xmax_rel = xmin/base_width,xmax/base_width
            ymin_rel,ymax_rel = ymin/base_height,ymax/base_height
            resized_bounding_box.append([int(xmin_rel*new_width),int(ymin_rel*new_height),int(xmax_rel* new_width),int(ymax_rel*new_height)])
        
        self.counter = self.counter + 1
        
        return resized_image, resized_bounding_box,image_identifier
        
    def get_number_of_images(self) -> int: 
        return len(self.images)
    
    def _read_xml_annotation(self,xml_file:str):
        bbox_coordinates = []
        with open(xml_file) as file:
            file_data = file.read() # read file contents   
            # parse data using package
            dict_data = xmltodict.parse(file_data) 
            for object in dict_data['annotation']['object']:
                        if object == 'name':
                            xmin = int(float(dict_data['annotation']['object']['bndbox']['xmin']))
                            ymin = int(float(dict_data['annotation']['object']['bndbox']['ymin']))
                            xmax = int(float(dict_data['annotation']['object']['bndbox']['xmax']))
                            ymax = int(float(dict_data['annotation']['object']['bndbox']['ymax']))
                            bbox_coordinates.append([xmin, ymin, xmax, ymax])
                            break
                        else:
                            xmin = int(float(object['bndbox']['xmin']))
                            ymin = int(float(object['bndbox']['ymin']))
                            xmax = int(float(object['bndbox']['xmax']))
                            ymax = int(float(object['bndbox']['ymax']))
                            bbox_coordinates.append([xmin, ymin, xmax, ymax])
        return bbox_coordinates