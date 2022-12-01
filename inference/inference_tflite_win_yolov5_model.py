from pathlib import Path
from inference.abstract_inference_model import AbstractInferenceModel
import tensorflow as tf
import cv2 
import numpy as np


def xywhtoxyxy(x):
    y1 = x[:, 0] - x[:, 2] / 2  
    y2 = x[:, 1] - x[:, 3] / 2  
    y3 = x[:, 0] + x[:, 2] / 2  
    y4 = x[:, 1] + x[:, 3] / 2  
    y=np.squeeze(np.dstack((y1,y2,y3,y4)))
    return y

def scale_coords(img1_shape, coords, img0_shape):

	gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  
	pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2 

	coords[:, [0, 2]] -= pad[0]  
	coords[:, [1, 3]] -= pad[1]  
	coords[:, :4] /= gain
	clip_coords(coords, img0_shape)
	return coords


def clip_coords(boxes, shape):

		boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, shape[1])  
		boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, shape[0])  


def letterbox(im, new_shape=(640, 640), color=(114, 114, 114),  stride=32):

	shape = im.shape[:2]  

	ratio = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

	new_unpad = int(round(shape[1] * ratio)), int(round(shape[0] * ratio))
	dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  

	dw = dw/2  
	dh = dh/2

	if shape[::-1] != new_unpad:  
		im = cv2.resize(im, new_unpad)
	top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
	left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
	im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color) 

	return im



def nms(rect_box,scores,nms_threshold):
	x1 = rect_box[:, 0]
	y1 = rect_box[:, 1]
	x2 = rect_box[:, 2]
	y2 = rect_box[:, 3]
	areas = (x2 - x1 + 1) * (y2 - y1 + 1)
	temp_iou=[]
	order=scores.argsort()[::-1]

	while order.size > 0:
		i = order[0]
		temp_iou.append(i)

		xx1 = np.maximum(x1[i], x1[order[1:]])
		yy1 = np.maximum(y1[i], y1[order[1:]])
		xx2 = np.minimum(x2[i], x2[order[1:]])
		yy2 = np.minimum(y2[i], y2[order[1:]])


		w = np.maximum(0.0, xx2 - xx1 + 1)
		h = np.maximum(0.0, yy2 - yy1 + 1)
		inter = w * h

		ovr = inter / (areas[i] + areas[order[1:]] - inter)

		inds = np.where(ovr <= nms_threshold)[0]

		order = order[inds + 1]
	return temp_iou


class InferenceTfliteImx8Yolov5model(AbstractInferenceModel):
    def __init__(self,path_to_model:str):
        super().__init__(path_to_model)
        self.interpreter = tf.lite.Interpreter(model_path=self.path_to_model)
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
        self.inference()
        
        hight,width = self.input_image_size[1:3]
        output_data = self.interpreter.get_tensor(self.model_output_details[0]['index'])
        if 'quantization' in self.model_output_details[0]:
            scale, zero_point = self.model_output_details[0]['quantization']
            if not (scale == 0 & zero_point==0):
                if scale == 0:
                    output_data=output_data - zero_point
                else:
                    output_data= scale * (output_data - zero_point)
        
        
        detected_boxes, detected_scores = self._prepare_yolo_output(output_data)  
        return detected_boxes, detected_scores
    
    def get_image_size(self):
        return self.input_image_size
     
    def get_model_name(self)->str:
        model_path = Path(self.path_to_model)
        return model_path.stem
    
    def get_model_dir(self)->str:
        model_path = Path(self.path_to_model)
        return model_path.parent
    
    def _prepare_yolo_output(self,output_data):
        hight,width = self.input_image_size[1:3]
        conf_threshold = 0.2
        x=np.squeeze(output_data)
        xc=x[...,4]>conf_threshold
        x=x[xc]
        nms_threshold=0.45
        max_wh = 7680  
        
        x[...,0]*=width
        x[...,1]*=hight
        x[...,2]*=width
        x[...,3]*=hight
        
        detected_boxes = []
        detected_scores = []
        if x.shape[0]:
            x[:, 5:] *= x[:, 4:5]
            rect_box=xywhtoxyxy(x[:, :4])
            conf=np.max(x[:,5:],axis=1)
            index=np.argmax(x[:,5:],axis=1)
            if len(rect_box.shape)==1:
                x=np.concatenate((rect_box,conf,index))
            else:
                conf=conf.reshape(len(conf),1)
                index=index.reshape(len(index),1)
                x=np.concatenate((rect_box,conf,index),axis=1)
                
            if len(x.shape)==1:
                x=x.reshape(1,x.shape[0])
          
            c=x[:,5:6]*max_wh
            boxes, scores = x[:, :4] + c, x[:, 4]
            result_index=nms(boxes,scores,0.0)
            
            result=x[result_index]
            result[:,:4]=scale_coords([width,hight],result[:,:4],[hight,width]).round()
            num_result=result.shape[0]
            
            for i in range(0,num_result):
                left=int(result[i][0])
                top=int(result[i][1])
                right=int(result[i][2])
                bottom=int(result[i][3])

                score=result[i][4]
                class_id=result[i][5]
                detected_boxes.append([left,top,right,bottom])
                detected_scores.append(score)
        return detected_boxes, detected_scores
