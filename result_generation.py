


from pathlib import Path
from PIL import Image, ImageFont, ImageDraw, ImageEnhance
from dataset_reader import AbstractDatasetReader
from inference.abstract_inference_model import AbstractInferenceModel
import numpy as np
import copy
import os
import shutil

class ResultGeneration(object):
    def __init__(self,dataset_reader:AbstractDatasetReader,model:AbstractInferenceModel,save_path:Path,load:str,filter_low_score:float,iou_threshold:float,save_only:bool,peak_results_path:str = "") -> None:
        self.dataset_reader = dataset_reader
        self.inference_model = model
        self.load = load
        self.filter_low_score = filter_low_score
        self.iou_threshold = iou_threshold
        self.save_path = save_path
        self.categoryName = "person"
        self.save_only = save_only
        self.peak_results_path = peak_results_path
        
    def _drawRectangle(self,resized_image,prediction_bbox,ground_truth_bbox,path_to_store,image_identifier): 
        draw_img = ImageDraw.Draw(resized_image)
        
        for prediction in prediction_bbox:
            top_left = (int(prediction[0]),int(prediction[1]))
            bottom_right = (int(prediction[2]),int(prediction[3]))
            score = prediction[4]
            draw_img.rectangle((top_left,bottom_right), outline="yellow")
            draw_img.text((top_left[0],top_left[1]- 2), str(score),color="red")
        
        for ground_truth in ground_truth_bbox:
            print(ground_truth)
            top_left = (int(ground_truth[0]),int(ground_truth[1]))
            bottom_right = (int(ground_truth[2]),int(ground_truth[3]))
            draw_img.rectangle((top_left,bottom_right), outline="cyan")
            
        resized_image.save(os.path.join(path_to_store,image_identifier), "JPEG")
        return

    def start_evaluation(self):
        number_of_images = self.dataset_reader.get_number_of_images()
        image_dict = {}
        summarized_resuls = []
        
        if not self.load: 
            for _ in range(0,number_of_images):
                resized_image,ground_truth_bbox,image_identifier = self.dataset_reader.read_next_sample()
                pred = self._predict_and_filter(resized_image)
                
                label = np.array((ground_truth_bbox), dtype=np.float32)
                predDict = {self.categoryName: pred}
                labelDict = {self.categoryName: label}

                tp, fp, fn = self._getMetrics(copy.deepcopy(predDict), copy.deepcopy(labelDict), [self.categoryName], self.iou_threshold)

                
                if self.peak_results_path and _ % 50 == 0:
                    self._drawRectangle(resized_image,pred,ground_truth_bbox,self.peak_results_path,image_identifier)
                
                d = {
                    'imageId': image_identifier,
                    'groundTruth': labelDict,
                    'prediction': predDict,
                    'truePositives': tp,
                    'falsePositives': fp,
                    'falseNegatives': fn,
                }

                summarized_resuls.append(d)
                img = np.array(resized_image)[..., (2,1,0)] # rgb to bgr
                image_dict[image_identifier] = img
            if self.save_only:
                model_name = self.inference_model.get_model_name()
                model_dir = self.inference_model.get_model_dir()
                folder_to_save_path = os.path.join(model_dir,"saves")
                if os.path.exists(folder_to_save_path):
                    shutil.rmtree(folder_to_save_path)
                os.makedirs(folder_to_save_path)
                np.save(os.path.join(model_dir,"saves",model_name+".npy"),summarized_resuls)
        else:
            summarized_resuls = np.load(self.load,allow_pickle=True)
            for _ in range(0,number_of_images):
                resized_image,ground_truth_bbox,image_identifier = self.dataset_reader.read_next_sample()
                img = np.array(resized_image)[..., (2,1,0)] # rgb to bgr
                image_dict[image_identifier] = img
        return summarized_resuls,image_dict
        
    def _predict_and_filter(self,resized_image):
        predict_bbox, detected_scores = self.inference_model.predict(resized_image)
        predict_bbox, detected_scores = self._filterLowConfidence(predict_bbox, detected_scores)
        try:
            pred = np.concatenate((np.array(predict_bbox, dtype=np.float32), np.array(detected_scores, dtype=np.float32).reshape(-1,1)), axis=1)
        except:
            pred = np.array([]).reshape((0, 5))
        return pred

    def _filterLowConfidence(self,predBbox, predScores):
        predBboxNew = []
        predScoresNew = []
        for box, score in zip(predBbox, predScores):
            if score >= self.filter_low_score :
                predBboxNew.append(box)
                predScoresNew.append(score)
        return predBboxNew, predScoresNew
    
    
    def _getMetrics(self,predDict, labelDict, partNames, iouThrs):
        tpd = {}
        fpd = {}
        fnd = {}

        for partName in partNames:
            bboxLabel = labelDict[partName]
            bboxPred = predDict[partName]
            bboxPred = bboxPred[bboxPred[:, 4].argsort()[::-1]] # sort by score

            boxIou = self._box_iou(bboxPred, bboxLabel)
            if boxIou.size > 0:
                # init
                tp = np.zeros((boxIou.shape[0], 1), dtype=bool) 
                fn = np.ones(boxIou.shape[1], dtype=bool)
                for predIndex, predIou in enumerate(boxIou):
                    maxPredIouValue = predIou.max()
                    matchMax = (predIou >= iouThrs) & (predIou == maxPredIouValue) & (predIou > 0)
                    numMatch = sum(matchMax)
                    if numMatch > 0:
                        assert numMatch == 1
                        labelMatchIndex = matchMax.argmax()
                        if fn[labelMatchIndex] == True:
                            fn[labelMatchIndex] = False
                            tp[predIndex] = True
                        else:
                            tp[predIndex] = False
                fp = ~tp

            elif len(bboxLabel) == 0 and len(bboxPred) > 0:
                tp = np.array([[False] for i in range(len(bboxPred))])
                fp = np.array([[True] for i in range(len(bboxPred))])
                fn = np.array([True for i in range(len(bboxLabel))])
            elif len(bboxLabel) > 0 and len(bboxPred) == 0:
                tp = np.array([])
                fp = np.array([])
                fn = np.array([True for i in range(len(bboxLabel))])
            else:
                tp = np.array([])
                fp = np.array([])
                fn = np.array([True for i in range(len(bboxLabel))])

            assert (sum(tp) + sum(fn)) == len(bboxLabel)

            tpd[partName] = tp
            fpd[partName] = fp
            fnd[partName] = fn

        return tpd, fpd, fnd
    
    def _box_iou(self,box1, box2):
        """
        Return intersection-over-union (Jaccard index) of boxes.
        Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
        Arguments:
            box1 (Tensor[N, 4])
            box2 (Tensor[M, 4])
        Returns:
            iou (Tensor[N, M]): the NxM matrix containing the pairwise
                IoU values for every element in boxes1 and boxes2
        """

        def box_area(box):
            # box = 4xn
            return (box[2] - box[0]) * (box[3] - box[1])

        area1 = box_area(box1.T)
        area2 = box_area(box2.T)

        # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
        inter = (np.minimum(box1[:, None, 2:4], box2[:, 2:]) - np.maximum(box1[:, None, :2], box2[:, :2])).clip(0,None).prod(2)
        return inter / (area1[:, None] + area2 - inter)  # iou = inter / (area1 + area2 - inter)

