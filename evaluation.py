


from dataset_reader import AbstractDatasetReader
from inference_tflite_model import AbstractInferenceModel
import numpy as np
import copy

class Evaluation(object):
    def __init__(self,dataset_reader:AbstractDatasetReader,model:AbstractInferenceModel,load:bool,filter_low_score:float) -> None:
        self.dataset_reader = dataset_reader
        self.inference_model = model
        self.load = load
        self.filter_low_score = filter_low_score
        
    def start_evaluation(self):
        number_of_images = self.dataset_reader.get_number_of_images()
        image_dict = {}
        summarized_resuls = []
        
        if not self.load: 
            for index in range(0,number_of_images):
                resized_image,ground_truth_bbox,image_identifier = self.dataset_reader.read_next_sample()
                predict_bbox, detected_scores = self.inference_model.predict(resized_image)
                predict_bbox, detected_scores = self._filterLowConfidence(predict_bbox, detected_scores)
                try:
                    pred = np.concatenate((np.array(predict_bbox, dtype=np.float32), np.array(detected_scores, dtype=np.float32).reshape(-1,1)), axis=1)
                except:
                    pred = np.array([]).reshape((0, 5))
                label = np.array((ground_truth_bbox), dtype=np.float32)
                predDict = {categoryName: pred}
                labelDict = {categoryName: label}

                categoryName = "Person"
                tp, fp, fn = self._getMetrics(copy.deepcopy(predDict), copy.deepcopy(labelDict), [categoryName], iouThreshold)

                d = {
                    'imageId': image_identifier,
                    'groundTruth': labelDict,
                    'prediction': predDict,
                    'truePositives': tp,
                    'falsePositives': fp,
                    'falseNegatives': fn,
                }
                # s = pd.Series(d)
                summarized_resuls.append(d)
                img = np.array(resized_image)[..., (2,1,0)] # rgb to bgr
                image_dict[image_identifier] = img
        else:
             for index in range(0,number_of_images):
                resized_image,ground_truth_bbox,image_identifier = self.dataset_reader.read_next_sample()
                img = np.array(resized_image)[..., (2,1,0)] # rgb to bgr
                image_dict[image_identifier] = img
                np.load()

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

