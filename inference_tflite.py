import tensorflow as tf
import argparse
import numpy as np
from PIL import Image
import io
import cv2
import os
from tqdm import tqdm
import pandas as pd
import copy
from pathlib import Path
import shutil
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve
from datetime import datetime

def main():

    print('\nstart script: {}'.format(__file__))

    args = getArgs()

    # needed to read .record file
    tf.enable_eager_execution()
    tflite_model = InferenceTflite(args.modelname)

    raw_dataset = tf.data.TFRecordDataset(args.testimages)
    

    # hard coded stuff
    categoryIndex = 0 # person
    categoryName = 'person'
    iouThreshold = 0.5
    scoreThreshold = 0.5
    outputDir = Path(args.savepath)
    debugMode = False

    evalDf, imageDict = createData(raw_dataset, tflite_model, categoryName, scoreThreshold, iouThreshold, debugMode)   

    correctImagesPath, fpImagePaths, fnImagePaths, exportDirRoot = generateImages(args,evalDf, imageDict, categoryName, outputDir, scoreThreshold, iouThreshold)
    precisionRecallCurvePath = createPrecisionRecallCurve(evalDf, categoryName, exportDirRoot)
    precDict, recallDict, numPosDict, numTruePosDict, numFalsePosDict, numFalseNegDict, numberOfImages = getOverallResults(evalDf, categoryName)
    
    exportEval(
        correctImagesPath, 
        fpImagePaths, 
        fnImagePaths, 
        precDict, 
        recallDict, 
        numPosDict, 
        numTruePosDict, 
        numFalsePosDict, 
        numFalseNegDict, 
        numberOfImages, 
        scoreThreshold, 
        iouThreshold, 
        categoryName,
        precisionRecallCurvePath,
        exportDirRoot,)

    print('\nscript successfully finished')


def getArgs():
    parser = argparse.ArgumentParser(
                        prog = 'ProgramName',
                        description = 'What the program does',
                        epilog = 'Text at the bottom of help')

    parser.add_argument('-m','--modelname',default="/tensorflow/models/research/ssd_training/pretrained_model/ssdlite_mobiledet_edgetpu_320x320_coco_2020_05_19/fp32/model.tflite")
    parser.add_argument('-t','--testimages',default="/tensorflow/models/research/ssd_training/dataset/records/custom_coco_trainval.record-00000-of-00005")
    parser.add_argument('-s','--savepath',default="/tensorflow/models/research/ssd_training/train/eval_andreas/saved_image_path")
    parser.add_argument('-n','--name',default="coco_base")
    args = parser.parse_args()

    return args


def createPrecisionRecallCurve(evalDf, categoryName, exportDir):
    scores = []
    matches = []
    precisions = [1.0]
    recalls = [0.0]
    tp = 0
    fp = 0

    numOfGroundTruthObjects = sum([len(ds.groundTruth[categoryName]) for index, ds in evalDf.iterrows()])

    for index, ds in evalDf.iterrows():
        labelBoxs = ds.groundTruth[categoryName]
        prediction = ds.prediction[categoryName]
        predicitonScores = prediction[:, 4]
        predictionMatches = ds.truePositives[categoryName].ravel()

        scores += list(predicitonScores)
        matches += list(predictionMatches)
    
    sortedByScore = sorted(zip(scores, matches), key=lambda x: x[0], reverse=True)

    for score, match in sortedByScore:
        if match:
            tp += 1
        else:
            fp += 1

        precision = tp/(tp+fp)
        recall = tp/numOfGroundTruthObjects

        precisions.append(precision)
        recalls.append(recall)


    fig = plt.figure(figsize=(12, 6))
    plt.plot([0, 1, 1, 0, 0], [0, 0, 1, 1, 0], c='black')
    plt.plot(recalls, precisions, 'o--')
    plt.fill_between(recalls, precisions, step="pre", alpha=0.4)

    plt.xlabel('recall')
    plt.ylabel('precision')
    plt.title('Precision Recall Curve')
    
    plt.ylim((-0.05, 1.05)) 
    plt.xlim((-0.05, 1.05)) 
    plt.grid()

    plt.tight_layout()

    path = exportDir.joinpath('precisionRecallCurve.svg')
    plt.savefig(str(path))
    plt.close()

    imagePath = str(path.name)

    return imagePath



def exportEval(
    correctImagesPath, 
    fpImagePaths, 
    fnImagePaths, 
    precisionDict, 
    recallDict, 
    numPosDict, 
    numTruePosDict, 
    numFalsePosDict, 
    numFalseNegDict, 
    numberOfImages, 
    confidenceThreshold, 
    iouThreshold, 
    partName, 
    precisionRecallCurvePath,
    exportDirRoot):

    partNames = [partName]

    # metric matrix
    metricMatrix = np.zeros((len(partNames), 7), dtype=object)

    # html
    h = '<!DOCTYPE html>\n'
    h += '<html>\n'
    h += '<style>\n'
    h += 'body {font: normal 12px Verdana, Arial, sans-serif; }\n'
    h += '.column{float:left; padding-right: 50px}\n'
    h += '.row::after{content: ""; clear: both; display: table}\n'
    h += '</style>\n'
    h += '<body>\n'
    h += '<h1 id="toc">Evaluation Summary</h1>\n'

    h += '<h2>Part Table of Content</h2>'
    h += '<ul>\n'
    for partName in partNames:
        h += '<li><a href="#{}">{}</a></li>\n'.format('link_{}'.format(partName), partName.capitalize())

        h += '<ul>\n'
        h += '<li><a href="#{}">Metrics</a></li>\n'.format('link_{}_{}'.format(partName, 'metric'))
        h += '<li><a href="#{}">Images with false positives</a></li>\n'.format('link_{}_{}'.format(partName, 'fp'))
        h += '<li><a href="#{}">Images with false negatives</a></li>\n'.format('link_{}_{}'.format(partName, 'fn'))
        h += '<li><a href="#{}">Images all correct</a></li>\n'.format('link_{}_{}'.format(partName, 'correct'))
        h += '</ul>\n'

    h += '</ul>\n'

    h += '<h2>Settings</h2>\n'
    h += '<p>Confidence Threshold: {:.2f}</p>\n'.format(confidenceThreshold)
    h += '<p>Intersection over Union Threshold: {:.2f}</p>\n'.format(iouThreshold)

    h += '<h2>Images</h2>\n'
    h += '<p>Cyan box: ground truth</p>\n'
    h += '<p>Green box: correct prediction</p>'
    h += '<p>Red box: wrong prediction</p>'
    h += '<p>Box value: detection confidence</p>'
    h += '<p>Number of images: {}</p>\n'.format(numberOfImages)

    h += '<h2>Metrics Overview</h2>\n'
    h += '<METRIC_MATRIX>\n'
    h += '<br>\n'
    h += '<a href="<METRIC_PLOT>"><img src="<METRIC_PLOT>" height="300"></a>\n'
    h += '<br>\n'
    h += '<a href="{}"><img src="{}" height="300"></a>\n'.format(precisionRecallCurvePath, precisionRecallCurvePath)


    for i, partName in enumerate(partNames):
        h += '<h2 id="{}">{}</h2>\n'.format('link_{}'.format(partName), partName.capitalize())
        
        h += '<h3 id="{}">Metrics</h3>\n'.format('link_{}_{}'.format(partName, 'metric'))
        h += '<p>Number of objects: {}</p>\n'.format(numPosDict[partName])
        h += '<p>True positives: {}</p>\n'.format(numTruePosDict[partName])
        h += '<p>False positives: {}</p>\n'.format(numFalsePosDict[partName])
        h += '<p>False negatives: {}</p>\n'.format(numFalseNegDict[partName])
        h += '<p>Precision: {:.3f}</p>\n'.format(precisionDict[partName])
        h += '<p>Recall: {:.3f}</p>\n'.format(recallDict[partName])

        metricMatrix[i, 0] = partName
        metricMatrix[i, 1] = numPosDict[partName]
        metricMatrix[i, 2] = numTruePosDict[partName]
        metricMatrix[i, 3] = numFalsePosDict[partName]
        metricMatrix[i, 4] = numFalseNegDict[partName]
        metricMatrix[i, 5] = precisionDict[partName]
        metricMatrix[i, 6] = recallDict[partName]

        h += '<h3 id="{}">Images with false positives</h3>\n'.format('link_{}_{}'.format(partName, 'fp'))
        for p in fpImagePaths[partName]:
            h += '<a href="{}"><img src="{}"></a>\n'.format(p, p)

        h += '<h3 id="{}">Images with false negatives</h3>\n'.format('link_{}_{}'.format(partName, 'fn'))
        for p in fnImagePaths[partName]:
            h += '<a href="{}"><img src="{}"></a>\n'.format(p, p)

        h += '<h3 id="{}">Images all correct</h3>\n'.format('link_{}_{}'.format(partName, 'correct'))
        for p in correctImagesPath[partName]:
            h += '<a href="{}"><img src="{}"></a>\n'.format(p, p)


    h += '</body>\n'
    h += '</html>\n'

    metricDf = pd.DataFrame(metricMatrix, columns=['part', '# objects', 'true pos.', 'false pos.', 'false neg.', 'precision', 'recall'])
    metricDf.precision = metricDf.precision.apply(lambda x: np.round(x,3))
    metricDf.recall = metricDf.recall.apply(lambda x: np.round(x,3))
    # metricDf.part = metricDf.part.apply(lambda x: x.capitalize())
    metricMatrixHtml = metricDf.to_html(index=False)
    h = h.replace('<METRIC_MATRIX>', metricMatrixHtml)
    metricPlotPath = plotMetricMatrix(metricDf, exportDirRoot)
    h = h.replace('<METRIC_PLOT>', str(metricPlotPath.name))

    exportPath = exportDirRoot.joinpath('evalSummary.html') 
    exportPath.write_text(h)
    print('\nexported file: {}'.format(str(exportPath)))

    exportPath = exportDirRoot.joinpath('detectionResultMatrix.csv') 
    metricDf.to_csv(exportPath, index=False)
    print('\nexported file: {}'.format(str(exportPath)))


def plotMetricMatrix(df, exportDir):

    xTicks = list(range(len(df)))
    xTickLabels = list(df.part)
    precision = list(df.precision)
    recall = list(df.recall)
    majorTicks = np.arange(-0.1, 1.1, 0.1)

    fig = plt.figure(figsize=(12, 6))
    plt.plot(xTicks, precision, 'o--', label='precision')
    plt.plot(xTicks, recall, 'o--', label='recall')
    plt.xticks(xTicks, xTickLabels, rotation=30, ha='right')
    plt.xlabel('vehicle parts')
    plt.ylabel('score')
    ax = plt.gca()
    ax.set_yticks(majorTicks)
    plt.grid()
    plt.ylim((-0.05, 1.05)) 
    plt.legend()

    plt.tight_layout()

    path = exportDir.joinpath('detectionResultMatrixPlot.svg')
    plt.savefig(str(path))
    plt.close()

    return path


def getOverallResults(df, partName):
    partNames = [partName]
    numberOfImages = len(df)

    precDict = {name: None for name in partNames}
    recallDict = {name: None for name in partNames}
    numPosDict = {name: None for name in partNames}
    numTruePosDict = {name: None for name in partNames}
    numFalsePosDict = {name: None for name in partNames}
    numFalseNegDict = {name: None for name in partNames}
    for partName in partNames:
        p = getNumPositives(df, partName)
        tp = getNumTruePositives(df, partName)
        fn = getNumFalseNegatives(df, partName)
        # assert p == tp + fn
        fp = getNumFalsePositives(df, partName)
        prec = tp/(tp+fp)
        recall = tp/(tp+fn)
        precDict[partName] = prec
        recallDict[partName] = recall
        numPosDict[partName] = p
        numTruePosDict[partName] = tp
        numFalsePosDict[partName] = fp
        numFalseNegDict[partName] = fn

    return precDict, recallDict, numPosDict, numTruePosDict, numFalsePosDict, numFalseNegDict, numberOfImages


def generateImages(args, df, imageDict, categoryName, outputDir, confidenceThreshold, iouThreshold):
    now = datetime.now()
    date_time = now.strftime("%m_%d_%Y_%H_%M")
    exportDirRoot = Path(outputDir).joinpath('{}_eval_confThrs{:.2f}_iouThrs{:.2f}_{}'.format(args.name,confidenceThreshold, iouThreshold,date_time))

    correctImagesPath = {categoryName: []}
    fpImagePaths = {categoryName: []}
    fnImagePaths = {categoryName: []}

    # clean export dirs
    exportDirCorrect = exportDirRoot.joinpath('correctImg_{}'.format(categoryName))
    exportDirFp = exportDirRoot.joinpath('falsePosImg_{}'.format(categoryName))
    exportDirFn = exportDirRoot.joinpath('falseNegImg_{}'.format(categoryName))
    for dir_ in [exportDirCorrect, exportDirFp, exportDirFn]:
        if dir_.exists():
            print('\n export folder found and will be deleted: {}'.format(str(dir_)))
            shutil.rmtree(str(dir_))

    # get data frames
    df_fp = getFalsePositives(df, categoryName) # false positives
    df_fn = getFalseNegatives(df, categoryName) # false negatives
    df_correct = getTruePositives(df, categoryName) # all corrects

    # save images
    for index, s in tqdm(df.iterrows(), total=len(df), desc='save result images for part {}'.format(categoryName)):
        img = imageDict[s.imageId].copy()
        gt = s.groundTruth[categoryName]
        pred = s.prediction[categoryName]
        tp = s.truePositives[categoryName]

        # ground truth rectangle drawing
        for gt_i in gt:
            color = (255, 255, 0)
            img = drawRectangle(img, gt_i, color)
        
        # prediction rectangle drawing
        for i, pred_i in enumerate(pred):
            if tp[i, 0] == True:
                color = (0, 255, 0)
            else:
                color = (0, 0, 255)
            img = drawRectangle(img, pred_i, color, pred_i[4])
            
        # save image and path
        if index in df_correct.index:
            saveImage(exportDirCorrect, correctImagesPath[categoryName], s.imageId, img)
        if index in df_fp.index:
            saveImage(exportDirFp, fpImagePaths[categoryName], s.imageId, img)
        if index in df_fn.index:
            saveImage(exportDirFn, fnImagePaths[categoryName], s.imageId, img)
    
    return correctImagesPath, fpImagePaths, fnImagePaths, exportDirRoot


def saveImage(exportDir, imagePath, imagepathOrig, img):
    exportFile = Path(imagepathOrig).stem + '_res.jpg'
    exportPath = exportDir.joinpath(exportFile)
    if not os.path.exists(exportDir):
        os.makedirs(exportDir)
    imagePath.append(str(Path(*exportPath.parts[-2:])))
    cv2.imwrite(str(exportPath), img)

def drawRectangle(img, output, color, score=None):
    start = [int(np.round(v)) for v in output[:2]]
    end = [int(np.round(v1+v2)) for v1, v2 in zip(output[2:4], output[:2])]
    img = cv2.rectangle(img, start, end, color, thickness=1)

    if not score is None:
        scoreString = '{:.2f}'.format(score)
        img = cv2.putText(img, scoreString, (output[:2] - [0, 2]).astype(int), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)

    return img

def getNumPositives(df, part):

    return int(df['groundTruth'].apply(lambda x: len(x[part])).sum())

def getNumTruePositives(df, part):

    return int(df['truePositives'].apply(lambda x: x[part].sum()).sum())

def getNumFalsePositives(df, part):

    return int(df['falsePositives'].apply(lambda x: x[part].sum()).sum())

def getNumFalseNegatives(df, part):

    return int(df['falseNegatives'].apply(lambda x: x[part].sum()).sum())

def getFalsePositives(df, part):

    return df[df['falsePositives'].apply(lambda x: x[part].sum()) > 0]

def getFalseNegatives(df, part):

    return df[df['falseNegatives'].apply(lambda x: x[part].sum()) > 0]

def getTruePositives(df, part):
    noFalseNegatives = df['falseNegatives'].apply(lambda x: x[part].sum() == 0)
    noFalsePositives = df['falsePositives'].apply(lambda x: x[part].sum() == 0)

    return df[(noFalseNegatives & noFalsePositives)]


def createData(raw_dataset, tflite_model, categoryName, scoreThreshold, iouThreshold, debugMode):
    numOfImages = sum([1 for r in raw_dataset])
    seriesList = []
    imageDict = {}
    pbar = tqdm(raw_dataset, total=numOfImages)
    for i, raw_record in enumerate(pbar):
        resized_image,ground_truth_bbox,uuid = get_image_and_features(raw_record,tflite_model.input_image_size)
        predict_bbox, detected_scores = tflite_model.predict(resized_image)
        predict_bbox, detected_scores = filterLowConfidence(predict_bbox, detected_scores, scoreThreshold)
        pred = convertToYoloFormat(predict_bbox, detected_scores)
        label = convertToYoloFormat(ground_truth_bbox)

        if len(pred) == 0:
            pred = np.array([]).reshape((0, 5))

        predDict = {categoryName: pred}
        labelDict = {categoryName: label}

        tp, fp, fn = getMetrics(copy.deepcopy(predDict), copy.deepcopy(labelDict), [categoryName], iouThreshold)

        d = {
            'imageId': uuid,
            'groundTruth': labelDict,
            'prediction': predDict,
            'truePositives': tp,
            'falsePositives': fp,
            'falseNegatives': fn,
        }
        s = pd.Series(d)
        seriesList.append(s)
        img = np.array(resized_image)[..., (2,1,0)] # rgb to bgr
        imageDict[uuid] = img

        if debugMode:
            if i == 99:
                break
    
    df = pd.concat(seriesList, axis=1).T

    if not len(df) == len(imageDict):
        raise AssertionError('not same length!')

    return df, imageDict


def getMetrics(predDict, labelDict, partNames, iouThrs):
    tpd = {}
    fpd = {}
    fnd = {}

    for partName in partNames:
        bboxLabel = labelDict[partName]
        bboxLabel[:, 2] = bboxLabel[:, 0] + bboxLabel[:, 2]
        bboxLabel[:, 3] = bboxLabel[:, 1] + bboxLabel[:, 3]

        bboxPred = predDict[partName][:, :4]
        bboxPred[:, 2] = bboxPred[:, 0] + bboxPred[:, 2]
        bboxPred[:, 3] = bboxPred[:, 1] + bboxPred[:, 3]

        boxIou = box_iou(bboxPred, bboxLabel)
        if boxIou.size > 0:
            boxIouMaxPerLabel = copy.deepcopy(boxIou)
            boxIouNotMaxPerLabel = copy.deepcopy(boxIou)
            maxIouPerLabel = boxIou == boxIou.max(axis=0)
            boxIouMaxPerLabel[np.where(~maxIouPerLabel)] = 0
            boxIouNotMaxPerLabel[np.where(maxIouPerLabel)] = 0
            tpb =  (boxIouMaxPerLabel.max(axis=1) >= iouThrs).reshape(-1, 1) # find best match per ground truth. (reshape: shape of prediction)
            tpn = (boxIouNotMaxPerLabel.max(axis=1) >= iouThrs).reshape(-1, 1) # finds matches without best match per ground truth. (reshape: shape of prediction)
            tpn[(tpb==True) & (tpn==True)] = False # set tpn detection as false if this detection is a tpb (occures if one detection has high iou with two ground truth boxes)
            
            tp = tpb & ~tpn #true positives 
            fp = (1 - tp).astype(np.bool) # shape of prediction
            fn = boxIou.max(axis=0) < iouThrs  # shape of label

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

        # assert (tp.sum() + fn.sum()) == len(bboxLabel)

        tpd[partName] = tp
        fpd[partName] = fp
        fnd[partName] = fn

    return tpd, fpd, fnd


def box_iou(box1, box2):
    """
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
    inter = (np.minimum(box1[:, None, 2:], box2[:, 2:]) - np.maximum(box1[:, None, :2], box2[:, :2])).clip(0, None).prod(2)
    return inter / (area1[:, None] + area2 - inter)  # iou = inter / (area1 + area2 - inter)


def filterLowConfidence(predBbox, predScores, threshold):

    predBboxNew = []
    predScoresNew = []
    for box, score in zip(predBbox, predScores):
        if score >= threshold:
            predBboxNew.append(box)
            predScoresNew.append(score)

    return predBboxNew, predScoresNew


def convertToYoloFormat(bboxs, scores=None):

    yoloFormat = []
    if scores is None:
        scores = [None for _ in bboxs]
    for box, score in zip(bboxs, scores):
        # [x0, y0, x1, y1] -> [x0, y0, w, h]
        x0, y0, x1, y1 = box
        w = x1 - x0
        h = y1 - y0
        boxNew = [x0, y0, w, h]

        if score is not None:
            newFormat = boxNew + [score]
        else:
            newFormat = boxNew

        yoloFormat.append(newFormat)

    return np.array(yoloFormat, dtype=np.float32)



def get_image_and_features(raw_dataset, wanted_image_size):
    feature_description = {
        'image/filename': tf.io.FixedLenFeature([], tf.string),
        'image/encoded': tf.io.FixedLenFeature([],tf.string),
        'image/object/bbox/xmin':tf.io.FixedLenSequenceFeature([], tf.float32,allow_missing=True),
        'image/object/bbox/xmax':tf.io.FixedLenSequenceFeature([],tf.float32,allow_missing=True),
        'image/object/bbox/ymin':tf.io.FixedLenSequenceFeature([],tf.float32,allow_missing=True),
        'image/object/bbox/ymax':tf.io.FixedLenSequenceFeature([],tf.float32,allow_missing=True),
    }

    record = tf.io.parse_single_example(raw_dataset, feature_description)
    x_min_rel = record['image/object/bbox/xmin']
    y_min_rel = record['image/object/bbox/ymin']
    x_max_rel = record['image/object/bbox/xmax']
    y_max_rel = record['image/object/bbox/ymax']
    identifier = record['image/filename'].numpy().decode("utf-8") 
    image = Image.open(io.BytesIO(record['image/encoded'].numpy()))
    
    resized_image = image.resize(wanted_image_size[1:3])
    width, height = resized_image.size
    x_min_abs = x_min_rel.numpy()*width
    y_min_abs = y_min_rel.numpy()*height
    x_max_abs = x_max_rel.numpy()*width
    y_max_abs = y_max_rel.numpy()*height

    annotations = []
    for bboxIndex in range(len(x_min_abs)):
        annotations.append([int(x_min_abs[bboxIndex]),int(y_min_abs[bboxIndex]),int(x_max_abs[bboxIndex]),int(y_max_abs[bboxIndex])])
    return resized_image, annotations,identifier.split('.')[0]



class InferenceTflite(object):
    def __init__(self,path_to_model:str):
        self.interpreter = tf.lite.Interpreter(model_path=path_to_model)
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
        detected_class_labels =  np.squeeze(self.interpreter.get_tensor(self.model_output_details[1]['index']).astype(int))
        detected_boxes =  np.squeeze(self.interpreter.get_tensor( self.model_output_details[0]['index']))
        detected_scores =  np.squeeze(self.interpreter.get_tensor( self.model_output_details[2]['index']))

        detected_boxes = detected_boxes[:, (1,0,3,2)]
        detected_boxes[:] *= width,hight,width,hight
        personIndex = 0
        return detected_boxes[detected_class_labels == personIndex].tolist(), detected_scores[detected_class_labels == personIndex].tolist()


if __name__ == '__main__':
    main()

