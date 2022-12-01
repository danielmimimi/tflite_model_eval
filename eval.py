from pathlib import Path
import pandas as pd
import shutil
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt


class Evaluation(object):
    def __init__(self,summarized_results, images_analyzed,save_path:Path,filter_low_score:float,iou_threshold:float) -> None:
        self.summarized_results = self._convert_results(summarized_results)
        self.images_analyzed = images_analyzed
        self.save_path = save_path
        self.filter_low_score = filter_low_score
        self.iou_threshold = iou_threshold
        self.categoryName = "person"
        
    def export(self):
        correctImagesPath, fpImagePaths, fnImagePaths, _ = self.export_images_with_bounding_boxes()
        preccision_recall_curve_path = self.export_precision_recall_curve()
        precisionDict, recallDict, numPosDict, numTruePosDict, numFalsePosDict, numFalseNegDict, numberOfImages = self.get_overall_results()
        
        partNames = [self.categoryName]

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
        h += '<p>Confidence Threshold: {:.2f}</p>\n'.format(self.filter_low_score)
        h += '<p>Intersection over Union Threshold: {:.2f}</p>\n'.format(self.iou_threshold)

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
        h += '<a href="{}"><img src="{}" height="300"></a>\n'.format(preccision_recall_curve_path, preccision_recall_curve_path)


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
        metricPlotPath = self._plotMetricMatrix(metricDf, self.save_path)
        h = h.replace('<METRIC_PLOT>', str(metricPlotPath.name))

        exportPath = self.save_path.joinpath('evalSummary.html') 
        exportPath.write_text(h)
        print('\nexported file: {}'.format(str(exportPath)))

        exportPath = self.save_path.joinpath('detectionResultMatrix.csv') 
        metricDf.to_csv(exportPath, index=False)
        print('\nexported file: {}'.format(str(exportPath)))

    
    def get_overall_results(self):
        partNames = [self.categoryName]
        numberOfImages = len(self.summarized_results)

        precDict = {name: None for name in partNames}
        recallDict = {name: None for name in partNames}
        numPosDict = {name: None for name in partNames}
        numTruePosDict = {name: None for name in partNames}
        numFalsePosDict = {name: None for name in partNames}
        numFalseNegDict = {name: None for name in partNames}
        for partName in partNames:
            p = self._getNumPositives(self.summarized_results, partName)
            tp = self._getNumTruePositives(self.summarized_results, partName)
            fn = self._getNumFalseNegatives(self.summarized_results, partName)
            # assert p == tp + fn
            fp = self._getNumFalsePositives(self.summarized_results, partName)
            prec = tp/(tp+fp)
            recall = tp/(tp+fn)
            precDict[partName] = prec
            recallDict[partName] = recall
            numPosDict[partName] = p
            numTruePosDict[partName] = tp
            numFalsePosDict[partName] = fp
            numFalseNegDict[partName] = fn

        return precDict, recallDict, numPosDict, numTruePosDict, numFalsePosDict, numFalseNegDict, numberOfImages

  
    def export_precision_recall_curve(self):
        scores = []
        matches = []
        precisions = [1.0]
        recalls = [0.0]
        tp = 0
        fp = 0

        numOfGroundTruthObjects = sum([len(ds.groundTruth[self.categoryName]) for index, ds in self.summarized_results.iterrows()])

        for index, ds in self.summarized_results.iterrows():
            labelBoxs = ds.groundTruth[self.categoryName]
            prediction = ds.prediction[self.categoryName]
            predicitonScores = prediction[:, 4]
            predictionMatches = ds.truePositives[self.categoryName].ravel()

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
        # Save values for summary
        np.save(self.save_path.joinpath('recalls.npy'),recalls)
        np.save(self.save_path.joinpath('precisions.npy'),precisions)
        # Save plot
        path = self.save_path.joinpath('precisionRecallCurve.svg')
        plt.savefig(str(path))
        plt.close()

        imagePath = str(path.name)
        return imagePath
    
    def export_images_with_bounding_boxes(self):
        correctImagesPath = {self.categoryName: []}
        fpImagePaths = {self.categoryName: []}
        fnImagePaths = {self.categoryName: []}
        self.exportDirCorrect = self.save_path.joinpath('correctImg_{}'.format(self.categoryName))
        self.exportDirFp = self.save_path.joinpath('falsePosImg_{}'.format(self.categoryName))
        self.exportDirFn = self.save_path.joinpath('falseNegImg_{}'.format(self.categoryName))
        for dir_ in [self.exportDirCorrect, self.exportDirFp, self.exportDirFn]:
            if dir_.exists():
                print('\n export folder found and will be deleted: {}'.format(str(dir_)))
                shutil.rmtree(str(dir_))
                
            # get data frames
        df_fp = self._getFalsePositives(self.summarized_results, self.categoryName) # false positives
        df_fn = self._getFalseNegatives(self.summarized_results, self.categoryName) # false negatives
        df_correct = self._getTruePositives(self.summarized_results, self.categoryName) # all corrects

        # save images
        for index, s in self.summarized_results.iterrows():
            if ".jpg" in s.imageId:
                img = self.images_analyzed[s.imageId].copy()
            else:
                img = self.images_analyzed[s.imageId+".jpg"].copy()
            gt = s.groundTruth[self.categoryName]
            pred = s.prediction[self.categoryName]
            tp = s.truePositives[self.categoryName]

            # ground truth rectangle drawing
            for gt_i in gt:
                color = (255, 255, 0)
                img = self._drawRectangle(img, gt_i, color)
            
            # prediction rectangle drawing
            for i, pred_i in enumerate(pred):
                if tp[i, 0] == True:
                    color = (0, 255, 0)
                else:
                    color = (0, 0, 255)
                img = self._drawRectangle(img, pred_i, color, pred_i[4])
                
            # save image and path
            if index in df_correct.index:
                self._saveImage(self.exportDirCorrect, correctImagesPath[self.categoryName], s.imageId, img)
            if index in df_fp.index:
                self._saveImage(self.exportDirFp, fpImagePaths[self.categoryName], s.imageId, img)
            if index in df_fn.index:
                self._saveImage(self.exportDirFn, fnImagePaths[self.categoryName], s.imageId, img)
        
        return correctImagesPath, fpImagePaths, fnImagePaths, self.categoryName
    
    
    def _saveImage(self,exportDir, imagePath, imagepathOrig, img):
        exportFile = Path(imagepathOrig).stem + '_res.jpg'
        exportPath = exportDir.joinpath(exportFile)
        if not os.path.exists(exportDir):
            os.makedirs(exportDir)
        imagePath.append(str(Path(*exportPath.parts[-2:])))
        cv2.imwrite(str(exportPath), img)

    
    def _drawRectangle(self,img, output, color, score=None):
        start = [int(np.round(v)) for v in output[:2]]
        end = [int(np.round(v)) for v in output[2:4]]
        #end = [int(np.round(v1+v2)) for v1, v2 in zip(output[2:4], output[:2])]
        img = cv2.rectangle(img, start, end, color, thickness=1)

        if not score is None:
            scoreString = '{:.2f}'.format(score)
            img = cv2.putText(img, scoreString, (output[:2] - [0, 2]).astype(int), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
        return img

    def _convert_results(self,summarized_results):
        summarized = []
        for result in summarized_results:
            summarized.append(pd.Series(result))
        return pd.concat(summarized,axis=1).T
                
    def _getNumPositives(self,df, part):
        return int(df['groundTruth'].apply(lambda x: len(x[part])).sum())
                   
    def _getNumTruePositives(self,df, part):
        return int(df['truePositives'].apply(lambda x: x[part].sum()).sum())

    def _getNumFalsePositives(self,df, part):
        return int(df['falsePositives'].apply(lambda x: x[part].sum()).sum())

    def _getNumFalseNegatives(self,df, part):
        return int(df['falseNegatives'].apply(lambda x: x[part].sum()).sum())

    def _getFalsePositives(self,df, part):
        return df[df['falsePositives'].apply(lambda x: x[part].sum()) > 0]

    def _getFalseNegatives(self,df, part):
        return df[df['falseNegatives'].apply(lambda x: x[part].sum()) > 0]

    def _getTruePositives(self,df, part):
        noFalseNegatives = df['falseNegatives'].apply(lambda x: x[part].sum() == 0)
        noFalsePositives = df['falsePositives'].apply(lambda x: x[part].sum() == 0)
        return df[(noFalseNegatives & noFalsePositives)]
    
    def _plotMetricMatrix(self,df, exportDir):

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