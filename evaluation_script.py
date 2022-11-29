import argparse

from dataset_reader import AbstractDatasetReader, ImageAnnotationReader, TfRecordReader
from evaluation import Evaluation
from inference_tflite_model import InferenceTflitemodel

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser(
                    prog = 'ProgramName',
                    description = 'What the program does',
                    epilog = 'Text at the bottom of help')

parser.add_argument('-m','--model_path',default="/tflite_model_eval/test/model/ssdlite_1_320_320_COCO_trial_009.tflite")
parser.add_argument('-t','--test_images_path',default="/tflite_model_eval/test/data/record")
parser.add_argument('-s','--save_path',default="/media/usb-icarus/coco_icarus_eval/eval_andreas/saved_image_path")
parser.add_argument('-l','--only_load', type=str2bool, nargs='?',const=True, default=False)
parser.add_argument('-d','--test_data_type',default="regular")
parser.add_argument('-n','--output_name',default="testing")
parser.add_argument('-f','--filter_low_score',default=0.0)
args = parser.parse_args()

if __name__ == '__main__':
    args = parser.parse_args()
    
    model = InferenceTflitemodel(args.model_path)
    
    # LOAD DATASET
    if args.test_data_type == "regular" : 
        dataset_reader = ImageAnnotationReader(args.test_images_path,model.get_image_size())
    else:
        dataset_reader = TfRecordReader(args.test_images_path,model.get_image_size())
    dataset_reader.load_dataset()
    
    evaluation = Evaluation(dataset_reader,model,args.only_load,args.filter_low_score)
    evaluation.start_evaluation()
    
    
    
    
    
    