import argparse
from pathlib import Path

from dataset_reader import AbstractDatasetReader, ImageAnnotationReader, TfRecordReader
from evaluation import Evaluation
from inference_tflite_model import InferenceTflitemodel
from datetime import datetime

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
# parser.add_argument('-l','--only_load', type=str2bool, nargs='?',const=True, default=False)
parser.add_argument('-l','--only_load_path',default="/tflite_model_eval/test/model/saves/ssdlite_1_320_320_COCO_trial_009_cpu.npy")
parser.add_argument('-d','--test_data_type',default="regular")
parser.add_argument('-n','--output_name',default="testing")
parser.add_argument('-f','--filter_low_score',default=0.0)
parser.add_argument('-i','--iou_treshold',default=0.5)
parser.add_argument('-o','--save_only', type=str2bool, nargs='?',const=True, default=False)

args = parser.parse_args()


def create_output_file_name(args):
    now = datetime.now()
    date_time = now.strftime("%m_%d_%Y_%H_%M")
    full_output_save_path = Path(args.save_path).joinpath('{}_evaluation_filtered_score_{:.2f}_iouThrs{:.2f}_{}'.format(args.output_name,args.filter_low_score, args.iou_treshold,date_time))
    return full_output_save_path

if __name__ == '__main__':
    args = parser.parse_args()
    
    model = InferenceTflitemodel(args.model_path)
    
    # LOAD DATASET
    if args.test_data_type == "regular" : 
        dataset_reader = ImageAnnotationReader(args.test_images_path,model.get_image_size())
    else:
        dataset_reader = TfRecordReader(args.test_images_path,model.get_image_size())
    dataset_reader.load_dataset()
    
    evaluation = Evaluation(dataset_reader,model,create_output_file_name(args),args.only_load_path,args.filter_low_score,args.iou_treshold,args.save_only)
    summarized_results, images_analyzed = evaluation.start_evaluation()
    
    evaluation.export_images_with_bounding_boxes(summarized_results, images_analyzed)
    
    
    
    
    
    
    