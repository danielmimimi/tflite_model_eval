import argparse
from dataset_reader import ImageAnnotationReader

from inference.inference_tflite_imx8_ssd_npu_model import InferenceTfliteImx8SsdCpumodel, InferenceTfliteImx8SsdNpumodel
from result_generation import ResultGeneration

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

parser.add_argument('-m','--model_path',default="/tflite_model_eval/test/model/efficient_det_1_320_320_002.tflite")
parser.add_argument('-t','--test_images_path',default="/tflite_model_eval/test/data/record")
parser.add_argument('-d','--delegate_path',default="/tflite_model_eval/test/data/record")
parser.add_argument('-o','--save_only', type=str2bool, nargs='?',const=True, default=True)
parser.add_argument('-f','--filter_low_score',default=0.0)
parser.add_argument('-i','--iou_treshold',default=0.5)

if __name__ == '__main__':
    args = parser.parse_args()

    if not args.delegate_path:
        model = InferenceTfliteImx8SsdCpumodel(args.model_path)
    elif args.delegate_path == "/usr/lib/libvx_delegate.so":
        model = InferenceTfliteImx8SsdNpumodel(args.model_path)
    else:
        raise Exception("No possible model selected")
    
    dataset_reader = ImageAnnotationReader(args.test_images_path,model.get_image_size())
    dataset_reader.load_dataset()
    
    result_generator = ResultGeneration(dataset_reader,model,"","",args.filter_low_score,args.iou_treshold,args.save_only)
    summarized_results, images_analyzed = result_generator.start_evaluation()
    
    