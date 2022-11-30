import argparse
from dataset_reader import ImageAnnotationReader
from inference.inference_tflite_imx8_ssd_model import InferenceTfliteImx8SsdNpumodel,InferenceTfliteImx8SsdCpumodel
from inference.inference_tflite_imx8_efficient_model import InferenceTfliteImx8EfficientCpumodel,InferenceTfliteImx8EfficientNpumodel

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

parser.add_argument('-m','--model_path',default="/media/usb-icarus/tflite_model_eval/test/model/efficient_det_0_trial_002/efficient_det_1_320_320_002.tflite")
parser.add_argument('-t','--test_images_path',default="/media/usb-icarus/tflite_model_eval/test/data/record")
parser.add_argument('-d','--delegate_path',default="/usr/lib/libvx_delegate.so")
parser.add_argument('-o','--save_only', type=str2bool, nargs='?',const=True, default=True)
parser.add_argument('-f','--filter_low_score',default=0.0)
parser.add_argument('-i','--iou_treshold',default=0.5)
parser.add_argument('-p','--peak_results',default="/media/usb-icarus/peak")

if __name__ == '__main__':
    args = parser.parse_args()
    if not args.delegate_path:
        if "efficient" in args.model_path:
            model = InferenceTfliteImx8EfficientCpumodel(args.model_path)
        elif "ssdlite" in args.model_path:
            model = InferenceTfliteImx8SsdCpumodel(args.model_path)
        else:
            raise Exception("No possible model selected")
    elif args.delegate_path == "/usr/lib/libvx_delegate.so":
        if "efficient" in args.model_path:
            model = InferenceTfliteImx8EfficientNpumodel(args.model_path)
        elif "ssdlite" in args.model_path:
            model = InferenceTfliteImx8SsdNpumodel(args.model_path)
        else:
            raise Exception("No possible model selected")
    else:
        raise Exception("No possible model selected")
    
    dataset_reader = ImageAnnotationReader(args.test_images_path,model.get_image_size())
    dataset_reader.load_dataset()
    
    result_generator = ResultGeneration(dataset_reader,model,"","",args.filter_low_score,args.iou_treshold,args.save_only,args.peak_results)
    summarized_results, images_analyzed = result_generator.start_evaluation()
    
    