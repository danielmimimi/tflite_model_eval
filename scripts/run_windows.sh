
TARGET=CPU_WIN

python /tflite_model_eval/evaluation_script_win.py \
    --model_path=/tflite_model_eval/test/model/ssd_lite_trial_009/ssdlite_1_320_320_COCO_trial_009.tflite \
    --output_name=$TARGET \
    --filter_low_score=0.1 \
    --save_only=1


python /tflite_model_eval/evaluation_script_win.py \
    --model_path=/tflite_model_eval/test/model/efficient_det_0_trial_002/efficient_det_1_320_320_002.tflite \
    --output_name=$TARGET \
    --filter_low_score=0.1 \
    --save_only=1


# Tensorflow version probably wrong
# python /tflite_model_eval/evaluation_script_win.py \
#     --model_path=/tflite_model_eval/test/model/nxp_yolov5/yolov5s-int8.tflite \
#     --output_name=$TARGET \
#     --filter_low_score=0.1 \
#     --save_only=1

# python /tflite_model_eval/evaluation_script_win.py \
#     --model_path=/tflite_model_eval/test/model/odysseas_yolov5/25k_theo_icarus_small_low-int8.tflite \
#     --output_name=$TARGET \
#     --filter_low_score=0.1 \
#     --save_only=1