IOU_THRESHOLD=0.5
SCORE_THRESHOLD=0.1

CPU_WIN_TARGET=CPU_WIN
CPU_IMX8_TARGET=CPU_IMX8
NPU_IMX8_TARGET=NPU_IMX8

echo "EVAL IMX8 CPU"

python /tflite_model_eval/evaluation_script_win.py \
    --model_path=/tflite_model_eval/test/model/ssd_lite_trial_009/ssdlite_1_320_320_COCO_trial_009.tflite \
    --filter_low_score=$SCORE_THRESHOLD \
    --iou_treshold=$IOU_THRESHOLD \
    --output_name=$(($CPU_IMX8_TARGET+_SSDLITE)) \
    --only_load_path=/tflite_model_eval/test/model/ssd_lite_trial_009/saves/ssdlite_1_320_320_COCO_trial_009_CPU_IMX8_evaluation_filtered_score_0.10_iouThrs0.npy

python /tflite_model_eval/evaluation_script_win.py \
    --model_path=/tflite_model_eval/test/model/nxp_yolov5/yolov5s-int8.tflite \
    --filter_low_score=$SCORE_THRESHOLD \
    --iou_treshold=$IOU_THRESHOLD \
    --output_name=$CPU_IMX8_TARGET+_YOLO_NXP \
    --only_load_path=/tflite_model_eval/test/model/nxp_yolov5/saves/yolov5s-int8_CPU_IMX8_evaluation_filtered_score_0.10_iouThrs0.npy

python /tflite_model_eval/evaluation_script_win.py \
    --model_path=/tflite_model_eval/test/model/odysseas_yolov5/25k_theo_icarus_small_low-int8.tflite \
    --filter_low_score=$SCORE_THRESHOLD \
    --iou_treshold=$IOU_THRESHOLD \
    --output_name=$CPU_IMX8_TARGET+_YOLO_ODY \
    --only_load_path=/tflite_model_eval/test/model/odysseas_yolov5/saves/25k_theo_icarus_small_low-int8_CPU_IMX8_evaluation_filtered_score_0.10_iouThrs0.npy

python /tflite_model_eval/evaluation_script_win.py \
    --model_path=/tflite_model_eval/test/model/efficient_det_0_trial_003/efficient_det_1_320_320_003.tflite \
    --filter_low_score=$SCORE_THRESHOLD \
    --iou_treshold=$IOU_THRESHOLD \
    --output_name=$CPU_IMX8_TARGET+_EFFICIENT_3 \
    --only_load_path=/tflite_model_eval/test/model/efficient_det_0_trial_003/saves/efficient_det_1_320_320_003_CPU_IMX8_evaluation_filtered_score_0.10_iouThrs0.npy

python /tflite_model_eval/evaluation_script_win.py \
    --model_path=/tflite_model_eval/test/model/efficient_det_0_trial_002/efficient_det_1_320_320_002.tflite \
    --filter_low_score=$SCORE_THRESHOLD \
    --iou_treshold=$IOU_THRESHOLD \
    --output_name=$CPU_IMX8_TARGET+_EFFICIENT_2 \
    --only_load_path=/tflite_model_eval/test/model/efficient_det_0_trial_002/saves/efficient_det_1_320_320_002_CPU_IMX8_evaluation_filtered_score_0.10_iouThrs0.npy


echo "EVAL WIN"
python /tflite_model_eval/evaluation_script_win.py \
    --model_path=/tflite_model_eval/test/model/ssd_lite_trial_009/ssdlite_1_320_320_COCO_trial_009.tflite \
    --filter_low_score=$SCORE_THRESHOLD \
    --iou_treshold=$IOU_THRESHOLD \
    --output_name=$(($CPU_WIN_TARGET+_SSDLITE)) \
    --only_load_path=/tflite_model_eval/test/model/ssd_lite_trial_009/saves/ssdlite_1_320_320_COCO_trial_009_CPU_WIN_evaluation_filtered_score_0.10_iouThrs0.npy

python /tflite_model_eval/evaluation_script_win.py \
    --model_path=/tflite_model_eval/test/model/efficient_det_0_trial_002/efficient_det_1_320_320_002.tflite \
    --filter_low_score=$SCORE_THRESHOLD \
    --iou_treshold=$IOU_THRESHOLD \
    --output_name=$CPU_WIN_TARGET+_EFFICIENT_2 \
    --only_load_path=/tflite_model_eval/test/model/efficient_det_0_trial_002/saves/efficient_det_1_320_320_002_CPU_WIN_evaluation_filtered_score_0.10_iouThrs0.npy


echo "EVAL IMX8 NPU"

python /tflite_model_eval/evaluation_script_win.py \
    --model_path=/tflite_model_eval/test/model/ssd_lite_trial_009/ssdlite_1_320_320_COCO_trial_009.tflite \
    --filter_low_score=$SCORE_THRESHOLD \
    --iou_treshold=$IOU_THRESHOLD \
    --output_name=$NPU_IMX8_TARGET+_SSDLITE \
    --only_load_path=/tflite_model_eval/test/model/ssd_lite_trial_009/saves/ssdlite_1_320_320_COCO_trial_009_NPU_IMX8_evaluation_filtered_score_0.10_iouThrs0.npy

python /tflite_model_eval/evaluation_script_win.py \
    --model_path=/tflite_model_eval/test/model/nxp_yolov5/yolov5s-int8.tflite \
    --filter_low_score=$SCORE_THRESHOLD \
    --iou_treshold=$IOU_THRESHOLD \
    --output_name=$NPU_IMX8_TARGET+_YOLO_NXP  \
    --only_load_path=/tflite_model_eval/test/model/nxp_yolov5/saves/yolov5s-int8_NPU_IMX8_evaluation_filtered_score_0.10_iouThrs0.npy

python /tflite_model_eval/evaluation_script_win.py \
    --model_path=/tflite_model_eval/test/model/odysseas_yolov5/25k_theo_icarus_small_low-int8.tflite \
    --filter_low_score=$SCORE_THRESHOLD \
    --iou_treshold=$IOU_THRESHOLD \
    --output_name=$NPU_IMX8_TARGET+_YOLO_ODY \
    --only_load_path=/tflite_model_eval/test/model/odysseas_yolov5/saves/25k_theo_icarus_small_low-int8_NPU_IMX8_evaluation_filtered_score_0.10_iouThrs0.npy

python /tflite_model_eval/evaluation_script_win.py \
    --model_path=/tflite_model_eval/test/model/efficient_det_0_trial_003/efficient_det_1_320_320_003.tflite \
    --filter_low_score=$SCORE_THRESHOLD \
    --iou_treshold=$IOU_THRESHOLD \
    --output_name=$NPU_IMX8_TARGET+_EFFICIENT_3 \
    --only_load_path=/tflite_model_eval/test/model/efficient_det_0_trial_003/saves/efficient_det_1_320_320_003_NPU_IMX8_evaluation_filtered_score_0.10_iouThrs0.npy

python /tflite_model_eval/evaluation_script_win.py \
    --model_path=/tflite_model_eval/test/model/efficient_det_0_trial_002/efficient_det_1_320_320_002.tflite \
    --filter_low_score=$SCORE_THRESHOLD \
    --iou_treshold=$IOU_THRESHOLD \
    --output_name=$NPU_IMX8_TARGET+_EFFICIENT_2 \
    --only_load_path=/tflite_model_eval/test/model/efficient_det_0_trial_002/saves/efficient_det_1_320_320_002_NPU_IMX8_evaluation_filtered_score_0.10_iouThrs0.npy
