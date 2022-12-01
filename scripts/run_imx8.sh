TARGET=CPU_IMX8


echo "Start Inferencing"
echo "25k_theo_icarus_small_low"
python3 /media/usb-icarus/inference_imx8_script.py \
    --model_path=/media/usb-icarus/test/model/odysseas_yolov5/25k_theo_icarus_small_low-int8.tflite \
    --output_name=$TARGET \
    --filter_low_score=0.1 \
    --save_only=1

echo "yolov5s"
python3 /media/usb-icarus/inference_imx8_script.py \
    --model_path=/media/usb-icarus/test/model/nxp_yolov5/yolov5s-int8.tflite \
    --output_name=$TARGET \
    --filter_low_score=0.1 \
    --save_only=1

echo "efficient_det_0_trial_002"
python3 /media/usb-icarus/inference_imx8_script.py \
    --model_path=/media/usb-icarus/test/model/efficient_det_0_trial_002/efficient_det_1_320_320_002.tflite \
    --output_name=$TARGET \
    --filter_low_score=0.1 \
    --save_only=1

echo "efficient_det_0_trial_003"
python3 /media/usb-icarus/inference_imx8_script.py \
    --model_path=/media/usb-icarus/test/model/efficient_det_0_trial_003/efficient_det_1_320_320_003.tflite \
    --output_name=$TARGET \
    --filter_low_score=0.1 \
    --save_only=1


echo "ssdlite_1_320_320_COCO_trial_009"
python3 /media/usb-icarus/inference_imx8_script.py \
    --model_path=/media/usb-icarus/test/model/ssd_lite_trial_009/ssdlite_1_320_320_COCO_trial_009.tflite \
    --output_name=$TARGET \
    --filter_low_score=0.1 \
    --save_only=1 