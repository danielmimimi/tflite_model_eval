TARGET=TPU_CORAL

DELEGATE_PATH=libedgetpu.so.1
echo "Start Inferencing"

echo "ssdlite_1_320_320_COCO_trial_009"
python3 /media/usb-icarus/inference_imx8_script.py \
    --model_path=/media/usb-icarus/test/model/ssd_lite_trial_009/ssdlite_1_320_320_COCO_trial_009.tflite \
    --output_name=$TARGET \
    --filter_low_score=0.1 \
    --save_only=1 \
    --delegate_path=$DELEGATE_PATH


echo "efficient_det_0_trial_003"
python3 /media/usb-icarus/inference_imx8_script.py \
    --model_path=/media/usb-icarus/test/model/efficient_det_0_trial_003/efficient_det_1_320_320_003.tflite \
    --output_name=$TARGET \
    --filter_low_score=0.1 \
    --save_only=1 \
    --delegate_path=$DELEGATE_PATH