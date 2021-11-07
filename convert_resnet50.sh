TFTRT_FP32="resnet50_saved_model_TFTRT_FP32"
TFTRT_FP16="resnet50_saved_model_TFTRT_FP16"
TFTRT_INT8="resnet50_saved_model_TFTRT_INT8"


CUDA_VISIBLE_DEVICES=$1 python get_resnet_50.py
saved_model_cli show --all --dir resnet50_saved_model

CUDA_VISIBLE_DEVICES=$1 python convert_model.py --mode FP32
saved_model_cli show --dir ${TFTRT_FP32} --tag_set serve --signature_def "serving_default"

CUDA_VISIBLE_DEVICES=$1 python convert_model.py --mode FP16
saved_model_cli show --dir ${TFTRT_FP16} --tag_set serve --signature_def "serving_default"

CUDA_VISIBLE_DEVICES=$1 python convert_model.py --mode INT8
saved_model_cli show --dir ${TFTRT_INT8} --tag_set serve --signature_def "serving_default"

echo "===========   RUN SANITY ON FP32   ==========="
CUDA_VISIBLE_DEVICES=$1 python benchmark_trt_fp.py --input-model ${TFTRT_FP32} --predict

echo "===========   RUN SANITY ON FP16   ==========="
CUDA_VISIBLE_DEVICES=$1 python benchmark_trt_fp.py --input-model ${TFTRT_FP16} --predict

echo "===========   RUN SANITY ON IN8   ==========="
CUDA_VISIBLE_DEVICES=$1 python benchmark_trt_fp.py --input-model ${TFTRT_INT8} --predict
