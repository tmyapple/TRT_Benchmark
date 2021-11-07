TFTRT_FP32="resnet50_saved_model_TFTRT_FP32"
TFTRT_FP16="resnet50_saved_model_TFTRT_FP16"
TFTRT_INT8="resnet50_saved_model_TFTRT_INT8"
pip install pillow
rm -rf ./data
mkdir -p ./data
wget  -O ./data/img0.JPG "https://d17fnq9dkz9hgj.cloudfront.net/breed-uploads/2018/08/siberian-husky-detail.jpg?bust=1535566590&width=630"
wget  -O ./data/img1.JPG "https://www.hakaimagazine.com/wp-content/uploads/header-gulf-birds.jpg"
wget  -O ./data/img2.JPG "https://www.artis.nl/media/filer_public_thumbnails/filer_public/00/f1/00f1b6db-fbed-4fef-9ab0-84e944ff11f8/chimpansee_amber_r_1920x1080.jpg__1920x1080_q85_subject_location-923%2C365_subsampling-2.jpg"
wget  -O ./data/img3.JPG "https://www.familyhandyman.com/wp-content/uploads/2018/09/How-to-Avoid-Snakes-Slithering-Up-Your-Toilet-shutterstock_780480850.jpg"

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
