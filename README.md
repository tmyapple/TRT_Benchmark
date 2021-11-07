# TRT_Benchmark
Run inference of ResNet50(Keras) with TensorRT on A4000

<hr>
  
## Build The Docker Image:
Make sure you have docker and nvidia-docker installed first.
```
nvidia-docker build -t tamirt/trt_benchmark:1.0 .
```

<hr>
  
## Start Docker Image:   

```
docker run -it --name <docker-name> --rm --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 tamirt/trt_benchmark:1.0
```  

<hr>
  
## How to Run the Benchamrk?  
* From inside the git repo directory run the following command:
```
./convert_resnet50.sh <gpu-number>
```  
The script will convert resnet50 to TRT with FP32/FP16/INT8, and will run a sanity on one image.  
* Run the benchmark on the TRT models:  
```
CUDA_VISIBLE_DEVICES=0 python benchmark_trt.py --input-model resnet50_saved_model_TFTRT_FP32
CUDA_VISIBLE_DEVICES=0 python benchmark_trt.py --input-model resnet50_saved_model_TFTRT_FP16
CUDA_VISIBLE_DEVICES=0 python benchmark_trt.py --input-model resnet50_saved_model_TFTRT_INT8
```  
You can change the test batch-size by using the argument --batch-size

