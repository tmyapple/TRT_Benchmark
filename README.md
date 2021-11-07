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
