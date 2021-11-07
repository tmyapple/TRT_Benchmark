FROM nvcr.io/nvidia/tensorflow:21.10-tf2-py3

RUN apt-get update && apt-get install -y wget git
RUN pip install pillow matplotlib ipython

RUN git clone https://github.com/tmyapple/TRT_Benchmark.git

RUN cd TRT_Benchmark && /bin/bash ./download_sample_images.sh

