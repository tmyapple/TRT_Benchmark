from __future__ import absolute_import, division, print_function, unicode_literals
import argparse

import numpy as np

import tensorflow as tf
from tensorflow.python.compiler.tensorrt import trt_convert as trt
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions

def convert_to_trtfp(mode):
    if mode == "FP32":
        print('Converting to TF-TRT FP32...')
        conversion_params = trt.DEFAULT_TRT_CONVERSION_PARAMS._replace(
            precision_mode=trt.TrtPrecisionMode.FP32,
            max_workspace_size_bytes=8000000000)
    if mode == "FP16":
        print('Converting to TF-TRT FP16...')
        conversion_params = trt.DEFAULT_TRT_CONVERSION_PARAMS._replace(
            precision_mode=trt.TrtPrecisionMode.FP16,
            max_workspace_size_bytes=8000000000) 

    converter = trt.TrtGraphConverterV2(input_saved_model_dir='resnet50_saved_model',
                                        conversion_params=conversion_params)
    converter.convert()
    converter.save(output_saved_model_dir='resnet50_saved_model_TFTRT_{}'.format(mode))
    print('Done Converting to TF-TRT {}'.format(mode))


def convert_to_trtint():
    batch_size = 8
    batched_input = np.zeros((batch_size, 224, 224, 3), dtype=np.float32)

    for i in range(batch_size):
        img_path = './data/img%d.JPG' % (i % 4)
        img = image.load_img(img_path, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        batched_input[i, :] = x
    batched_input = tf.constant(batched_input)
    print('batched_input shape: ', batched_input.shape)

    print('Converting to TF-TRT INT8...')
    conversion_params = trt.DEFAULT_TRT_CONVERSION_PARAMS._replace(
        precision_mode=trt.TrtPrecisionMode.INT8, 
        max_workspace_size_bytes=8000000000, 
        use_calibration=True)
    converter = trt.TrtGraphConverterV2(
        input_saved_model_dir='resnet50_saved_model', 
        conversion_params=conversion_params)

    def calibration_input_fn():
        yield (batched_input, )
    
    converter.convert(calibration_input_fn=calibration_input_fn)
    converter.save(output_saved_model_dir='resnet50_saved_model_TFTRT_INT8')

    print('Done Converting to TF-TRT INT8')

def main():
    argparser = argparse.ArgumentParser("Convert Resnet 50 to lower bit representation model")
    argparser.add_argument("--mode", action='store', help="choose from [FP32, FP16, INT8]", type=str)

    args = argparser.parse_args()

    if args.mode in ["FP32", "FP16"]:
        convert_to_trtfp(args.mode)

    if args.mode == "INT8":
        convert_to_trtint()

if __name__ == "__main__":
    main()
