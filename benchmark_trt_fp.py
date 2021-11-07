from __future__ import absolute_import, division, print_function, unicode_literals

import argparse
import os
import time

import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.python.compiler.tensorrt import trt_convert as trt
from tensorflow.python.saved_model import tag_constants
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions


def get_batched_input(batch_size=8):
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
    return batched_input


def predict_tftrt(input_saved_model):
    """Runs prediction on a single image and shows the result.
    input_saved_model (string): Name of the input model stored in the current dir
    """
    img_path = './data/img0.JPG'  # Siberian_husky
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    x = tf.constant(x)
    
    saved_model_loaded = tf.saved_model.load(input_saved_model, tags=[tag_constants.SERVING])
    signature_keys = list(saved_model_loaded.signatures.keys())
    print(signature_keys)

    infer = saved_model_loaded.signatures['serving_default']
    print(infer.structured_outputs)

    labeling = infer(x)
    preds = labeling['predictions'].numpy()
    print('{} - Predicted: {}'.format(img_path, decode_predictions(preds, top=3)[0]))


def benchmark_tftrt(input_saved_model, batch_size=8):
    batched_input = get_batched_input(batch_size=batch_size)
    saved_model_loaded = tf.saved_model.load(input_saved_model, tags=[tag_constants.SERVING])
    infer = saved_model_loaded.signatures['serving_default']

    N_warmup_run = 50
    N_run = 500
    elapsed_time = []

    for i in range(N_warmup_run):
      labeling = infer(batched_input)

    for i in range(N_run):
      start_time = time.time()
      labeling = infer(batched_input)
      #prob = labeling['probs'].numpy()
      end_time = time.time()
      elapsed_time = np.append(elapsed_time, end_time - start_time)
      if i % 50 == 0:
        print('Step {}: {:4.1f}ms'.format(i, (elapsed_time[-50:].mean()) * 1000))

    print('Throughput: {:.0f} images/s'.format(N_run * batch_size / elapsed_time.sum()))


def main():
    argparser = argparse.ArgumentParser(description="Get ResNet50, benchmark througput and save as SavedModel")
    argparser.add_argument("--input-model", action='store', help="input saved model", type=str, default=None)
    argparser.add_argument("--batch-size", action='store', help="batch size to test", type=int, default=8)
    argparser.add_argument("--predict", action='store_true', help="run 1 image prediction")

    args = argparser.parse_args()
    assert args.input_model is not None, "You have to choose input model"
    if args.predict:
        predict_tftrt(args.input_model)
    else:
        benchmark_tftrt(args.input_model, batch_size=args.batch_size)

if __name__ == '__main__':
    main()
    