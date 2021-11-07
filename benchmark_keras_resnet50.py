import argparse
import numpy as np

import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
import time

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


def benchmark_resnet50(model, batch_size=8):
    batched_input = get_batched_input(batch_size=batch_size)
    # Benchmarking throughput
    
    N_warmup_run = 50
    N_run = 500
    elapsed_time = []

    for i in range(N_warmup_run):
        preds = model.predict(batched_input)

    for i in range(N_run):
        start_time = time.time()
        preds = model.predict(batched_input)
        end_time = time.time()
        elapsed_time = np.append(elapsed_time, end_time - start_time)
        if i % 50 == 0:
            print('Step {}: {:4.1f}ms'.format(i, (elapsed_time[-50:].mean()) * 1000))

    print('Throughput: {:.0f} images/s'.format(N_run * batch_size / elapsed_time.sum()))


def main():
    argparser = argparse.ArgumentParser(description="Get ResNet50, benchmark througput and save as SavedModel")
    argparser.add_argument("--batch-size", action='store', help="batch size to test", type=int, default=8)
    model = ResNet50(weights='imagenet')
    args = argparser.parse_args()
    benchmark_resnet50(model=model, batch_size=args.batch_size) 

if __name__ == '__main__':
    main()
    