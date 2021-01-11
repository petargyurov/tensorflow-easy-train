import os
import numpy as np

from tuner import Tuner
from utils import load_image_into_numpy_array, plot_detections


MODEL = 'ssd_resnet50_v1_fpn_640x640_coco17_tpu-8'

t = Tuner(
    config_path=os.path.join('pre_trained_models', MODEL, 'pipeline.config'),
    checkpoint_path=os.path.join('pre_trained_models', MODEL, 'checkpoint', 'ckpt-0'),
    num_classes=1)

t.load_training_images(
    path='models/research/object_detection/test_images/ducky/train/')

t.set_annotation_data(data=[
    np.array([[0.436, 0.591, 0.629, 0.712]], dtype=np.float32),
    np.array([[0.539, 0.583, 0.73, 0.71]], dtype=np.float32),
    np.array([[0.464, 0.414, 0.626, 0.548]], dtype=np.float32),
    np.array([[0.313, 0.308, 0.648, 0.526]], dtype=np.float32),
    np.array([[0.256, 0.444, 0.484, 0.629]], dtype=np.float32)
])

t.prepare_data()
t.restore_weights()
t.fine_tune(batch_size=4, learning_rate=0.01, num_batches=100)

test_img_path = 'models/research/object_detection/test_images/ducky/test/out1.jpg'
detections = t.detect(test_img_path)

input_image = np.expand_dims(load_image_into_numpy_array(test_img_path), axis=0)
plot_detections(
    input_image[0],
    detections['detection_boxes'][0].numpy(),
    detections['detection_classes'][0].numpy().astype(np.uint32) + 1,
    detections['detection_scores'][0].numpy(),
    t.category_index, figsize=(15, 20),
    image_name="test.jpg")
