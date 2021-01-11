import os
import random

import numpy as np
import tensorflow as tf
from object_detection.builders import model_builder
from object_detection.utils import config_util

from utils import load_image_into_numpy_array


class Tuner(object):
    def __init__(self, config_path, checkpoint_path, num_classes):
        self.pipeline_config = config_path
        self.checkpoint_path = checkpoint_path
        self.num_classes = num_classes

        self.config = config_util.get_configs_from_pipeline_file(
            self.pipeline_config)

        model_config = self.config['model']
        model_config.ssd.num_classes = self.num_classes
        model_config.ssd.freeze_batchnorm = True

        self.model = model_builder.build(
            model_config=model_config, is_training=True)

        self.train_images_np = []
        self.train_image_tensors = []
        self.gt_boxes = []  # ground truth
        self.gt_classes_one_hot_tensors = []
        self.gt_box_tensors = []

        self.category_index = {  # TODO
            1: {'id': 1, 'name': 'rubber_ducky'}}

    def load_training_images(self, path):
        for i in range(1, 6):  # TODO: improve this method
            image_path = os.path.join(path, 'robertducky' + str(i) + '.jpg')
            self.train_images_np.append(load_image_into_numpy_array(image_path))

    def set_annotation_data(self, data):
        self.gt_boxes = data

    def prepare_data(self):
        label_id_offset = 1
        for (train_image_np, gt_box_np) in zip(
                self.train_images_np, self.gt_boxes):
            self.train_image_tensors.append(tf.expand_dims(tf.convert_to_tensor(
                train_image_np, dtype=tf.float32), axis=0))
            self.gt_box_tensors.append(
                tf.convert_to_tensor(gt_box_np, dtype=tf.float32))
            zero_indexed_groundtruth_classes = tf.convert_to_tensor(
                np.ones(shape=[gt_box_np.shape[0]],
                        dtype=np.int32) - label_id_offset)
            self.gt_classes_one_hot_tensors.append(tf.one_hot(
                zero_indexed_groundtruth_classes, self.num_classes))

    def restore_weights(self):
        tf.keras.backend.clear_session()

        fake_box_predictor = tf.compat.v2.train.Checkpoint(
            _base_tower_layers_for_heads=self.model._box_predictor._base_tower_layers_for_heads,
            # _prediction_heads=self.model._box_predictor._prediction_heads,
            #    (i.e., the classification head that we *will not* restore)
            _box_prediction_head=self.model._box_predictor._box_prediction_head,
        )
        fake_model = tf.compat.v2.train.Checkpoint(
            _feature_extractor=self.model._feature_extractor,
            _box_predictor=fake_box_predictor)
        ckpt = tf.compat.v2.train.Checkpoint(model=fake_model)
        ckpt.restore(self.checkpoint_path).expect_partial()

        # Run model through a dummy image so that variables are created
        image, shapes = self.model.preprocess(tf.zeros([1, 640, 640, 3]))
        prediction_dict = self.model.predict(image, shapes)
        _ = self.model.postprocess(prediction_dict, shapes)

    def fine_tune(self, batch_size, learning_rate, num_batches):
        tf.keras.backend.set_learning_phase(True)

        # Select variables in top layers to fine-tune.
        trainable_variables = self.model.trainable_variables
        to_fine_tune = []
        prefixes_to_train = [
            'WeightSharedConvolutionalBoxPredictor/WeightSharedConvolutionalBoxHead',
            'WeightSharedConvolutionalBoxPredictor/WeightSharedConvolutionalClassHead']
        for var in trainable_variables:
            if any([var.name.startswith(prefix) for prefix in
                    prefixes_to_train]):
                to_fine_tune.append(var)

        optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate,
                                            momentum=0.9)
        train_step_fn = self._get_model_train_step_function(optimizer,
                                                            to_fine_tune,
                                                            batch_size)

        print('Start fine-tuning!', flush=True)
        for idx in range(num_batches):
            # Grab keys for a random subset of examples
            all_keys = list(range(len(self.train_images_np)))
            random.shuffle(all_keys)
            example_keys = all_keys[:batch_size]

            # Note that we do not do data augmentation in this demo.  If you want a
            # a fun exercise, we recommend experimenting with random horizontal flipping
            # and random cropping :)
            gt_boxes_list = [self.gt_box_tensors[key] for key in example_keys]
            gt_classes_list = [self.gt_classes_one_hot_tensors[key] for key in
                               example_keys]
            image_tensors = [self.train_image_tensors[key] for key in
                             example_keys]

            # Training step (forward pass + backwards pass)
            total_loss = train_step_fn(image_tensors, gt_boxes_list,
                                       gt_classes_list)

            if idx % 10 == 0:
                print('batch ' + str(idx) + ' of ' + str(num_batches)
                      + ', loss=' + str(total_loss.numpy()), flush=True)

        print('Done fine-tuning!')

    def _get_model_train_step_function(self, optimizer, vars_to_fine_tune,
                                       batch_size):
        """Get a tf.function for training step."""

        # Use tf.function for a bit of speed.
        # Comment out the tf.function decorator if you want the inside of the
        # function to run eagerly.
        @tf.function
        def train_step_fn(image_tensors,
                          groundtruth_boxes_list,
                          groundtruth_classes_list):
            """A single training iteration.

            Args:
              image_tensors: A list of [1, height, width, 3] Tensor of type tf.float32.
                Note that the height and width can vary across images, as they are
                reshaped within this function to be 640x640.
              groundtruth_boxes_list: A list of Tensors of shape [N_i, 4] with type
                tf.float32 representing groundtruth boxes for each image in the batch.
              groundtruth_classes_list: A list of Tensors of shape [N_i, num_classes]
                with type tf.float32 representing groundtruth boxes for each image in
                the batch.

            Returns:
              A scalar tensor representing the total loss for the input batch.
            """
            shapes = tf.constant(batch_size * [[640, 640, 3]], dtype=tf.int32)
            self.model.provide_groundtruth(
                groundtruth_boxes_list=groundtruth_boxes_list,
                groundtruth_classes_list=groundtruth_classes_list)
            with tf.GradientTape() as tape:
                preprocessed_images = tf.concat(
                    [self.model.preprocess(image_tensor)[0]
                     for image_tensor in image_tensors], axis=0)
                prediction_dict = self.model.predict(preprocessed_images,
                                                     shapes)
                losses_dict = self.model.loss(prediction_dict, shapes)
                total_loss = losses_dict['Loss/localization_loss'] + \
                             losses_dict['Loss/classification_loss']
                gradients = tape.gradient(total_loss, vars_to_fine_tune)
                optimizer.apply_gradients(zip(gradients, vars_to_fine_tune))
            return total_loss

        return train_step_fn

    @tf.function
    def detect(self, input_image):
        input_image = np.expand_dims(load_image_into_numpy_array(input_image),
                                     axis=0)
        input_tensor = tf.convert_to_tensor(input_image, dtype=tf.float32)
        """Run detection on an input image.

        Args:
          input_tensor: A [1, height, width, 3] Tensor of type tf.float32.
            Note that height and width can be anything since the image will be
            immediately resized according to the needs of the model within this
            function.

        Returns:
          A dict containing 3 Tensors (`detection_boxes`, `detection_classes`,
            and `detection_scores`).
        """
        preprocessed_image, shapes = self.model.preprocess(input_tensor)
        prediction_dict = self.model.predict(preprocessed_image, shapes)

        return self.model.postprocess(prediction_dict, shapes)
