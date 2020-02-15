# Importing necessary libraries
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Lambda, Conv2D, Conv2DTranspose
import matplotlib.pyplot as plt
import numpy as np
import functools
import math

# Setting some global Environments
seed = 42
batch_size = 64
num_epochs = 100
scale_factor = 4

#######  Helper Functions ########
def _prepare_data_fn(features, scale_factor=4, augment=True,
                    return_batch_as_tuple=True, seed=None):
    
    image = features['image']
    image = tf.image.convert_image_dtype(image, tf.float32)
    
    original_shape = tf.shape(image)
    original_size = original_shape[-3:-1]
    scaled_size = original_size // scale_factor
    
    original_size_mult = scaled_size * scale_factor
    
    if augment:
        original_shape_mult = (original_size_mult, [tf.shape(image)[-1]])
        if len(image.shape) > 3:
            original_shape_mult = ([tf.shape(image)[0]], *original_shape_mult)
        original_shape_mult = tf.concat(original_shape_mult, axis=0)
        
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_brightness(image, max_delta=0.1, seed=seed)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5, seed=seed)
        image = tf.clip_by_value(image, 0.0, 1.0)
        
        random_scale_factor = tf.random.uniform([1], minval=1.0, maxval=1.2,
                                                dtype=tf.float32, seed= seed)
        scaled_height = tf.cast(tf.multiply(tf.cast(original_size[0], tf.float32),
                                           random_scale_factor),
                               tf.int32)
        scaled_width = tf.cast(tf.multiply(tf.cast(original_size[1], tf.float32),
                                          random_scale_factor),
                              tf.int32)
        scaled_shape = tf.squeeze(tf.stack([scaled_height, scaled_width]))
        image = tf.image.resize(image, scaled_shape)
        image = tf.image.random_crop(image, original_shape, seed=seed)
    
    image_downscaled = tf.image.resize(image, scaled_size)
    
    original_size_mult = scaled_size * scale_factor
    image = tf.image.resize(image, original_size_mult)
    
    features = (image_downscaled, image) if return_batch_as_tuple else {'image': image_downscaled,
                                                                       'label': image}
    
    return features

def get_hands_dataset_for_superres(phase='train', scale_factor=4, batch_size=32,
                                  num_epochs=None, shuffle=True, augment=False,
                                  return_batch_as_tuple=True, seed=None):
    
    assert(phase=='train' or phase == 'test')
    
    prepare_data_fn = functools.partial(
        _prepare_data_fn, scale_factor=scale_factor, augment=augment,
        return_batch_as_tuple=return_batch_as_tuple, seed=seed
    )
    
    superres_dataset = hands_builder.as_dataset(
        split=tfds.Split.TRAIN if phase == 'train' else tfds.Split.TEST
    )
    
    superres_dataset = superres_dataset.repeat(num_epochs)
    
    if shuffle:
        superres_dataset = superres_dataset.shuffle(
            hands_builder.info.splits[phase].num_examples, seed=seed
        )
    
    superres_dataset = superres_dataset.batch(batch_size)
    superres_dataset = superres_dataset.map(prepare_data_fn, num_parallel_calls=4)
    superres_dataset = superres_dataset.prefetch(1)
    
    return superres_dataset

# Downloading and preparing dataset
hands_builder = tfds.builder('rock_paper_scissors')
hands_builder.download_and_prepare()

# Setting up Useful variables
num_train_imgs = hands_builder.info.splits['train'].num_examples
num_val_imgs = hands_builder.info.splits['test'].num_examples

train_steps_per_epoch = math.ceil(num_train_imgs / batch_size)
val_steps_per_epoch = math.ceil(num_val_imgs / batch_size)

input_shape = hands_builder.info.features['image'].shape

# Defining Input Pipelines
train_hands_dataset = get_hands_dataset_for_superres(
    phase='train', scale_factor=scale_factor, batch_size=batch_size, 
    num_epochs=num_epochs, augment=True, shuffle=True, seed = seed
)

val_hands_dataset = get_hands_dataset_for_superres(
    phase = 'test', scale_factor=scale_factor, batch_size=batch_size,
    num_epochs=1, augment=False, shuffle=False, seed=seed
)

# Custom Useful Layers for handling input and output sizes
Upscale = lambda name: Lambda(
    lambda images: tf.image.resize(images, tf.shape(images)[-3:-1] * scale_factor), 
    name=name)

ResizeToSame = lambda name: Lambda(
    lambda images: tf.image.resize(images[0], tf.shape(images[1])[-3:-1]), 
    name=name)


## Encoder-Decoder architechure (Model's internal layers)
def simple_dae(inputs, kernel_size=3, filters_orig=16, layer_depth=4):
    
    filters = filters_orig
    x = inputs
    
    for i in range(layer_depth):
        x = Conv2D(filters = filters, kernel_size=kernel_size,
                  activation='relu', strides=2, padding='same',
                  name='enc_conv{}'.format(i))(x)
        filters = min(filters * 2, 512)
        
    for i in range(layer_depth):
        filters = max(filters // 2 , filters_orig)
        x = Conv2DTranspose(filters=filters, kernel_size=kernel_size,
                           activation='relu', strides=2, padding='same',
                           name='dec_deconv{}'.format(i))(x)
    
    decoded = Conv2D(filters = inputs.shape[-1], kernel_size=1,
                    activation='sigmoid', padding='same', name='dec_output')(x)
    
    return decoded


def train_for_superres(inputs, kernel_size=3, filters_orig=16, layer_depth = 4):

    resized_inputs = Upscale(name='upscale_input')(inputs)
    decoded = simple_dae(resized_inputs, kernel_size, filters_orig, layer_depth)
    decoded = ResizeToSame(name='dec_output_scale')([decoded, resized_inputs])
    
    return decoded

kernel_size  =  4
filters_orig = 32
layer_depth  =  4

# Defining Model
inputs = Input(shape=(None, None, input_shape[-1]))
decoded = train_for_superres(inputs, kernel_size, filters_orig, layer_depth)

autoencoder = Model(inputs, decoded)
#autoencoder.summary()

# Defining PSNR Metric
psnr_metric = functools.partial(tf.image.psnr, max_val=1.)
psnr_metric.__name__ = 'psnr'

# Defining Callbacks
callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
    tf.keras.callbacks.TensorBoard(log_dir='./graph'),
]

optimizer = tf.optimizers.Adam(learning_rate=1e-4)
autoencoder.compile(optimizer=optimizer, loss='mae', metrics=[psnr_metric, 'accuracy'])

autoencoder.fit(train_hands_dataset, epochs=num_epochs, steps_per_epoch=train_steps_per_epoch, 
               validation_data=val_hands_dataset, validation_steps=val_steps_per_epoch,
               callbacks=callbacks)

autoencoder.save('dae_model.h5')