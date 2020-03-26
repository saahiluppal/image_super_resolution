# Importing necessary libraries
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Lambda, Conv2D, Conv2DTranspose
import matplotlib.pyplot as plt
import numpy as np
import functools
from tensorflow.keras.layers import Dropout, MaxPooling2D, LeakyReLU, concatenate, BatchNormalizationimport math

# Setting some global Environments
seed = 42
BATCH_SIZE = 64
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

def get_hands_dataset_for_superres(phase='train', scale_factor=4, BATCH_SIZE=32,
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
    
    superres_dataset = superres_dataset.batch(BATCH_SIZE)
    superres_dataset = superres_dataset.map(prepare_data_fn, num_parallel_calls=4)
    superres_dataset = superres_dataset.prefetch(tf.data.experimental.AUTOTUNE)
    
    return superres_dataset

# Downloading and preparing dataset
hands_builder = tfds.builder('rock_paper_scissors')
hands_builder.download_and_prepare()

# Setting up Useful variables
num_train_imgs = hands_builder.info.splits['train'].num_examples
num_val_imgs = hands_builder.info.splits['test'].num_examples

train_steps_per_epoch = math.ceil(num_train_imgs / BATCH_SIZE)
val_steps_per_epoch = math.ceil(num_val_imgs / BATCH_SIZE)

input_shape = hands_builder.info.features['image'].shape

# Defining Input Pipelines
train_hands_dataset = get_hands_dataset_for_superres(
    phase='train', scale_factor=scale_factor, batch_size=BATCH_SIZE, 
    num_epochs=num_epochs, augment=True, shuffle=True, seed = seed
)

val_hands_dataset = get_hands_dataset_for_superres(
    phase = 'test', scale_factor=scale_factor, batch_size=BATCH_SIZE,
    num_epochs=1, augment=False, shuffle=False, seed=seed
)

# Custom Useful Layers for handling input and output sizes
Upscale = lambda name: Lambda(
    lambda images: tf.image.resize(images, tf.shape(images)[-3:-1] * scale_factor), 
    name=name)

ResizeToSame = lambda name: Lambda(
    lambda images: tf.image.resize(images[0], tf.shape(images[1])[-3:-1]), 
    name=name)

def name_layer_factory(num=0, name_prefix="", name_suffix=""):
    """
    Helper function to name all our layers.
    """
    def name_layer_fn(layer):
        return '{}{}{}-{}'.format(name_prefix, layer, name_suffix, num)
    
    return name_layer_fn

def conv_bn_lrelu(filters, kernel_size=3, batch_norm=True,
                  kernel_initializer='he_normal', padding='same',
                  name_fn=lambda layer: "conv_bn_lrelu-{}".format(layer)):
    
    def block(x):
        x = Conv2D(filters, kernel_size=kernel_size, 
                   activation=None, kernel_initializer=kernel_initializer, 
                   padding=padding, name=name_fn('conv'))(x)
        if batch_norm:
            x = BatchNormalization(name=name_fn('bn'))(x)
        x = LeakyReLU(alpha=0.3, name=name_fn('act'))(x)
        return x
    
    return block

def unet_conv_block(filters, kernel_size=3,
                    batch_norm=True, dropout=False, 
                    name_prefix="enc_", name_suffix=0):
    
    def block(x):
        # First convolution:
        name_fn = name_layer_factory(1, name_prefix, name_suffix)
        x = conv_bn_lrelu(filters, kernel_size=kernel_size, batch_norm=batch_norm, 
                          name_fn=name_layer_factory(1, name_prefix, name_suffix))(x)
        if dropout:
            x = Dropout(0.2, name=name_fn('drop'))(x)

        # Second convolution:
        name_fn = name_layer_factory(2, name_prefix, name_suffix)
        x = conv_bn_lrelu(filters, kernel_size=kernel_size, batch_norm=batch_norm, 
                          name_fn=name_layer_factory(2, name_prefix, name_suffix))(x)

        return x
    
    return block

def unet(x, layer_depth=4, filters_orig=32, kernel_size=4, 
         batch_norm=True, dropout=True, final_activation='sigmoid'):

    num_channels = x.shape[-1]
    
    filters = filters_orig
    outputs_for_skip = []
    for i in range(layer_depth):
        
        x_conv = unet_conv_block(filters, kernel_size, 
                                 dropout=dropout, batch_norm=batch_norm, 
                                 name_prefix="enc_", name_suffix=i)(x)
        
        outputs_for_skip.append(x_conv)

        x = MaxPooling2D(2)(x_conv)

        filters = min(filters * 2, 512)

    x = unet_conv_block(filters, kernel_size, dropout=dropout, 
                        batch_norm=batch_norm, name_suffix='_btleneck')(x)

    for i in range(layer_depth):
        filters = max(filters // 2, filters_orig)

        # Upsampling:
        name_fn = name_layer_factory(3, "ups_", i)
        x = Conv2DTranspose(filters, kernel_size=kernel_size, strides=2, 
                            activation=None, kernel_initializer='he_normal', 
                            padding='same', name=name_fn('convT'))(x)
        if batch_norm:
            x = BatchNormalization(name=name_fn('bn'))(x)
        x = LeakyReLU(alpha=0.3, name=name_fn('act'))(x)
    
        shortcut = outputs_for_skip[-(i + 1)]
        x = ResizeToSame(name='resize_to_same{}'.format(i))([x, shortcut])
        
        x = concatenate([x, shortcut], axis=-1, name='dec_conc{}'.format(i))

        use_dropout = dropout and (i < (layer_depth - 2))
        x = unet_conv_block(filters, kernel_size, 
                            batch_norm=batch_norm, dropout=use_dropout,
                            name_prefix="dec_", name_suffix=i)(x)

    x = Conv2D(filters=num_channels, kernel_size=1, activation=final_activation, 
               padding='same', name='dec_output')(x)
    
    return x

kernel_size  =  4
filters_orig = 32
layer_depth  =  4
use_batch_norm = BATCH_SIZE > 1

inputs = tf.keras.layers.Input(shape=(None, None, input_shape[-1]), name='input')
resized_inputs = Upscale(name='upscale_input')(inputs)
outputs = unet(resized_inputs, layer_depth, filters_orig, kernel_size, use_batch_norm)
unet_model = tf.keras.Model(inputs, outputs)

callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=5, monitor='val_loss', restore_best_weights=True),
]

psnr_metric = functools.partial(tf.image.psnr, max_val=1.)
psnr_metric.__name__ = 'psnr'

optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
unet_model.compile(optimizer=optimizer, loss='mae', metrics=[psnr_metric])

unet_model.fit(
    train_hands_dataset, epochs=num_epochs, steps_per_epoch=train_steps_per_epoch,
    validation_data = val_hands_dataset, validation_steps=val_steps_per_epoch, 
    callbacks=callbacks
)

unet_model.save('unet_model.h5')