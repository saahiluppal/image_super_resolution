import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.layers import (Conv2D, BatchNormalization, LeakyReLU, Dropout, Conv2DTranspose,
                            MaxPooling2D, concatenate, Lambda)

def prepare(features, scale_factor=4, augment=True,
                     return_batch_as_tuple=True, seed=None):

    image = features
    image = tf.image.convert_image_dtype(image, tf.float32)
    
    original_shape = tf.shape(image)
    original_size = original_shape[-3:-1]
    scaled_size = original_size // scale_factor
    original_size_mult = scaled_size * scale_factor
    
    if augment:
        original_shape_mult = (original_size_mult, [tf.shape(image)[-1]])
        if len(image.shape) > 3: # batched data:
            original_shape_mult = ([tf.shape(image)[0]], *original_shape_mult)
        original_shape_mult = tf.concat(original_shape_mult, axis=0)
        
        image = tf.image.random_flip_left_right(image)

        image = tf.image.random_brightness(image, max_delta=0.1, seed=seed)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5, seed=seed)
        image = tf.clip_by_value(image, 0.0, 1.0) # keeping pixel values in check

        random_scale_factor = tf.random.uniform([1], minval=1.0, maxval=1.2, 
                                                dtype=tf.float32, seed=seed)
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

def downscale(dataset='rock_paper_scissors', return_batch_as_tuple=True, scale_factor=4, split='train'):
    
    # Downloading and preparing dataset
    hands_builder = tfds.builder(dataset)
    hands_builder.download_and_prepare()

    train, test = hands_builder.as_dataset(split=['train','test'], batch_size=-1, as_supervised=False)

    assert split in ['train', 'test']

    # Removing labels as they are of no use for improving image quality
    train = train['image']
    test = test['image']

    downscaled_images, original_images = list(), list()

    if split == 'train':
        for i in range(len(train)):
            down, orig = prepare(train[i], scale_factor)
            downscaled_images.append(down)
            original_images.append(orig)
    elif split == 'test':
        for i in range(len(test)):
            down, orig = prepare(test[i], scale_factor)
            downscaled_images.append(down)
            original_images.append(orig)

    downscaled_images = tf.convert_to_tensor(downscaled_images)
    original_images = tf.convert_to_tensor(original_images)

    return (downscaled_images, original_images) if return_batch_as_tuple else {'image': downscaled_images,
                                                                                'label': original_images}


X_train, y_train = downscale(split='train')
X_test, y_test = downscale(split='test')
input_shape = y_train.shape
scale_factor = 4 # works well

Upscale = lambda name: Lambda(
    lambda images: tf.image.resize(images, tf.shape(images)[-3:-1] * scale_factor), 
    name=name)

ResizeToSame = lambda name: Lambda(
    lambda images: tf.image.resize(images[0], tf.shape(images[1])[-3:-1]), 
    # `images` is a tuple of 2 tensors.
    # We resize the first image tensor to the shape of the 2nd
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
    
    # Encoding layers:
    filters = filters_orig
    outputs_for_skip = []
    for i in range(layer_depth):
        
        # Convolution block:
        x_conv = unet_conv_block(filters, kernel_size, 
                                 dropout=dropout, batch_norm=batch_norm, 
                                 name_prefix="enc_", name_suffix=i)(x)
        
        # We save the pointer to the output of this encoding block,
        # to pass it to its parallel decoding block afterwards:
        outputs_for_skip.append(x_conv)

        # Downsampling:
        x = MaxPooling2D(2)(x_conv)

        filters = min(filters * 2, 512)

    # Bottleneck layers:
    x = unet_conv_block(filters, kernel_size, dropout=dropout, 
                        batch_norm=batch_norm, name_suffix='_btleneck')(x)

    # Decoding layers:
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
    
        # Concatenation with the output of the corresponding encoding block:
        shortcut = outputs_for_skip[-(i + 1)]
        x = ResizeToSame(name='resize_to_same{}'.format(i))([x, shortcut])
        
        x = concatenate([x, shortcut], axis=-1, name='dec_conc{}'.format(i))

        # Convolution block:
        use_dropout = dropout and (i < (layer_depth - 2))
        x = unet_conv_block(filters, kernel_size, 
                            batch_norm=batch_norm, dropout=use_dropout,
                            name_prefix="dec_", name_suffix=i)(x)

    # x = Conv2D(filters=filters, kernel_size=kernel_size, activation='relu', 
    #            padding='same', name='dec_out1')(x)  
    # x = Conv2D(filters=filters, kernel_size=kernel_size, activation='relu', 
    #            padding='same', name='dec_out2')(x)    
    x = Conv2D(filters=num_channels, kernel_size=1, activation=final_activation, 
               padding='same', name='dec_output')(x)
    
    return x


batch_size = 32
kernel_size = 4
filters_orig = 32
layer_depth = 4
use_batch_norm = batch_size > 1

# Defining model here with clear session so that every time we run this script
# new graph won't be created
tf.keras.backend.clear_session()
inputs = tf.keras.Input(shape=(None, None, input_shape[-1]), name='input')
resized_inputs = Upscale(name='upscale_inputs')(inputs)
outputs = unet(resized_inputs, layer_depth, filters_orig, kernel_size, use_batch_norm)

unet_model = tf.keras.Model(inputs, outputs)
optimizer = tf.optimizers.Adam(learning_rate=1e-4)

#unet_model.summary()

unet_model.compile(optimizer=optimizer, loss='mae')

unet_model.fit(X_train, y_train, validation_data=(X_test,y_test), epochs=10, batch_size=8,
                callbacks=[tf.keras.callbacks.EarlyStopping(), tf.keras.callbacks.TensorBoard(log_dir='./graph')])

unet_model.save('unet_model.h5')