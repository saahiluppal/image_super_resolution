import tensorflow as tf
from tensorflow.keras.layers import (Conv2D, Conv2DTranspose, Lambda, Dropout, 
                                     MaxPooling2D, LeakyReLU, concatenate, BatchNormalization)


def unet_conv_block(x, filters, kernel_size=3, batch_norm=True, dropout=False,
                    name_prefix="enc_", name_suffix=0):

    name_fn = lambda layer, num: '{}{}{}-{}'.format(name_prefix, layer, name_suffix, num)

    # First convolution:
    x = Conv2D(filters, kernel_size=kernel_size, activation=None,
               kernel_initializer='he_normal', padding='same',
               name=name_fn('conv', 1))(x)
    if batch_norm:
        x = BatchNormalization(name=name_fn('bn', 1))(x)
    x = LeakyReLU(alpha=0.3, name=name_fn('act', 1))(x)
    if dropout:
        x = Dropout(0.2, name=name_fn('drop', 1))(x)

    # Second convolution:
    x = Conv2D(filters, kernel_size=kernel_size, activation=None,
               kernel_initializer='he_normal', padding='same',
               name=name_fn('conv', 2))(x)
    if batch_norm:
        x = BatchNormalization(name=name_fn('bn', 2))(x)
    x = LeakyReLU(alpha=0.3, name=name_fn('act', 2))(x)

    return x


def unet_deconv_block(x, filters, kernel_size=2, strides=2, batch_norm=True, dropout=False,
                      name_prefix="dec_", name_suffix=0):

    name_fn = lambda layer, num: '{}{}{}-{}'.format(name_prefix, layer, name_suffix, num)

    # First convolution:
    x = Conv2D(filters, kernel_size=kernel_size, activation=None,
               kernel_initializer='he_normal', padding='same',
               name=name_fn('conv', 1))(x)
    if batch_norm:
        x = BatchNormalization(name=name_fn('bn', 1))(x)
    x = LeakyReLU(alpha=0.3, name=name_fn('act', 1))(x)
    if dropout:
        x = Dropout(0.2, name=name_fn('drop', 1))(x)

    # Second (de)convolution:
    x = Conv2DTranspose(filters, kernel_size=kernel_size, strides=strides,
                        activation=None, kernel_initializer='he_normal',
                        padding='same', name=name_fn('conv', 2))(x)
    if batch_norm:
        x = BatchNormalization(name=name_fn('bn', 2))(x)
    x = LeakyReLU(alpha=0.3, name=name_fn('act', 2))(x)

    return x


# As the dimensions of our images may not be normalized/even, it is possible that after 
# downsampling and upsampling, we do not reobtain the original size (with a difference 
# of +/- 1px).
# To avoid the problems this may cause, we define a layer to slightly resize the generated
# image to the dimensions of the target one:
ResizeToSame = lambda name: Lambda(
    lambda images: tf.image.resize(images[0], tf.shape(images[1])[1:3]),
    # `images` is a tuple of 2 tensors.
    # We resize the first image tensor to the shape of the 2nd
    name=name)


def unet(x, out_channels=3, layer_depth=4, filters_orig=32, kernel_size=4,
         batch_norm=True, final_activation='sigmoid'):

    # Encoding layers:
    filters = filters_orig
    outputs_for_skip = []
    for i in range(layer_depth):
        conv_block = unet_conv_block(x, filters, kernel_size,
                                     batch_norm=batch_norm, name_suffix=i)
        outputs_for_skip.append(conv_block)

        x = MaxPooling2D(2)(conv_block)

        filters = min(filters * 2, 512)

    # Bottleneck layers:
    x = unet_conv_block(x, filters, kernel_size, name_suffix='btleneck')

    # Decoding layers:
    for i in range(layer_depth):
        filters = max(filters // 2, filters_orig)

        use_dropout = i < (layer_depth - 2)
        deconv_block = unet_deconv_block(x, filters, kernel_size,
                                         batch_norm=batch_norm,
                                         dropout=use_dropout, name_suffix=i)

        shortcut = outputs_for_skip[-(i + 1)]
        deconv_block = ResizeToSame(
            name='resize_to_same{}'.format(i))([deconv_block, shortcut])

        x = concatenate([deconv_block, shortcut], axis=-1,
                        name='dec_conc{}'.format(i))

    x = Conv2D(filters=filters, kernel_size=kernel_size, activation='relu',
               padding='same', name='dec_out1')(x)
    x = Dropout(0.3, name='drop_out1')(x)
    x = Conv2D(filters=filters, kernel_size=kernel_size, activation='relu',
               padding='same', name='dec_out2')(x)
    x = Dropout(0.3, name='drop_out2')(x)
    x = Conv2D(filters=out_channels, kernel_size=1, activation=final_activation,
               padding='same', name='dec_output')(x)

    return x