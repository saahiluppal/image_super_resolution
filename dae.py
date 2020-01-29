import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.layers import Lambda, Conv2D, Conv2DTranspose, Input
from tensorflow.keras import Model

def prepare(features, scale_factor=4, augment=True,
                     return_batch_as_tuple=True, seed=None):

    # Tensorflow-Dataset returns batches as feature dictionaries, expected by Estimators.
    # To train Keras models, it is more straightforward to return the batch content as tuples.
    
    image = features
    # Convert the images to float type, also scaling their values from [0, 255] to [0., 1.]:
    image = tf.image.convert_image_dtype(image, tf.float32)
    
    # Computing the scaled-down shape:
    original_shape = tf.shape(image)
    original_size = original_shape[-3:-1]
    scaled_size = original_size // scale_factor
    # Just in case the original dimensions were not a multiple of `scale_factor`,
    # we slightly resize the original image so its dimensions now are
    # (to make the loss/metrics computations easier during training):
    original_size_mult = scaled_size * scale_factor
    
    # Opt. augmenting the image:
    if augment:
        original_shape_mult = (original_size_mult, [tf.shape(image)[-1]])
        if len(image.shape) > 3: # batched data:
            original_shape_mult = ([tf.shape(image)[0]], *original_shape_mult)
        original_shape_mult = tf.concat(original_shape_mult, axis=0)
        
        # Randomly applied horizontal flip:
        image = tf.image.random_flip_left_right(image)

        # Random B/S changes:
        image = tf.image.random_brightness(image, max_delta=0.1, seed=seed)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5, seed=seed)
        image = tf.clip_by_value(image, 0.0, 1.0) # keeping pixel values in check

        # Random resize and random crop back to expected size:
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
    
    # Generating the data pair for super-resolution task, 
    # i.e. the downscaled image + its original version
    image_downscaled = tf.image.resize(image, scaled_size)
    
    # Just in case the original dimensions were not a multiple of `scale_factor`,
    # we slightly resize the original image so its dimensions now are
    # (to make the loss/metrics computations easier during training):
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


def simple_dae(inputs, kernel_size=3, filter_orig=16, layer_depth=4):
    filters = filter_orig
    x = inputs
    
    for i in range(layer_depth):
        x = Conv2D(filters=filters, kernel_size=kernel_size, activation='relu', strides=2,
                   padding='same')(x)
        filters = min(filters * 2, 512)
    
    for i in range(layer_depth):
        filters = max(filters//2, filter_orig)
        x = Conv2DTranspose(filters=filters, kernel_size=kernel_size, activation='relu',
                           padding='same')(x)
    
    decoded = Conv2D(filters=inputs.shape[-1], kernel_size=1, activation='sigmoid',
                     padding='same')(x)
    
    return decoded

X_train, y_train = downscale(split='train')
X_test, y_test = downscale(split='test')

input_shape = X_train.shape
scale_factor = 4

# Custom layers to Upscale and Resize images
Upscale = lambda name: Lambda(
    lambda images: tf.image.resize(images, tf.shape(images)[-3:-1] * scale_factor), 
    name=name)

ResizeToSame = lambda name: Lambda(
    lambda images: tf.image.resize(images[0], tf.shape(images[1])[-3:-1]), 
    name=name)

def concate_layers(inputs, kernel_size=3, filters_orig=16, layer_depth=4):
    # Upscaling inputs before feeding them into model
    resized_inputs = Upscale(name='upscale_input')(inputs)

    # Get decoded layer from AE
    decoded = simple_dae(resized_inputs, kernel_size, filters_orig, layer_depth)

    # Resizing back to original size
    decoded = ResizeToSame(name='dec_output_scale')([decoded, resized_inputs])

    return decoded

kernel_size  =  4
filters_orig = 32
layer_depth  =  4

inputs = Input(shape=(None, None, input_shape[-1]))
decoded = concate_layers(inputs, kernel_size, filters_orig, layer_depth)

autoencoder = Model(inputs, decoded)

#autoencoder.summary()

callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True),
    tf.keras.callbacks.TensorBoard(log_dir='./graph', histogram_freq=0, write_graph=True)
]

autoencoder.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), loss='mae')

autoencoder.fit(X_train, y_train, epochs=200, batch_size=128, validation_data=(X_test, y_test),
                callbacks=callbacks)

autoencoder.save('dae_model.h5')