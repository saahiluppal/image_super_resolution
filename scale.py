import tensorflow as tf
import tensorflow_datasets as tfds

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

def downscale(dataset='rock_paper-scissors', return_batch_as_tuple=True):
    
    # Downloading and preparing dataset
    hands_builder = tfds.builder(dataset)
    hands_builder.download_and_prepare()

    train, test = hands_builder.as_dataset(split=['train','test'], batch_size=-1, as_supervised=False)

    # Removing labels as they are of no use for improving image quality
    train = train['image']
    test = test['image']

    downscaled_images, original_images = list(), list()

    for i in range(len(train)):
        down, orig = prepare(train[i])
        downscaled_images.append(down)
        original_images.append(orig)

    downscaled_images = tf.convert_to_tensor(downscaled_images)
    original_images = tf.convert_to_tensor(original_images)

    return (downscaled_images, original_images) if return_batch_as_tuple else {'image': downscaled_images,
                                                                                'label': original_images}

