import tensorflow as tf
from tensorflow.keras import layers

def lrelu(x):
   return tf.maximum(x * 0.2, x)

def upsample_and_concat(x1, x2, output_channels, in_channels):
   pool_size = 2
   deconv_filter = tf.Variable(tf.random.normal([pool_size, pool_size, output_channels, in_channels], stddev=0.02))
   deconv = layers. layers.Conv2DTranspose(output_channels, (pool_size, pool_size), strides=(pool_size, pool_size), padding='same')(x1)

   deconv_output = layers.Concatenate(axis=3)([deconv, x2])
   deconv_output.set_shape([None, None, None, output_channels * 2])

   return deconv_output


def unet(input):
    conv1 = layers.Conv2D(32, (3, 3), padding='same', activation=lrelu)(input)
    conv1 = layers.Conv2D(32, (3, 3), padding='same', activation=lrelu)(conv1)
    conv1 = layers.Conv2D(32, (3, 3), padding='same', activation=lrelu)(conv1)
    conv1 = layers.Conv2D(32, (3, 3), padding='same', activation=lrelu)(conv1)
    pool1 = layers.MaxPooling2D((2, 2), padding='same')(conv1)

    conv2 =  layers.Conv2D( 64, [3, 3],padding='same',  activation=lrelu)(pool1)
    conv2 =  layers.Conv2D( 64, [3, 3], padding='same', activation=lrelu)(conv2)
    conv2 =  layers.Conv2D( 64, [3, 3],padding='same',  activation=lrelu)(conv2)
    conv2 =  layers.Conv2D( 64, [3, 3], padding='same', activation=lrelu)(conv2)
    pool2 = layers.MaxPooling2D( [2, 2], padding='same')(conv2)

    conv3 =  layers.Conv2D( 128, [3, 3], padding='same', activation=lrelu)(pool2)
    conv3 =  layers.Conv2D( 128, [3, 3], padding='same', activation=lrelu)(conv3)
    conv3 =  layers.Conv2D( 128, [3, 3], padding='same', activation=lrelu)(conv3)
    conv3 =  layers.Conv2D( 128, [3, 3], padding='same', activation=lrelu)(conv3)
    pool3 = layers.MaxPooling2D( [2, 2], padding='same')(conv3)

    conv4 =  layers.Conv2D( 256, [3, 3], padding='same', activation=lrelu)(pool3)
    conv4 =  layers.Conv2D( 256, [3, 3],padding='same',  activation=lrelu)(conv4)
    conv4 =  layers.Conv2D( 256, [3, 3], padding='same', activation=lrelu)(conv4)
    conv4 =  layers.Conv2D( 256, [3, 3], padding='same', activation=lrelu)(conv4)
    pool4 = layers.MaxPooling2D([2, 2], padding='same')(conv4)

    conv5 =  layers.Conv2D( 512, [3, 3], padding='same', activation=lrelu)(pool4)
    conv5 =  layers.Conv2D( 512, [3, 3],padding='same',  activation=lrelu)(conv5)
    conv5 =  layers.Conv2D( 512, [3, 3], padding='same', activation=lrelu)(conv5)
    conv5 =  layers.Conv2D( 512, [3, 3],  apadding='same',ctivation=lrelu)(conv5)

    up6 = upsample_and_concat(conv5, conv4, 256, 512)
    conv6 =  layers.Conv2D(256, [3, 3],  padding='same', activation=lrelu)(up6)
    conv6 =  layers.Conv2D( 256, [3, 3],  padding='same', activation=lrelu)(conv6)
    conv6 =  layers.Conv2D(256, [3, 3],  padding='same', activation=lrelu)(conv6)

    up7 = upsample_and_concat(conv6, conv3, 128, 256)
    conv7 =  layers.Conv2D( 128, [3, 3],   padding='same',activation=lrelu)(up7)
    conv7 =  layers.Conv2D( 128, [3, 3],   padding='same',activation=lrelu)(conv7)
    conv7 =  layers.Conv2D( 128, [3, 3],   padding='same',activation=lrelu)(conv7)

    up8 = upsample_and_concat(conv7, conv2, 64, 128)
    conv8 =  layers.Conv2D( 64, [3, 3],  padding='same', activation=lrelu)(up8)
    conv8 =  layers.Conv2D( 64, [3, 3],   padding='same',activation=lrelu)(conv8)
    conv8 =  layers.Conv2D( 64, [3, 3],   padding='same',activation=lrelu)(conv8)

    up9 = upsample_and_concat(conv8, conv1, 32, 64)
    conv9 =  layers.Conv2D( 32, [3, 3],   padding='same',activation=lrelu)(up9)
    conv9 =  layers.Conv2D( 32, [3, 3],   padding='same',activation=lrelu)(conv9)
    conv9 =  layers.Conv2D( 32, [3, 3],   padding='same',activation=lrelu)(conv9)

    conv10 =  layers.Conv2D( 1, [1, 1],   padding='same',activation=None)(conv9)
    #out = tf.depth_to_space(conv10, 2)
    return conv10


def feature_encoding(input):
    conv1 = layers.Conv2D(32, (3, 3), padding='same', activation=lrelu, name='fe_conv1')(input)
    conv2 = layers.Conv2D(32, (3, 3), padding='same', activation=lrelu, name='fe_conv2')(conv1)
    conv3 = layers.Conv2D(32, (3, 3), padding='same', activation=lrelu, name='fe_conv3')(conv2)
    conv4 = layers.Conv2D(32, (3, 3), padding='same', activation=lrelu, name='fe_conv4')(conv3)
    conv4 = squeeze_excitation_layer(conv4, 32, 2)
    output = layers.Conv2D(1, (3, 3), padding='same', activation=lrelu, name='fe_conv5')(conv4)
    return output


def avg_pool(feature_map):
    ksize = [[1, 1, 1, 1], [1, 2, 2, 1], [1, 4, 4, 1], [1, 8, 8, 1], [1, 16, 16, 1]]
    pool1 = layers.AveragePooling2D(pool_size=ksize[0], strides=ksize[0], padding='valid')(feature_map)
    pool2 = layers.AveragePooling2D(pool_size=ksize[1], strides=ksize[1], padding='valid')(feature_map)
    pool3 = layers.AveragePooling2D(pool_size=ksize[2], strides=ksize[2], padding='valid')(feature_map)
    pool4 = layers.AveragePooling2D(pool_size=ksize[3], strides=ksize[3], padding='valid')(feature_map)
    pool5 = layers.AveragePooling2D(pool_size=ksize[4], strides=ksize[4], padding='valid')(feature_map)

    return pool1, pool2, pool3, pool4, pool5


def all_unet(pool1, pool2, pool3, pool4, pool5):
    unet1 = unet(pool1)
    unet2 = unet(pool2)
    unet3 = unet(pool3)
    unet4 = unet(pool4)
    unet5 = unet(pool5)

    return unet1, unet2, unet3, unet4, unet5


def resize_all_image(unet1, unet2, unet3, unet4, unet5):
    resize1 = tf.image.resize_images(images=unet1, size=[tf.shape(unet1, out_type=tf.int32)[1],tf.shape(unet1, out_type=tf.int32)[2]], method=tf.image.ResizeMethod.BILINEAR)
    resize2 = tf.image.resize_images(images=unet2, size=[tf.shape(unet1, out_type=tf.int32)[1],tf.shape(unet1, out_type=tf.int32)[2]], method=tf.image.ResizeMethod.BILINEAR)
    resize3 = tf.image.resize_images(images=unet3, size=[tf.shape(unet1, out_type=tf.int32)[1],tf.shape(unet1, out_type=tf.int32)[2]], method=tf.image.ResizeMethod.BILINEAR)
    resize4 = tf.image.resize_images(images=unet4, size=[tf.shape(unet1, out_type=tf.int32)[1],tf.shape(unet1, out_type=tf.int32)[2]], method=tf.image.ResizeMethod.BILINEAR)
    resize5 = tf.image.resize_images(images=unet5, size=[tf.shape(unet1, out_type=tf.int32)[1],tf.shape(unet1, out_type=tf.int32)[2]], method=tf.image.ResizeMethod.BILINEAR)

    return resize1, resize2, resize3, resize4, resize5


def to_clean_image(feature_map, resize1, resize2, resize3, resize4, resize5):
    concat = layers.Concatenate(axis=3)([feature_map, resize1, resize2, resize3, resize4, resize5])
    sk_conv1 = layers.Conv2D(7, (3, 3), padding='same', activation=lrelu)(concat)
    sk_conv2 = layers.Conv2D(7, (5, 5), padding='same', activation=lrelu)(concat)
    sk_conv3 = layers.Conv2D(7, (7, 7), padding='same', activation=lrelu)(concat)
    sk_out = selective_kernel_layer(sk_conv1, sk_conv2, sk_conv3, 4, 7)
    output = layers.Conv2D(1, (3, 3), padding='same', activation=None)(sk_out)

    return output


def squeeze_excitation_layer(input_x, out_dim, middle):
    squeeze = layers.GlobalAveragePooling2D()(input_x)
    excitation = layers.Dense(middle, use_bias=True)(squeeze)
    excitation = layers.ReLU()(excitation)
    excitation = layers.Dense(out_dim, use_bias=True)(excitation)
    excitation = layers.Activation('sigmoid')(excitation)
    excitation = layers.Reshape((1, 1, out_dim))(excitation)
    scale = layers.Multiply()([input_x, excitation])
    return scale


def selective_kernel_layer(sk_conv1, sk_conv2, sk_conv3, middle, out_dim):
    sum_u = layers.Add()([sk_conv1, sk_conv2, sk_conv3])
    squeeze = layers.GlobalAveragePooling2D()(sum_u)
    squeeze = layers.Reshape((1, 1, out_dim))(squeeze)
    z = layers.Dense(middle, use_bias=True)(squeeze)
    z = layers.ReLU()(z)
    a1 = layers.Dense(out_dim, use_bias=True)(z)
    a2 = layers.Dense(out_dim, use_bias=True)(z)
    a3 = layers.Dense(out_dim, use_bias=True)(z)

    before_softmax = layers.Concatenate(axis=1)([a1, a2, a3])
    after_softmax = layers.Softmax(axis=1)(before_softmax)
    a1 = after_softmax[:, 0, :, :]
    a1 = layers.Reshape((1, 1, out_dim))(a1)
    a2 = after_softmax[:, 1, :, :]
    a2 = layers.Reshape((1, 1, out_dim))(a2)
    a3 = after_softmax[:, 2, :, :]
    a3 = layers.Reshape((1, 1, out_dim))(a3)
    select_1 = layers.Multiply()([sk_conv1, a1])
    select_2 = layers.Multiply()([sk_conv2, a2])
    select_3 = layers.Multiply()([sk_conv3, a3])

    out = layers.Add()([select_1, select_2 , select_3])

    return out


def network(in_image):
    print('here')
    feature_map = feature_encoding(in_image)
    # print('feature_map ')
    feature_map_2 = tf.concat([in_image, feature_map], 3)
    # print('feature_map 2')
    pool1, pool2, pool3, pool4, pool5 = avg_pool(feature_map_2)
    
    # print('pool')
    unet1, unet2, unet3, unet4, unet5 = all_unet(pool1, pool2, pool3, pool4, pool5)
    # print('unet')
    resize1, resize2, resize3, resize4, resize5 = resize_all_image(unet1, unet2, unet3, unet4, unet5)
    # print('resize')
    out_image = to_clean_image(feature_map_2, resize1, resize2, resize3, resize4, resize5)

    return out_image