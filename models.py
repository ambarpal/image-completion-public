import tensorflow as tf
def generator(x):
    fc = tf.layers.dense(inputs=x, units=4*4*1024, activation=tf.nn.relu, name="projection")
    fc = tf.reshape(fc,[-1,4,4,1024])
    deconv1 = tf.layers.conv2d_transpose(inputs=fc, filters=512,
                               kernel_size=5, strides=2, padding="same",activation=tf.nn.relu, name="deconv1")
    deconv2 = tf.layers.conv2d_transpose(deconv1, filters=256,
                               kernel_size=5, strides=2, padding="same",activation=tf.nn.relu, name="deconv2")
    deconv3 = tf.layers.conv2d_transpose(inputs=deconv2, filters=128,
                               kernel_size=5, strides=2, padding="same",activation=tf.nn.relu, name="deconv3")
    deconv4 = tf.layers.conv2d_transpose(inputs=deconv3, filters=3,
                               kernel_size=5, strides=2, padding="same",activation=tf.nn.relu, name="deconv4")
    return deconv4

def discriminator(x):
    conv1 = tf.layers.conv2d(inputs=x, filters=64,
                            kernel_size=5, strides=2, padding="same", activation=tf.nn.relu, name="conv1")
    conv2 = tf.layers.conv2d(inputs=conv1, filters=128,
                            kernel_size=5, strides=2, padding="same", activation=tf.nn.relu, name="conv2")
    conv3 = tf.layers.conv2d(inputs=conv2, filters=256,
                            kernel_size=5, strides=2, padding="same", activation=tf.nn.relu, name="conv3")
    conv4 = tf.layers.conv2d(inputs=conv3, filters=512,
                            kernel_size=5, strides=2, padding="same", activation=tf.nn.relu, name="conv4")
    prob = tf.layers.dense(inputs=tf.reshape(conv4, [-1, 4*4*512]), units=1, activation=tf.nn.relu, name="prob")
    return prob