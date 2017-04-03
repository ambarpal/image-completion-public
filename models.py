import tensorflow as tf
from helper import lrelu

def generator(x):
    fc = tf.layers.dense(inputs=x, units=4*4*1024, activation=None, name="projection",kernel_initializer = tf.contrib.layers.xavier_initializer())
    fc = tf.reshape(fc,[-1,4,4,1024])
    fc_normed = tf.nn.relu(tf.layers.batch_normalization(inputs=fc, name="fc_normed"))
    
    deconv1 = tf.layers.conv2d_transpose(inputs=fc_normed, filters=512,
                               kernel_size=5, strides=2, padding="same",activation=None, name="deconv1",kernel_initializer = tf.contrib.layers.xavier_initializer())
    deconv1_normed = tf.nn.relu(tf.layers.batch_normalization(inputs=deconv1, name="deconv1_normed"))
    
    deconv2 = tf.layers.conv2d_transpose(deconv1_normed, filters=256,
                               kernel_size=5, strides=2, padding="same",activation=None, name="deconv2",kernel_initializer = tf.contrib.layers.xavier_initializer())
    deconv2_normed = tf.nn.relu(tf.layers.batch_normalization(inputs=deconv2, name="deconv2_normed"))
    
    deconv3 = tf.layers.conv2d_transpose(inputs=deconv2_normed, filters=128,
                               kernel_size=5, strides=2, padding="same",activation=None, name="deconv3",kernel_initializer = tf.contrib.layers.xavier_initializer())
    deconv3_normed = tf.nn.relu(tf.layers.batch_normalization(inputs=deconv3, name="deconv3_normed"))
    
    deconv4 = tf.layers.conv2d_transpose(inputs=deconv3_normed, filters=3,
                               kernel_size=5, strides=2, padding="same",activation=tf.nn.tanh, name="deconv4",kernel_initializer = tf.contrib.layers.xavier_initializer())
    
    return deconv4

def discriminator(x):
    conv1 = tf.layers.conv2d(inputs=x, filters=64,
                            kernel_size=5, strides=2, padding="same", activation=None, name="conv1",kernel_initializer = tf.contrib.layers.xavier_initializer())
    conv1_normed = lrelu(tf.layers.batch_normalization(inputs=conv1, name="conv1_normed"))
    
    conv2 = tf.layers.conv2d(inputs=conv1_normed, filters=128,
                            kernel_size=5, strides=2, padding="same", activation=None, name="conv2",kernel_initializer = tf.contrib.layers.xavier_initializer())
    conv2_normed = lrelu(tf.layers.batch_normalization(inputs=conv2, name="conv2_normed"))
    
    conv3 = tf.layers.conv2d(inputs=conv2_normed, filters=256,
                            kernel_size=5, strides=2, padding="same", activation=None, name="conv3",kernel_initializer = tf.contrib.layers.xavier_initializer())
    conv3_normed = lrelu(tf.layers.batch_normalization(inputs=conv3, name="conv3_normed"))
    
    conv4 = tf.layers.conv2d(inputs=conv3_normed, filters=512,
                            kernel_size=5, strides=2, padding="same", activation=None, name="conv4",kernel_initializer = tf.contrib.layers.xavier_initializer())
    conv4_normed = lrelu(tf.layers.batch_normalization(inputs=conv4, name="conv4_normed"))
    
    prob = tf.layers.dense(inputs=tf.reshape(conv4_normed, [-1, 4*4*512]), units=1, activation=tf.nn.sigmoid, name="prob",kernel_initializer = tf.contrib.layers.xavier_initializer())
    
    return prob
