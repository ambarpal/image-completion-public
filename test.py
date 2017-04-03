import tensorflow as tf
import load_data
from models import generator, discriminator
import numpy as np
import cv2
from helper import d3_scale
import pdb

if __name__ == '__main__':
    X = tf.placeholder(tf.float32, shape = [None, 64, 64, 3])
    Z = tf.placeholder(tf.float32, shape = [None, 100])
    
    # Do not set scopes with the loss as a name as we need to share the discriminator instead
    # of making multiple copies
    
    with tf.variable_scope("gen"):
        GZ = generator(Z)
    
    with tf.variable_scope("disc") as scope:
        DGZ = discriminator(GZ)
        scope.reuse_variables()
        DX = discriminator(X)
    
    with tf.variable_scope("loss_calculation"):
        l2 = tf.log1p(-1 * DGZ)
        l1 = tf.log(DX)
        loss_disc = -1 * tf.reduce_mean(l1 + l2)
        # loss_gen = tf.reduce_mean(l2)
        loss_gen = -1 * tf.reduce_mean(DGZ)
        
    optimizer_disc = tf.train.GradientDescentOptimizer(learning_rate = 2e-4)
    train_op_disc = optimizer_disc.minimize(loss_disc, var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "disc"))
    
    optimizer_gen = tf.train.AdamOptimizer(learning_rate = 2e-4, beta1 = 0.5)
    train_op_gen = optimizer_gen.minimize(loss_gen, var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "gen"))
    
    loader = load_data.DataLoaderMNIST()
    for ind in range(5):
        loader.register_data_file('data/CIFAR/cifar-10-batches-py/data_batch_%d' % (ind + 1))
    X_feed = loader.load_batch_X('data/CIFAR/cifar-10-batches-py/data_batch_%d' % 1, 1)
    Z_feed = loader.load_batch_Z(1)
    
    saver = tf.train.Saver()
    sess = tf.Session()
    saver = tf.train.import_meta_graph('snapshots/it_30000.ckpt.meta')
    saver.restore(sess, 'snapshots/it_30000.ckpt')
    
    _, cost_disc = sess.run(fetches = [train_op_disc, loss_disc], feed_dict = {X:X_feed, Z:Z_feed})
    print "D %f" % cost_disc
    _, cost_gen = sess.run(fetches = [train_op_gen, loss_gen], feed_dict = {Z:Z_feed})
    print "G %f" % cost_gen
    
    Z_feed = np.zeros((1, 100))
    Z_feed[0][0] = 1
    im, im_score = sess.run(fetches=[GZ, DGZ], feed_dict={Z:Z_feed})
    print im_score
    
    im_sqz = np.squeeze(np.asarray(im))
    im_sqz = d3_scale(im_sqz, out_range=(0, 255))
    cv2.imwrite('analysis/mnist_test.png', im_sqz)
    pdb.set_trace()
    # print im_sqz