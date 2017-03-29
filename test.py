import tensorflow as tf
import load_data
from train import generator, discriminator
import numpy as np
import cv2

if __name__ == '__main__':
    X = tf.placeholder(tf.float32, shape = [None, 64, 64, 3])
    Z = tf.placeholder(tf.float32, shape = [None, 100])
    
    with tf.variable_scope("gen"):
        GZ = generator(Z)
    
    with tf.variable_scope("disc") as scope:
        DGZ = discriminator(GZ)
        scope.reuse_variables()
        DX = discriminator(X)
    
    with tf.variable_scope("loss_calculation"):
        l2 = tf.log1p(-1 * DGZ)
        l1 = tf.log(DX)
        loss_disc = tf.reduce_mean(l1 + l2)
        loss_gen = tf.reduce_mean(l2)
        
    optimizer = tf.train.AdamOptimizer(learning_rate=1e-5, beta1=0.5)
    train_op_disc = optimizer.minimize(loss_disc, var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "disc"))
    train_op_gen = optimizer.minimize(loss_gen, var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "gen"))
    
    loader = load_data.DataLoader()
    for ind in range(5):
        loader.register_data_file('data/CIFAR/cifar-10-batches-py/data_batch_%d' % (ind + 1))
    X_feed = loader.load_batch_X('data/CIFAR/cifar-10-batches-py/data_batch_%d' % 1, 1)
    Z_feed = loader.load_batch_Z(1)
    
    saver = tf.train.Saver()
    sess = tf.Session()
    saver = tf.train.import_meta_graph('snapshots/it_500.ckpt.meta')
    saver.restore(sess, tf.train.latest_checkpoint('./snapshots/'))
    
    _, cost_disc = sess.run(fetches = [train_op_disc, loss_disc], feed_dict = {X:X_feed, Z:Z_feed})
    print "D %f" % cost_disc
    _, cost_gen = sess.run(fetches = [train_op_gen, loss_gen], feed_dict = {Z:Z_feed})
    print "G %f" % cost_gen
    
    Z_feed = loader.load_batch_Z(1)
    im, im_score = sess.run(fetches=[GZ, DGZ], feed_dict={Z:Z_feed})
    print im_score
    
    im_sqz = np.squeeze(np.asarray(im))
    cv2.imwrite('analysis/mnist_test.png', im_sqz)
    # print im_sqz