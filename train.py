import tensorflow as tf
import numpy as np
import load_data

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
    prob = tf.layers.dense(inputs=tf.reshape(conv4, [-1, 4*4*512]), units=1, activation=tf.nn.sigmoid, name="prob")
    return prob

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
        loss_gen = tf.reduce_mean(l2)
        
    optimizer = tf.train.AdamOptimizer(learning_rate=1e-5, beta1=0.5)
    train_op_disc = optimizer.minimize(loss_disc, var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "disc"))
    train_op_gen = optimizer.minimize(loss_gen, var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "gen"))
    
    loader = load_data.DataLoader()
    for ind in range(5):
        loader.register_data_file('data/CIFAR/cifar-10-batches-py/data_batch_%d' % (ind + 1))
    
    num_train_iter = 1000
    num_disc_steps = 10
    batch_size = 64
    save_checkpoint_every = 100
    
    saver = tf.train.Saver()
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for iteration in range(num_train_iter):
            for disc_step in range(num_disc_steps):
                X_feed = loader.load_batch_X('data/CIFAR/cifar-10-batches-py/data_batch_%d' % ((iteration % 5) + 1), batch_size)
                Z_feed = loader.load_batch_Z(batch_size)
                _, cost_disc = sess.run(fetches = [train_op_disc, loss_disc], feed_dict = {X:X_feed, Z:Z_feed})
                print ("D #%d\t%d\t%f" % (iteration, disc_step, cost_disc))
            Z_feed = loader.load_batch_Z(batch_size)
            _, cost_gen = sess.run(fetches = [train_op_gen, loss_gen], feed_dict = {Z:Z_feed})
            print ("G #%d\t%f" % (iteration, cost_gen))
            
            if iteration % save_checkpoint_every == 0 and iteration != 0:
                save_name = "snapshots/it_%d.ckpt" % iteration
                saver.save(sess, save_name)
                print "Snapshot saved to %s" % save_name