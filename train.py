import tensorflow as tf
import numpy as np
import load_data
from models import generator,discriminator 
from helper import d3_scale
import cv2
import pdb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

np.random.seed(123)
tf.set_random_seed(811)

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
        
    optimizer_disc = tf.train.GradientDescentOptimizer(learning_rate=2e-4)
    train_op_disc = optimizer_disc.minimize(loss_disc, var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "disc"))
    
    optimizer_gen = tf.train.AdamOptimizer(learning_rate = 2e-4, beta1 = 0.5)
    train_op_gen = optimizer_gen.minimize(loss_gen, var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "gen"))
    
    loader = load_data.DataLoaderMNIST()
    for ind in range(5):
        loader.register_data_file('data/CIFAR/cifar-10-batches-py/data_batch_%d' % (ind + 1))
    
    data_size = loader.get_data_size()
    num_train_epochs = 10
    num_disc_steps = 5
    batch_size = 128
    save_checkpoint_every = 100
    generate_samples_every = 50
    
    saver = tf.train.Saver()
    
    with tf.Session() as sess:
        # saver = tf.train.import_meta_graph('snapshots_cifar_30k/it_30000.ckpt.meta')
        # saver.restore(sess, 'snapshots_cifar_30k/it_30000.ckpt')
        
        sess.run(tf.global_variables_initializer())
        num_train_iter = num_train_epochs * (data_size / (batch_size * num_disc_steps))
        for iteration in range(num_train_iter):
            for disc_step in range(num_disc_steps):
                X_feed = loader.load_batch_X('data/CIFAR/cifar-10-batches-py/data_batch_%d' % ((iteration % 5) + 1), batch_size)
                Z_feed = loader.load_batch_Z(batch_size)
                _, cost_disc, DGZ_ret, DX_ret = sess.run(fetches = [train_op_disc, loss_disc, DGZ, DX], feed_dict = {X:X_feed, Z:Z_feed})
                # print ("D #%d\t%d\tLoss:%f\tDGZ:%f\tDX:%f" % (iteration, disc_step, -1 * cost_disc, np.mean(DGZ_ret), np.mean(DX_ret)))
                print ("D #%d\tLoss:%f\tDGZ:%f\tDX:%f" % (iteration, -1 * cost_disc, np.mean(DGZ_ret), np.mean(DX_ret)))                
            Z_feed = loader.load_batch_Z(batch_size)
            _, cost_gen, DGZ_ret = sess.run(fetches = [train_op_gen, loss_gen, DGZ], feed_dict = {Z:Z_feed})
            print ("G #%d\tLoss:%f\tDGZ:%f" % (iteration, -1 * cost_gen, np.mean(DGZ_ret)))
            
#             _, cost_gen, DGZ_ret = sess.run(fetches = [train_op_gen, loss_gen, DGZ], feed_dict = {Z:Z_feed})
#             print ("G #%d\tLoss:%f\tDGZ:%f" % (iteration, -1 * cost_gen, np.mean(DGZ_ret)))
            
            if iteration % save_checkpoint_every == 0 and iteration != 0:
                save_name = "snapshots/it_%d.ckpt" % iteration
                saver.save(sess, save_name)
                print "Snapshot saved to %s" % save_name
            
            if iteration % generate_samples_every == 0:
                Z_sample_all = loader.load_batch_Z(100)
                X_sample_all = loader.load_batch_X('data/CIFAR/cifar-10-batches-py/data_batch_%d' % ((iteration % 5) + 1), 100, update_iterator = False)
                gen_samples = []
                gt = []
                
                for sample_ind,(Z_sample, X_sample) in enumerate(zip(Z_sample_all, X_sample_all)):
                    Z_sample = Z_sample.reshape((-1, 100))
                    im_sample, im_score = sess.run(fetches=[GZ, DGZ], feed_dict={Z:Z_sample})
                    im_sqz = np.squeeze(np.asarray(im_sample))
                    im_sqz = d3_scale(im_sqz, out_range=(0, 255))
                    # im_sqz = cv2.cvtColor(im_sqz, cv2.COLOR_BGR2GRAY)
                    gen_samples.append(im_sqz)
                    # cv2.imwrite('analysis/generator_sample_%d.png' % (sample_ind), im_sqz)
                    
                    X_sample = X_sample.reshape((-1, 64, 64, 3))
                    X_score = sess.run(fetches=[DX], feed_dict={X:X_sample})
                    im_sqz = np.squeeze(np.asarray(X_sample))
                    im_sqz = d3_scale(im_sqz, out_range=(0, 255))
                    gt.append(im_sqz)
                    # cv2.imwrite('analysis/orig_%d.png' % sample_ind, im_sqz)
                    print "Sample # %d\tDGZ:%f\tDX:%f" % (sample_ind, np.mean(im_score), np.mean(X_score))
                
                f = plt.figure()
                for fig_ind in range(100):
                    sb = plt.subplot(10, 10, fig_ind + 1)
                    sb.axis('off')
                    sb.imshow(gen_samples[fig_ind])
                f.savefig("analysis/generator_sample_%d.png" % iteration, bbox_inches='tight')
                f.savefig("analysis/generator_latest.png", bbox_inches='tight')
                
                f2 = plt.figure()
                for fig_ind in range(100):
                    sb = plt.subplot(10, 10, fig_ind + 1)
                    sb.axis('off')
                    sb.imshow(gt[fig_ind])
                f2.savefig("analysis/orig_%d.png" % iteration, bbox_inches='tight')
                f2.savefig("analysis/orig_latest.png", bbox_inches='tight')
                
                plt.close()