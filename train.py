import tensorflow as tf
import numpy as np
import load_Data_new
from models import generator,discriminator 
from helper import d3_scale, save_sample_images
import cv2
import pdb

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
        DGZ_raw, DGZ = discriminator(GZ)
        scope.reuse_variables()
        DX_raw, DX = discriminator(X)
    
    with tf.variable_scope("loss_calculation"):
        loss_d_fake = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(DGZ_raw), logits=DGZ_raw)
        loss_d_real = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(DX_raw), logits=DX_raw)
        loss_g = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(DGZ_raw), logits=DGZ_raw)
        loss_disc = tf.reduce_mean(loss_d_fake + loss_d_real)
        loss_gen = tf.reduce_mean(loss_g)
        
    optimizer_disc = tf.train.AdamOptimizer(learning_rate=2e-4, beta1 = 0.5)
    train_op_disc = optimizer_disc.minimize(loss_disc, var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "disc"))
    
    optimizer_gen = tf.train.AdamOptimizer(learning_rate=2e-4, beta1 = 0.5)
    train_op_gen = optimizer_gen.minimize(loss_gen, var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "gen"))
    
    DATASET = 'CIFAR10'
    # DATASET = 'MNIST'
    loader = load_Data_new.DataLoader(DATASET)
    
    data_size = loader.get_data_size()
    if DATASET == 'MNIST':
        num_train_epochs = 10
        num_disc_steps = 1
        batch_size = 128
        save_checkpoint_every = 250
        generate_samples_every = 100
    elif DATASET == 'CIFAR10':
        num_train_epochs = 50
        num_disc_steps = 1
        batch_size = 128
        save_checkpoint_every = 2500
        generate_samples_every = 250
    
    saver = tf.train.Saver(max_to_keep=2)
    
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph('saved_analysis_snapshots/snapshots_cifar_complete_from_60_epoch/it_2500.ckpt.meta')
        saver.restore(sess, 'saved_analysis_snapshots/snapshots_cifar_complete_from_60_epoch/it_2500.ckpt')
        
        # sess.run(tf.global_variables_initializer())
        num_train_iter = num_train_epochs * (data_size / (batch_size * num_disc_steps))
        
        Z_sample_all = loader.load_batch_Z(100)
        sum_g_loss, num_g_loss = 0, 0
        sum_d_loss, num_d_loss = 0, 0
        
        for iteration in range(num_train_iter):
            for disc_step in range(num_disc_steps):
                X_feed = loader.load_batch_X(batch_size)
                Z_feed = loader.load_batch_Z(batch_size)
                _, cost_disc, DGZ_ret, DX_ret = sess.run(fetches = [train_op_disc, loss_disc, DGZ, DX], feed_dict = {X:X_feed, Z:Z_feed})
                sum_d_loss += cost_disc
                num_d_loss += 1
                # print ("#%d D #%d\tLossD:%f\tAvgD:%f\tDGZ:%f\tDX:%f" % ((iteration * batch_size * num_disc_steps) / data_size , iteration, cost_disc, sum_d_loss * 1.0 / num_d_loss, np.mean(DGZ_ret), np.mean(DX_ret)))  
                
            Z_feed = loader.load_batch_Z(batch_size)
            _, cost_gen, DGZ_ret = sess.run(fetches = [train_op_gen, loss_gen, DGZ], feed_dict = {Z:Z_feed})
            sum_g_loss += cost_gen
            num_g_loss += 1
            # print ("#%d G #%d\tLossG:%f\tAvgG:%f\tDGZ:%f" % ((iteration * batch_size * num_disc_steps) / data_size, iteration, cost_gen, sum_g_loss * 1.0 / num_g_loss, np.mean(DGZ_ret)))
            
            Z_feed = loader.load_batch_Z(batch_size)
            _, cost_gen, DGZ_ret = sess.run(fetches = [train_op_gen, loss_gen, DGZ], feed_dict = {Z:Z_feed})
            sum_g_loss += cost_gen
            num_g_loss += 1
            # print ("#%d G #%d\tLossG:%f\tAvgG:%f\tDGZ:%f" % ((iteration * batch_size * num_disc_steps) / data_size, iteration, cost_gen, sum_g_loss * 1.0 / num_g_loss, np.mean(DGZ_ret)))
            
            print ("!%d #%d\tAvgG:%f\tAvgD:%f" % ((iteration * batch_size * num_disc_steps) / data_size, iteration, sum_g_loss * 1.0 / num_g_loss, sum_d_loss * 1.0 / num_d_loss))
            if iteration % save_checkpoint_every == 0 and iteration != 0:
                save_name = "snapshots/it_%d.ckpt" % iteration
                saver.save(sess, save_name)
                print "Snapshot saved to %s" % save_name
            
            if iteration % generate_samples_every == 0:
                X_sample_all = loader.load_batch_X(100, update_iterator = False)
                im_samples, im_score = sess.run(fetches=[GZ, DGZ], feed_dict={Z:Z_sample_all})
                im_samples = save_sample_images(d3_scale(im_samples, out_range=(0,255)), 100)
                out = sess.run(im_samples)
                out = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
                cv2.imwrite("analysis/generator_sample_%d.png" % iteration, out)
                cv2.imwrite("analysis/generator_latest.png", out)
                
                X_score = sess.run(fetches=[DX], feed_dict={X:X_sample_all})
                X_samples = save_sample_images(d3_scale(X_sample_all, out_range=(0,255)), 100)
                out = sess.run(X_samples)
                out = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
                cv2.imwrite("analysis/orig_%d.png" % iteration, out)
                cv2.imwrite("analysis/orig_latest.png", out)
                
                print "Sample scores: \tDGZ:%f\tDX:%f" % (np.mean(im_score), np.mean(X_score))