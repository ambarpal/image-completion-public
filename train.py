import tensorflow as tf
import numpy as np
import load_Data_new
from models import generator,discriminator 
from helper import d3_scale, save_sample_images, evaluate
import cv2
import pdb
import os

np.random.seed(123)
tf.set_random_seed(811)

# restore = True
restore = False

    # Do not set scopes with the loss as a name as we need to share the discriminator instead
    # of making multiple copies

if __name__ == '__main__':
#     Create Placeholders for Data
    X = tf.placeholder(tf.float32, shape = [None, 64, 64, 3])
    Z = tf.placeholder(tf.float32, shape = [None, 100])
    labels_d_fake = tf.placeholder(tf.float32)
    labels_d_real = tf.placeholder(tf.float32)
    
    with tf.variable_scope("gen"):
        GZ = generator(Z)
    
    with tf.variable_scope("disc") as scope:
        DGZ_raw, DGZ = discriminator(GZ)
        scope.reuse_variables()
        DX_raw, DX = discriminator(X)
    
    with tf.variable_scope("loss_calculation"):
        # using Sigmoid Cross Entropy as loss function
        # discriminator tries to discriminate between X and GZ:
        #  if input to discriminator is X then output prob should be 0
        #  if input to discriminator is GZ then output prob should be 1         
        # loss_d_fake = log(1-DGZ_raw)        
        loss_d_fake = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels_d_fake, logits=DGZ_raw)
        # loss_d_real = log(DX_raw)         
        loss_d_real = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels_d_real, logits=DX_raw)
        loss_g = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(DGZ_raw), logits=DGZ_raw)
        # loss_disc = log(1-DGZ_raw) + log(DX_raw)         
        loss_disc = tf.reduce_mean(loss_d_fake + loss_d_real)
        # gen tries to fool D
        #  that is D should output 0 on seeing GZ
        #  therefore we minimize DGZ-raw(to make it 0)
        # loss_gen = log(DGZ_raw)        
        loss_gen = tf.reduce_mean(loss_g)

    GZ_summary = tf.summary.image('GZ', GZ, max_outputs = 10)
    DGZ_summary = tf.summary.scalar('DGZ', tf.reduce_mean(DGZ))
    DX_summary = tf.summary.scalar('DX', tf.reduce_mean(DX))    
    loss_disc_summary = tf.summary.scalar('loss_disc', loss_disc)
    loss_gen_summary = tf.summary.scalar('loss_gen', loss_gen)
    
    disc_merged = tf.summary.merge([DGZ_summary, DX_summary, loss_disc_summary, GZ_summary])
    gen_merged = tf.summary.merge([DGZ_summary, loss_gen_summary])
        
#        Defined Optimizers for both Discriminator and Generator
    optimizer_disc = tf.train.AdamOptimizer(learning_rate=2e-4, beta1 = 0.5)
    train_op_disc = optimizer_disc.minimize(loss_disc, var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "disc"))
    
    optimizer_gen = tf.train.AdamOptimizer(learning_rate=2e-4, beta1 = 0.5)
    train_op_gen = optimizer_gen.minimize(loss_gen, var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "gen"))
        
    # DATASET = 'CIFAR10'
    DATASET = 'MNIST'
    # DATASET = 'CELEBA'
    # DATASET = 'SVHN'
    
    loader = load_Data_new.DataLoader(DATASET)
    
    data_size = loader.get_data_size()
    if DATASET == 'MNIST':
        num_train_epochs = 10
        num_disc_steps = 1
        num_gen_steps = 2
        batch_size = 64
        save_checkpoint_every = 250
        generate_samples_every = 100
        flip_alpha = 0.3

    elif DATASET == 'CIFAR10':
        num_train_epochs = 50
        num_disc_steps = 1
        num_gen_steps = 1
        batch_size = 64
        save_checkpoint_every = 250
        generate_samples_every = 100
        flip_alpha = 0.2
    
    elif DATASET == 'CELEBA':
        num_train_epochs = 15
        num_disc_steps = 1
        num_gen_steps = 1
        batch_size = 64
        save_checkpoint_every = 250
        generate_samples_every = 100
        flip_alpha = 0.2
        
    elif DATASET == 'SVHN':
        num_train_epochs = 15
        num_disc_steps = 1
        num_gen_steps = 1
        batch_size = 64
        save_checkpoint_every = 250
        generate_samples_every = 100
        flip_alpha = 0.3

    TAG = 'plotting-graphs_alpha_'+str(flip_alpha)+'_'+DATASET
    
    saver = tf.train.Saver(max_to_keep=2)
    
    if(DATASET != 'CELEBA'):
        Xsample_val_set = loader.create_sample_set(100)
    
    sess = tf.InteractiveSession()
    # saver = tf.train.import_meta_graph('saved_analysis_snapshots/snapshots_cifar_complete_from_60_epoch_G_twice/it_2500.ckpt.meta')
    # saver.restore(sess, 'saved_analysis_snapshots/snapshots_cifar_complete_from_60_epoch_G_twice/it_2500.ckpt')
    
    train_writer = tf.summary.FileWriter('logs/' + TAG, sess.graph)

    if(restore):
        snapshot_name = tf.train.latest_checkpoint('logs/' + TAG + '/')
        saver = tf.train.import_meta_graph(snapshot_name + '.meta')
        print snapshot_name
        start_iter = int(snapshot_name.split('-')[-1].split('.')[0]) + 1
        print start_iter
        saver.restore(sess, snapshot_name)
        
    else:
        start_iter = 0
        sess.run(tf.global_variables_initializer())
    
    num_train_iter = num_train_epochs * (data_size / (batch_size * num_disc_steps))
    
    Z_sample_all = loader.load_batch_Z(100)
    sum_g_loss, num_g_loss = 0, 0
    sum_d_loss, num_d_loss = 0, 0
    
    for iteration in range(start_iter, num_train_iter):
        for disc_step in range(num_disc_steps):
            X_feed = loader.load_batch_X(batch_size)
            Z_feed = loader.load_batch_Z(batch_size)
            labels_d_real_feed = np.random.choice([0, 1], size=(batch_size,), p=[flip_alpha, 1-flip_alpha])
            labels_d_fake_feed = np.ones_like(labels_d_real_feed) - labels_d_real_feed
            _, cost_disc, DGZ_ret, DX_ret, disc_merged_ret = sess.run(fetches = [train_op_disc, loss_disc, DGZ, DX, disc_merged], feed_dict = {X:X_feed, Z:Z_feed, labels_d_fake: labels_d_fake_feed, labels_d_real:labels_d_real_feed})
            sum_d_loss += cost_disc
            num_d_loss += 1
            train_writer.add_summary(disc_merged_ret, iteration)
            # print ("#%d D #%d\tLossD:%f\tAvgD:%f\tDGZ:%f\tDX:%f" % ((iteration * batch_size * num_disc_steps) / data_size , iteration, cost_disc, sum_d_loss * 1.0 / num_d_loss, np.mean(DGZ_ret), np.mean(DX_ret)))  
            
        for gen_step in range(num_gen_steps):
            Z_feed = loader.load_batch_Z(batch_size)
            _, cost_gen, DGZ_ret, gen_merged_ret = sess.run(fetches = [train_op_gen, loss_gen, DGZ, gen_merged], feed_dict = {Z:Z_feed})
            sum_g_loss += cost_gen
            num_g_loss += 1
            train_writer.add_summary(gen_merged_ret, iteration)
            # print ("#%d G #%d\tLossG:%f\tAvgG:%f\tDGZ:%f" % ((iteration * batch_size * num_disc_steps) / data_size, iteration, cost_gen, sum_g_loss * 1.0 / num_g_loss, np.mean(DGZ_ret)))
        
        
        print ("!%d #%d\tAvgG:%f\tAvgD:%f" % ((iteration * batch_size * num_disc_steps) / data_size, iteration, sum_g_loss * 1.0 / num_g_loss, sum_d_loss * 1.0 / num_d_loss))
        if iteration % save_checkpoint_every == 0 and iteration != 0:
            # save_name = "snapshots/it_%d.ckpt" % iteration
            if not os.path.isdir('logs/' + TAG):
                os.makedirs('logs/' + TAG)
            save_name = 'logs/' + TAG + '/model.ckpt'
            saver.save(sess, save_name, iteration)
            print "Snapshot saved to %s" % save_name
        
        if iteration % generate_samples_every == 0:
            if not os.path.isdir('analysis/' + TAG):
                os.makedirs('analysis/' + TAG)
            X_sample_all = loader.load_batch_X(100, update_iterator = False)
            im_samples, im_score = sess.run(fetches=[GZ, DGZ], feed_dict={Z:Z_sample_all})
            im_samples = d3_scale(im_samples, out_range=(0,255))
            # print "evaluation : %d" % evaluate(Xsample_val_set, im_samples) 
            im_samples = save_sample_images(im_samples, 100)
            
            out = sess.run(im_samples)
            if DATASET != 'CELEBA':
                out = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
            cv2.imwrite("analysis/"+TAG+"/generator_sample_%d.png" % iteration, out)
            cv2.imwrite("analysis/"+TAG+"/generator_latest.png", out)
            
            X_score = sess.run(fetches=[DX], feed_dict={X:X_sample_all})
            X_samples = save_sample_images(d3_scale(X_sample_all, out_range=(0,255)), 100)
            out = sess.run(X_samples)
            if DATASET != 'CELEBA':
                out = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
            cv2.imwrite("analysis/"+TAG+"/orig_%d.png" % iteration, out)
            cv2.imwrite("analysis/"+TAG+"/orig_latest.png", out)
            
            print "Sample scores: \tDGZ:%f\tDX:%f" % (np.mean(im_score), np.mean(X_score))

    train_writer.close()