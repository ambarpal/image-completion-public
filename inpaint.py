import tensorflow as tf
import numpy as np
import load_Data_new
from models import generator,discriminator 
from helper import d3_scale, save_sample_images, write_Image
import cv2
import pdb
import os

np.random.seed(123)
tf.set_random_seed(811)

if __name__ == '__main__':    
    
    #     CONSTANTS
 
    #     GAN - PARAMETERS
    # DATASET = 'CELEBA'
    # DATASET = 'CIFAR10'
    # DATASET = 'MNIST'
    # DATASET = 'SVHN'
    DATASET_LOADED = 'CELEBA'
    DATASET = 'CELEBA'
    flip_alpha = 0.2
    MODEL_TAG = 'plotting-graphs_alpha_' + str(flip_alpha) + '_' + DATASET_LOADED
    
    #     INPAINTING PARAMETERS
    lbd = 0.1 # Final loss is loss_cont + lbd * loss_perc
    MODE = 2
    RANDOM_BLACKOUT = 0
    RUN_NAME = 'random_pixels_%0.1f' % RANDOM_BLACKOUT

    TAG = 'alpha_' + str(flip_alpha) + '_' + DATASET + '_lambda_' + str(lbd) + '_' + RUN_NAME + '_mode_' + str(MODE) + '_ MomentumOpt_lr_0.01_mom_0.9'
    loader = load_Data_new.DataLoader(DATASET)
    batch_size = min(100, loader.get_dataset_length())
    restore_from_full_ckpt = False
    
    num_train_iterations = 20000
    save_checkpoint_every = 250
    sample_images_every = 100

    #     load Data to feed into the network
    Z_feed = np.asarray(loader.load_batch_Z(batch_size), dtype=np.float32)

    # Blacking out a fixed percentage of pixels randomly
    mask, batch_X, corrupted_batch_X = loader.load_corrupted_batch_X(batch_size, RANDOM_BLACKOUT, MODE)
    
    if not os.path.isdir('analysis/inpainting/' + TAG):
        os.makedirs('analysis/inpainting/' + TAG)
    
    #     Create network
    M = tf.placeholder(tf.float32, shape = [None, 64, 64, 3])	# M is the uncorruption mask for y
    y = tf.placeholder(tf.float32, shape = [None, 64, 64, 3])	# y is the corrupted image
    
    with tf.variable_scope("input"):      
        Z = tf.Variable(Z_feed, name="Z_variable")

    with tf.variable_scope("gen"):
        GZ = generator(Z)

    with tf.variable_scope("disc") as scope:
        DGZ_raw, DGZ = discriminator(GZ)
        scope.reuse_variables()

    with tf.variable_scope("loss_calculation"):
        loss_g = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(DGZ_raw), logits=DGZ_raw)
        # log(DGZ_raw)
        loss_perceptual = tf.reduce_mean(loss_g)
        uncorr_gz = tf.multiply(M, GZ)
        uncorr_y = tf.multiply(M, y)
        # ||M.y - M.G(z)||_1
        loss_contextual = tf.reduce_mean(tf.norm(uncorr_gz - uncorr_y, ord = 1, axis=0))
        total_loss = loss_contextual + lbd * loss_perceptual

    #     Define optimizer to minimize total loss
    # optimizer_z = tf.train.AdamOptimizer(learning_rate=2e-4, beta1 = 0.5)
    # optimizer_z = tf.train.AdamOptimizer()
    optimizer_z = tf.train.MomentumOptimizer(learning_rate=0.01, momentum=0.9)
    train_op_z = optimizer_z.minimize(total_loss, var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "input"))

    recon_img = tf.multiply(M, y) + tf.multiply((tf.ones_like(M) - M), GZ)
    
    #     Add summaries
    GZ_summary = tf.summary.image('GZ', GZ, max_outputs = 10)
    recon_img_summary = tf.summary.image('recon_img', recon_img, max_outputs = 10)
    DGZ_summary = tf.summary.scalar('DGZ', tf.reduce_mean(DGZ))
    loss_per_summary = tf.summary.scalar('ldb_loss_perceptual', lbd * loss_perceptual)
    loss_cont_summary = tf.summary.scalar('loss_contextual', loss_contextual)
    total_loss_summary = tf.summary.scalar('total_loss', total_loss)
    gen_merged = tf.summary.merge_all()

    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())

    saver_full = tf.train.Saver()
    save_full_name = 'logs/inpainting/' + TAG + '/model.ckpt'

    write_Image(corrupted_batch_X, 'analysis/inpainting/'+TAG+'/corrupted.png', batch_size, sess)
    write_Image(batch_X, 'analysis/inpainting/'+TAG+'/uncorrupted.png', batch_size, sess)

    # Load checkpoint
    if not restore_from_full_ckpt:
        model_name = tf.train.latest_checkpoint('logs/' + MODEL_TAG + '/')
        variables_to_restore = [x for x in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES) if "gen" in x.name.lower() or "disc" in x.name.lower()]
        saver = tf.train.Saver(var_list=variables_to_restore)
        saver.restore(sess, model_name)
    else:
        model_full_name = tf.train.latest_checkpoint('logs/inpainting/' + TAG + '/')
        saver_full.restore(sess, model_full_name)

    #     Write summaries to Tensorboard
    train_writer = tf.summary.FileWriter('logs/inpainting/' + TAG, sess.graph)
    
    #     Training
    for iteration in range(num_train_iterations):
        _, GZ_ret, recon_img_ret, total_loss_ret, gen_merged_ret = sess.run(
            fetches=[train_op_z, GZ, recon_img, total_loss, gen_merged], 
            feed_dict={M:mask, y: corrupted_batch_X})
        train_writer.add_summary(gen_merged_ret, iteration)
        print "#%d: Loss: %f" % (iteration, total_loss_ret)
        
        if iteration % sample_images_every == 0:
            out = write_Image(recon_img_ret, 'analysis/inpainting/'+TAG+'/reconstructed_%d.png'%iteration, batch_size, sess)
            cv2.imwrite('analysis/inpainting/'+TAG+'/reconstructed_latest.png', out)

        if iteration % save_checkpoint_every == 0 and iteration != 0:
            saver_full.save(sess, save_full_name, iteration)
            print "Snapshot saved to %s" % save_full_name
        
    train_writer.close()