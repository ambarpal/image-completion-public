import tensorflow as tf
import numpy as np
import load_Data_new
from models import generator,discriminator 
from helper import d3_scale, save_sample_images
import cv2
import pdb
import os

np.random.seed(123)
tf.set_random_seed(811)

def write_Image(img, path, batch_size, sess):
    im_samples = d3_scale(img, out_range=(0, 255))
    im_samples = save_sample_images(im_samples, batch_size)
    out = sess.run(im_samples)
    cv2.imwrite(path, out)
    return out

if __name__ == '__main__':    
    lbd = 0.5
    
    DATASET = 'MNIST'
    RANDOM_BLACKOUT = 0.3
    RUN_NAME = 'random_pixels_%0.1f' % RANDOM_BLACKOUT
    flip_alpha = 0.3
    MODEL_TAG = 'plotting-graphs_alpha_' + str(flip_alpha) + '_' + DATASET
    TAG = 'alpha_' + str(flip_alpha) + '_' + DATASET + '_lambda_' + str(lbd) + '_' + RUN_NAME
    batch_size = 100
    
    loader = load_Data_new.DataLoader(DATASET)
    Z_feed = np.asarray(loader.load_batch_Z(batch_size), dtype=np.float32)

    # Blacking out a fixed percentage of pixels randomly
    Mask = np.random.choice([0, 1], size=(batch_size,64,64,3), p=[RANDOM_BLACKOUT, 1-RANDOM_BLACKOUT])
    Mask[:, :, :, 1] = Mask[:, :, :, 2] = Mask[:, :, :, 0]

    corruptedImage = loader.load_batch_X(batch_size) * Mask
    
    if not os.path.isdir('analysis/inpainting/' + TAG):
        os.makedirs('analysis/inpainting/' + TAG)
          
    Z = tf.Variable(Z_feed, name="Z_variable")
    M = tf.placeholder(tf.float32, shape = [None, 64, 64, 3])	# M is the uncorruption mask for y
    y = tf.placeholder(tf.float32, shape = [None, 64, 64, 3])	# y is the corrupted image

    with tf.variable_scope("gen"):
        GZ = generator(Z)

    with tf.variable_scope("disc") as scope:
        DGZ_raw, DGZ = discriminator(GZ)
        scope.reuse_variables()

    with tf.variable_scope("loss_calculation"):
        loss_g = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(DGZ_raw), logits=DGZ_raw)
        loss_perceptual = tf.reduce_mean(loss_g)
        uncorr_gz = tf.multiply(M, GZ)
        uncorr_y = tf.multiply(M, y)
        loss_contextual = tf.norm(uncorr_gz - uncorr_y, ord = 1)
        total_loss = loss_contextual + lbd * loss_perceptual

    optimizer_z = tf.train.AdamOptimizer(learning_rate=2e-4, beta1 = 0.5)
    train_op_z = optimizer_z.minimize(total_loss, var_list=[Z])

    recon_img = tf.multiply(M, y) + tf.multiply((tf.ones_like(M) - M), GZ)
    GZ_summary = tf.summary.image('GZ', GZ, max_outputs = 10)
    recon_img_summary = tf.summary.image('recon_img', recon_img, max_outputs = 10)
    DGZ_summary = tf.summary.scalar('DGZ', tf.reduce_mean(DGZ))
    loss_per_summary = tf.summary.scalar('loss_perceptual', loss_perceptual)
    loss_cont_summary = tf.summary.scalar('loss_contextual', loss_contextual)
    gen_merged = tf.summary.merge_all()

    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    
    write_Image(corruptedImage, 'analysis/inpainting/'+TAG+'/corrupted.png', batch_size, sess)

    train_writer = tf.summary.FileWriter('logs/inpainting/' + TAG, sess.graph)
    model_name = tf.train.latest_checkpoint('logs/' + MODEL_TAG + '/')
    variables_to_restore = [x for x in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES) if "gen" in x.name.lower() or "disc" in x.name.lower()]
    print len(variables_to_restore)
    saver = tf.train.Saver(var_list=variables_to_restore)
    saver.restore(sess, model_name)

    num_train_iterations = 10000
    for iteration in range(num_train_iterations):
        GZ_ret, recon_img_ret, total_loss_ret, gen_merged_ret = sess.run(fetches=[GZ, recon_img, total_loss, gen_merged], feed_dict={M:Mask, y: corruptedImage})
        train_writer.add_summary(gen_merged_ret, iteration)
        print "#%d: Loss: %d" % (iteration, total_loss_ret)
        
        out = write_Image(recon_img_ret, 'analysis/inpainting/'+TAG+'/reconstructed_%d.png'%iteration, batch_size, sess)
        cv2.imwrite('analysis/inpainting/'+TAG+'/reconstructed_latest.png', out)
        
    train_writer.close()