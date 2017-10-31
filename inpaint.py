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

def main(args):    
    loader = load_Data_new.DataLoader(args.dataset)
    batch_size = min(args.batch_size, loader.get_dataset_length())
    
    # load Data to feed into the network
    Z_feed = np.asarray(loader.load_batch_Z(batch_size), dtype=np.float32)

    # Blacking out a fixed percentage of pixels randomly
    mask, batch_X, corrupted_batch_X = loader.load_corrupted_batch_X(batch_size, args.random_blackout, args.mode)
    
    if not os.path.isdir('analysis/inpainting/' + args.tag):
        os.makedirs('analysis/inpainting/' + args.tag)
    
    # Create network
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
        total_loss = loss_contextual + args.lbd * loss_perceptual

    #     Define optimizer to minimize total loss
    if(args.optimizer == 'adam'):
        optimizer_z = tf.train.AdamOptimizer(learning_rate=args.learning_rate, beta1 = args.beta1)
    if(args.optimizer == 'momentum'):    
        optimizer_z = tf.train.MomentumOptimizer(learning_rate=args.learning_rate, momentum=args.momentum)

    train_op_z = optimizer_z.minimize(total_loss, var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "input"))

    recon_img = tf.multiply(M, y) + tf.multiply((tf.ones_like(M) - M), GZ)
    
    #     Add summaries
    GZ_summary = tf.summary.image('GZ', GZ, max_outputs = 10)
    recon_img_summary = tf.summary.image('recon_img', recon_img, max_outputs = 10)
    DGZ_summary = tf.summary.scalar('DGZ', tf.reduce_mean(DGZ))
    loss_per_summary = tf.summary.scalar('ldb_loss_perceptual', args.lbd * loss_perceptual)
    loss_cont_summary = tf.summary.scalar('loss_contextual', loss_contextual)
    total_loss_summary = tf.summary.scalar('total_loss', total_loss)
    gen_merged = tf.summary.merge_all()

    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())

    saver_full = tf.train.Saver()
    save_full_name = 'logs/inpainting/' + args.tag + '/model.ckpt'

    write_Image(corrupted_batch_X, 'analysis/inpainting/'+args.tag+'/corrupted.png', batch_size, sess)
    write_Image(batch_X, 'analysis/inpainting/'+args.tag+'/uncorrupted.png', batch_size, sess)

    # Load checkpoint
    if not args.restore_from_full_ckpt:
        model_name = tf.train.latest_checkpoint('logs/' + args.trained_model + '/')
        variables_to_restore = [x for x in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES) if "gen" in x.name.lower() or "disc" in x.name.lower()]
        saver = tf.train.Saver(var_list=variables_to_restore)
        saver.restore(sess, model_name)
    else:
        model_full_name = tf.train.latest_checkpoint('logs/inpainting/' + args.tag + '/')
        saver_full.restore(sess, model_full_name)

    #     Write summaries to Tensorboard
    train_writer = tf.summary.FileWriter('logs/inpainting/' + args.tag, sess.graph)
    
    #     Training
    for iteration in range(args.num_train_iterations):
        _, GZ_ret, recon_img_ret, total_loss_ret, gen_merged_ret = sess.run(
            fetches=[train_op_z, GZ, recon_img, total_loss, gen_merged], 
            feed_dict={M:mask, y: corrupted_batch_X})
        train_writer.add_summary(gen_merged_ret, iteration)
        print "#%d: Loss: %f" % (iteration, total_loss_ret)
        
        if iteration % args.sample_images_every == 0:
            out = write_Image(recon_img_ret, 'analysis/inpainting/'+args.tag+'/reconstructed_%d.png'%iteration, batch_size, sess)
            cv2.imwrite('analysis/inpainting/'+args.tag+'/reconstructed_latest.png', out)

        if iteration % args.save_checkpoint_every == 0 and iteration != 0:
            saver_full.save(sess, save_full_name, iteration)
            print "Snapshot saved to %s" % save_full_name
        
    train_writer.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", help="dataset name on which inpainting has to be trained", type=str, choices=['CIFAR10', 'MNIST', 'CELEBA', 'SVHN'], default='MNIST')
    parser.add_argument("--trained_model",help="Trained DCGAN model", type=str)
    parser.add_argument("--lbd",help="(lambda)weight of perceptual loss", type=int, default=0.1)
    parser.add_argument("--mode", help="Type of blackout that can be created - [Random, box] blackout", type=int, choice=['1','2'], default=2)
    parser.add_argument("--random_blackout", help="Percentage of random blackout", type=int, choice=['1','2'], default=0)
    parser.add_argument("--tag", help="Name of the Folder where you want to save the trained model", type=str, default="INPAINT")    
    parser.add_argument("--optimizer", help="Type of optimizer", type=str, default='adam',choice=['adam', 'momentum'])
    parser.add_argument("--learning_rate", help="learning rate of the optimizer", type=float, default=0.01)
    parser.add_argument("--beta1", help="beta of adam optimizer", type=float, default=0.9)
    parser.add_argument("--momentum", help="momentum of momentum optimizer", type=float, default=0.02)
    parser.add_argument("--num_train_iterations", type=int, help="number of training iterations", default=20000)
    parser.add_argument("--batch_size", type=int, help="batch size", default=100)
    parser.add_argument("--save_checkpoint_every", type=int, help="number of iterations after which a checkpoint is saved", default=250)
    parser.add_argument("--samples_images_every", type=int, help="number of iterations after which a sample is generated", default=100)
    parser.add_argument("--restore_from_full_ckpt", type=bool, help="Restore model from a previously saved checkpoint", default=False)
    
    args, unparsed = parser.parse_known_args()
    
    main(args)