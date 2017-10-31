import numpy as np
import tensorflow as tf
import cv2
import pdb
'''
 This function evaluates your trained model, where Xsamples represent images
 from your original data distribution and genSample are images generated using 
 the GAN. For each image in genImage, this function finds the closest image in
 the original data distribution by computing the euclidean distance between 
 the two. We then return the mean of all the minDistance for each image in gen_Images 
'''
def evaluate(Xsamples, gen_Images):
    dists = []
    for y in gen_Images:
        minDist = None
        for x in Xsamples:
            dist = np.linalg.norm(x-y)
            if(dist < minDist or minDist is None):
                minDist = dist
        dists += [minDist]
    return np.mean(np.array(dists))
'''
 Leaky relu activation function
'''
def lrelu(x=None, alpha=0.2, name="LeakyReLU"):
    with tf.name_scope(name) as scope:
        x = tf.maximum(x, alpha * x)
    return x
'''
 This function scales the input Image (out) to the given range specified by the input argument (out_range)
'''
def d3_scale(dat, out_range=(-1, 1)):
    domain = [np.min(dat), np.max(dat)]

    def interp(x):
        return out_range[0] * (1.0 - x) + out_range[1] * x

    def uninterp(x):
        b = 0
        if (domain[1] - domain[0]) != 0:
            b = domain[1] - domain[0]
        else:
            b =  1.0 / domain[1]
        return (x - domain[0]) / b

    return interp(uninterp(dat))
'''
 This funciton normalizes the Input dataset images (x) to the range (-1, 1)
'''
def normalize_batch(x, out_range=(-1, 1)):
    res = d3_scale(x, out_range=out_range)
    return res

'''
 This function saves the images in a tile format 
'''
def save_sample_images(images, size, n_row = 10, n_col = 10):
    if size < n_row * n_col:
        sq_size = 1
        while sq_size * sq_size < size:
            sq_size += 1
        left = sq_size ** 2 - images.shape[0]
        for ind in range(left):
            images = np.append(images, np.zeros((1, 64, 64, 3)), axis = 0)
        n_row = sq_size
        n_col = sq_size
        size = sq_size**2
    images = [image for image in tf.split(images.astype(np.uint8), size, axis = 0)]
    rows = []
    for i in range(n_row):
        rows.append(tf.concat(images[n_col*i + 0: n_col*i + n_col],2))
    return tf.squeeze(tf.concat(rows,1),[0])

'''
 This function rescales the image back to the range 0-255 and then writes converts 
 the images to tile format and then writes that to disk
'''
def write_Image(img, path, batch_size, sess):
    im_samples = d3_scale(img, out_range=(0, 255))
    im_samples = save_sample_images(im_samples, batch_size)
    out = sess.run(im_samples)
    cv2.imwrite(path, out)
    return out