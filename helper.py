import numpy as np
import tensorflow as tf

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

def lrelu(x=None, alpha=0.2, name="LeakyReLU"):
    with tf.name_scope(name) as scope:
        # x = tf.nn.relu(x)
        # m_x = tf.nn.relu(-x)
        # x -= alpha * m_x
        x = tf.maximum(x, alpha * x)
    return x

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

def normalize_batch(x, out_range=(-1, 1)):
    res = d3_scale(x, out_range=out_range)
    # res = x / 255.0
    # print res
    return res

def save_sample_images(images, size, n_row = 10, n_col = 10):
    images = [image for image in tf.split(images.astype(np.uint8), size, axis = 0)]
    rows = []
    for i in range(n_row):
        rows.append(tf.concat(images[n_col*i + 0: n_col*i + n_col],2))
    return tf.squeeze(tf.concat(rows,1),[0])