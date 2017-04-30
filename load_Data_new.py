import numpy as np
import cv2 
from helper import normalize_batch
from tensorflow.examples.tutorials.mnist import input_data
import pdb
import scipy.io as sio
import os
import progressbar

class DataLoader:
    def __init__(self, dataset_name='MNIST'):
        print "Loading Data..."
        data_processed_path = 'data/%s/processed' % dataset_name 
        if (not os.path.isdir(data_processed_path)) or dataset_name == 'CELEBA' or dataset_name == 'REAL':
            if not os.path.isdir(data_processed_path):
                os.makedirs(data_processed_path)

            if(dataset_name == 'MNIST'):
                self.dataset = input_data.read_data_sets("data/%s/" % dataset_name)
                self.data = self.dataset.train.images
                self.labels = np.array(self.dataset.train.labels)
                self.data = self.data.reshape(-1,28,28)
                temp = np.zeros((self.data.shape[0],3,28,28))
                for i in range(3):
                    temp[:,i,:,:] = self.data.copy()
                self.data = temp 
                
            elif (dataset_name == 'CIFAR10'):
                self.data = np.empty((0,3072))
                self.labels = np.empty((0))
                for i in range(1,6):
                    unpickled = self.unpickle('data/%s/cifar-10-batches-py/data_batch_%d'% (dataset_name, i))
                    self.data = np.append(self.data, unpickled['data'],axis=0)
                    self.labels = np.append(self.labels, np.asarray(unpickled['labels']))
                self.data = self.data.reshape(-1,3,32,32)

            elif (dataset_name == 'CELEBA'):
                # num_images = 202599
                num_images = 200
                self.data = np.zeros((num_images,64,64,3))
                bar = progressbar.ProgressBar()
                for i in bar(range(1,num_images+1)):
                    img_ = cv2.imread('./data/%s/imgs/%06d.jpg'%(dataset_name, i))
                    self.data[i-1] = img_
                    # self.data[i-1] = cv2.resize(img_, None, fx=64.0/img_.shape[1], fy=64.0/img_.shape[0])
                # np.random.shuffle(self.data)
                # pdb.set_trace()
                # np.save(open(data_processed_path + '/data.npz', 'w'), self.data)

            elif (dataset_name == 'SVHN'):
                dataset = sio.loadmat('./data/%s/train_32x32.mat' % dataset_name)
                self.data = dataset['X']
                self.data = np.swapaxes(self.data, 0, 3)
                self.data = np.swapaxes(self.data, 1, 2)
                self.data = np.swapaxes(self.data, 2, 3)
                self.labels = np.reshape(dataset['y'], (dataset['y'].shape[0],))
                
            elif (dataset_name == 'REAL'):
                exts = ['.png', '.jpg']
                imgs = []
                dataset_path = './data/%s' % dataset_name
                for f in os.listdir(dataset_path):
                    ext = os.path.splitext(f)[1]
                    if ext in exts:
                        img = cv2.imread(os.path.join(dataset_path, f))
                        img = cv2.resize(img, None, fx=64.0/img.shape[1], fy=64.0/img.shape[0])
                        imgs.append(img)              
                self.data = np.array(imgs)
                self.data = np.swapaxes(self.data, 1, 2).swapaxes(1, 3)
                self.labels = np.zeros(len(imgs))
                
            print "pre processing data..."
            print "loaded data:", self.data.shape
            if(dataset_name != 'CELEBA'):
                img_swapped = np.swapaxes(self.data[:], 1, 3).swapaxes(1, 2)
                print  "swapped axes:", img_swapped.shape
                temp = np.zeros((self.data.shape[0], 64, 64, 3))
                for index in range(self.data.shape[0]):
                    temp[index] = cv2.resize(img_swapped[index], None, fx = 64.0/img_swapped.shape[1], fy = 64.0/img_swapped.shape[2])
                self.data = temp.copy()
                print "final shape:" ,self.data.shape
                p = np.random.permutation(range(self.data.shape[0]))
                self.data = self.data[p]
                self.labels = self.labels[p]
            
            if dataset_name != 'CELEBA' and dataset_name != 'REAL':
                np.savez_compressed(open(data_processed_path + '/data.npz', 'w'), data = self.data, labels = self.labels)
        else:
            npzfile = np.load(data_processed_path + '/data.npz')
            self.data = npzfile['data']
            if dataset_name != 'CELEBA':
                self.labels = npzfile['labels']

        self.curIdx = 0
        print "Data Loaded and Pre Processed Successfully!"
    
    def get_dataset_length(self):
        assert(self.data != None)
        return self.data.shape[0]
    
    def create_sample_set(self, num_per_class):
        res = np.empty((0, 64, 64, 3))
        for i in range(10):
            res = np.append(res, self.data[self.labels == i][:num_per_class], axis=0)
        return res
    
    def unpickle(self, file):
        import cPickle
        fo = open(file, 'rb')
        dict = cPickle.load(fo)
        fo.close()
        return dict
   
    def get_data_size(self):
        return self.data.shape[0]
       
    def load_batch_X(self, batch_size, update_iterator = True):
        batch_x = np.zeros((batch_size, 64, 64,3))
        length = self.data.shape[0]
        if self.curIdx + batch_size <= length:
            batch_x = self.data[self.curIdx:self.curIdx + batch_size]
        else:
            print "Unhandled reach to end of data"
            self.curIdx = 0
            return self.load_batch_X(batch_size)
        
        # TODO: Wrap around on reaching end of data     
        
        if update_iterator: self.curIdx += batch_size
        return normalize_batch(batch_x)
    
    def load_batch_Z(self, batch_size):
        batch_z = np.random.uniform(-1,1,size =(batch_size,100))
        # batch_z = np.random.randint(low=0, high=255, size = (batch_size, 100))
        # batch_z = batch_z / np.linalg.norm(batch_z)
        return batch_z
    
    def load_corrupted_batch_X(self, batch_size, random_blackout, mode = 1, update_iterator = True):
        batch_x = np.zeros((batch_size, 64, 64,3))
        length = self.data.shape[0]
        if self.curIdx + batch_size <= length:
            batch_x = self.data[self.curIdx:self.curIdx + batch_size]
        else:
            print "Unhandled reach to end of data"
            self.curIdx = 0
            return self.load_batch_X(batch_size)
        
        # TODO: Wrap around on reaching end of data     
        
        if update_iterator: self.curIdx += batch_size

        if mode == 1: # random blackout
            mask_2D = np.random.choice([0, 1], size=(batch_size,64,64), p=[random_blackout, 1-random_blackout])
        elif mode == 2: # central block mask
            mask_2D = np.ones((batch_size, 64, 64))
            mask_2D[:, 16:48, 16:48] = 0

        mask = np.zeros((batch_size, 64, 64, 3))
        mask[:, :, :, 1] = mask[:, :, :, 2] = mask[:, :, :, 0] = mask_2D

        return mask, batch_x, normalize_batch(np.multiply(batch_x, mask))
    