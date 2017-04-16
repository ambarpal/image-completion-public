import numpy as np
import cv2 
from helper import normalize_batch
from tensorflow.examples.tutorials.mnist import input_data
import pdb

class DataLoader:
    def __init__(self, dataset_name='MNIST'):
        print "Loading Data..."
        if(dataset_name == 'MNIST'):
            self.dataset = input_data.read_data_sets("data/MNIST/")
            self.data = self.dataset.train.images
            self.labels = np.array(self.dataset.train.labels)
            self.data = self.data.reshape(-1,28,28)
            temp = np.zeros((self.data.shape[0],3,28,28))
            for i in range(3):
                temp[:,i,:,:] = self.data.copy()
            self.data = temp 
            
        if(dataset_name == 'CIFAR10'):
            self.data = np.empty((0,3072))
            self.labels = np.empty((0))
            for i in range(1,6):
                unpickled = self.unpickle('data/CIFAR/cifar-10-batches-py/data_batch_%d'% i)
                self.data = np.append(self.data, unpickled['data'],axis=0)
                self.labels = np.append(self.labels, np.asarray(unpickled['labels']))
            self.data = self.data.reshape(-1,3,32,32)
        print "pre processing data..."
        print "loaded data:", self.data.shape
        img_swapped = np.swapaxes(self.data[:], 1, 3).swapaxes(1, 2)
        temp = np.zeros((self.data.shape[0], 64, 64, 3))
        print  "swapped axes:", img_swapped.shape
        for index in range(self.data.shape[0]):
            temp[index] = cv2.resize(img_swapped[index], None, fx = 64.0/img_swapped.shape[1], fy = 64.0/img_swapped.shape[2])
        self.data = temp.copy()
        print "final shape:" ,self.data.shape
        # p = np.random.permutation(range(self.data.shape[0]))
        # self.data = self.data[p]
        # self.labels = self.labels[p]
        np.random.shuffle(self.data)
        self.curIdx = 0
        print "Data Loaded and Pre Processed Successfully!"
    
    # def create_sample_set(self, num_per_class):
    #     res = np.empty((0, 64, 64, 3))
    #     for i in range(10):
    #         res = np.append(res, self.data[self.labels == i][:10], axis=0)
    #     return res
    
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
        if self.curIdx + batch_size < length:
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