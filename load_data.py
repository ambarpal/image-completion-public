import numpy as np
import cv2 
from helper import normalize_batch
from tensorflow.examples.tutorials.mnist import input_data

class DataLoaderCIFAR:
    def __init__(self):
        self.file_to_batch_index = {}
        self.file_to_data_dict = {}
        
    def unpickle(self, file):
        import cPickle
        fo = open(file, 'rb')
        dict = cPickle.load(fo)
        fo.close()
        return dict
    
    def register_data_file(self, file_name):
        self.file_to_batch_index[file_name] = 0
        self.file_to_data_dict[file_name] = self.unpickle(file_name)        
        
    def load_batch_X(self, file_name, batch_size):
        assert(file_name in self.file_to_data_dict)
        batch_x_32 = np.zeros((batch_size, 3, 32, 32))
        
        curIdx = self.file_to_batch_index[file_name]
        length = self.file_to_data_dict[file_name]['data'].shape[0]
        if curIdx + batch_size < length:
            batch_x_32 = self.file_to_data_dict[file_name]['data'][curIdx:curIdx + batch_size].reshape((batch_size, 3, 32, 32))
        else:
            print "Unhandled reach to end of data"
            self.file_to_batch_index[file_name] = 0
            return self.load_batch_X(file_name, batch_size)
        
        # TODO: Wrap around on reaching end of data
        # else:
            # batch_x_32 = 
        
        batch_x_64 = np.zeros((batch_size, 64, 64, 3))
        for index in range(batch_size):
            img_swapped = np.swapaxes(batch_x_32[index], 0, 2).swapaxes(0, 1)
            batch_x_64[index] = cv2.resize(img_swapped, None, fx = 2, fy = 2)
        self.file_to_batch_index[file_name]+=batch_size
        return normalize_batch(batch_x_64)
    
    def load_batch_Z(self, batch_size):
        batch_z = np.random.uniform(-1,1,size =(batch_size,100))
        # batch_z = np.random.randint(low=0, high=255, size = (batch_size, 100))
        # batch_z = batch_z / np.linalg.norm(batch_z)
        return batch_z
    
class DataLoaderMNIST:
    def __init__(self):
        self.dataset = input_data.read_data_sets("data/MNIST/")
        self.curIdx = 0
    
    def register_data_file(self, file_name):
       '''
       '''
       
    def load_batch_X(self, file_name, batch_size):
        batch_x_32 = np.zeros((batch_size, 3, 32, 32))
        length = self.dataset.train.images.shape[0]
        if self.curIdx + batch_size < length:
            temp = self.dataset.train.images[self.curIdx:self.curIdx + batch_size].reshape((batch_size, 28, 28))
            batch_x_32 = np.zeros((batch_size,3,28,28))
            for i in range(3):
                batch_x_32[:,i,:,:] = temp.copy()
            
            
        else:
            print "Unhandled reach to end of data"
            self.curIdx = 0
            return self.load_batch_X(file_name, batch_size)
        
        # TODO: Wrap around on reaching end of data
        
        batch_x_64 = np.zeros((batch_size, 64, 64, 3))
        for index in range(batch_size):
            img_swapped = np.swapaxes(batch_x_32[index], 0, 2).swapaxes(0, 1)
            batch_x_64[index] = cv2.resize(img_swapped, None, fx = 64.0/28.0, fy = 64.0/28.0)
        self.curIdx+=batch_size
        return normalize_batch(batch_x_64)
    
    def load_batch_Z(self, batch_size):
        batch_z = np.random.uniform(-1,1,size =(batch_size,100))
        # batch_z = np.random.randint(low=0, high=255, size = (batch_size, 100))
        # batch_z = batch_z / np.linalg.norm(batch_z)
        return batch_z