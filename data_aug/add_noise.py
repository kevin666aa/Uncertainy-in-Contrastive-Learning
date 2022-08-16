import torch
import numpy as np

class GaussianNoise(object):
    def __init__(self, mean=0., std=0.1):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)



# https://stackoverflow.com/questions/14435632/impulse-gaussian-and-salt-and-pepper-noise-with-opencv
# https://stackoverflow.com/questions/22937589/how-to-add-noise-gaussian-salt-and-pepper-etc-to-image-in-python-with-opencv
class SaltAndPepperNoise(object):
    
    def __init__(self, salt_ratio=0.5, amount=0.08):
        self.salt_ratio = salt_ratio
        self.amount = amount
        
    def __call__(self, tensor):
        ch, row,col = tensor.shape
        size = row*col*ch
        # Salt mode
        num_salt = np.ceil(self.amount * size * self.salt_ratio)
        coords = [np.random.randint(0, i, int(num_salt))
                for i in tensor.shape]
        tensor[coords] = 1

        # Pepper mode
        num_pepper = np.ceil(self.amount* size * (1. - self.salt_ratio))
        coords = [np.random.randint(0, i - 1, int(num_pepper))
                for i in tensor.shape]
        tensor[coords] = 0
        return tensor
    
    def __repr__(self):
        return self.__class__.__name__ + '(Salt Ratio: {0}, Amount: {1})'.format(self.salt_ratio, self.amount)