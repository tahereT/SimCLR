import numpy as np
import random
import torch

class GaussianNoise(object):
    # we assume data is normalized
    def __init__(self, mean = 0.5 , std = 0.01):
      self.mean = mean
      self.std = std
    def __call__(self , sample):
      noise = np.random.normal(self.mean, self.std, sample.shape)
      new_data = sample + noise
      return new_data

class RandomCrop(object):
    def __init__(self, n = 100):
      self.n = n
    def __call__(self, sample):
      idx = random.sample(range(0, len(sample)), self.n)
      for i in idx:
          sample[i] = 0
      return sample

class ToTensor(object):
    def __call__(self, sample):
      return torch.Tensor(sample)
