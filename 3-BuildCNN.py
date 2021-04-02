from numpy.core.numeric import correlate
import torch as t
from torch.utils.data import DataLoader
import torchvision as tv

class CNN_Net(t.nn.Module):
    def __init__(self):
        super(CNN_Net,self).__init__()
        self.cnn_layers=t.nn.Sequential(
            t.nn.Conv2d(in_channels=1,out_channels=8,kernel_size=3,padding=1,stride=1),
            t.nn.MaxPool2d(kernel_size)
        )