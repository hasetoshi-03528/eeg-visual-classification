import scipy.io
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
import matplotlib.pyplot as plt
from torchvision import transforms

class EEGNet(nn.Module):
    def __init__(self, classes_num):
        super(EEGNet, self).__init__()
        self.drop_out = 0.25

        self.block_1 = nn.Sequential(
            # Pads the input tensor boundaries with zero
            # left, right, up, bottom
            nn.ZeroPad2d((31, 32, 0, 0)),
            nn.Conv2d(
                in_channels=1,  # input shape (1, C, T)
                out_channels=8,  # num_filters
                kernel_size=(1, 63),  # filter size
                bias=False
            ),  # output shape (8, C, T)
            nn.BatchNorm2d(8)  # output shape (8, C, T)
        )

        # block 2 and 3 are implementations of Depthwise Convolution and Separable Convolution
        self.block_2 = nn.Sequential(
            nn.Conv2d(
                in_channels=8,  # input shape (8, C, T)
                out_channels=16,  # num_filters
                kernel_size=(1, 22),  # filter size
                groups=8,
                bias=False
            ),  # output shape (16, 1, T)
            nn.BatchNorm2d(16),  # output shape (16, 1, T)
            nn.ELU(),
            nn.AvgPool2d((1, 4)),  # output shape (16, 1, T//4)
            nn.Dropout(self.drop_out)  # output shape (16, 1, T//4)
        )

        self.block_3 = nn.Sequential(
            nn.ZeroPad2d((7, 8, 0, 0)),
            nn.Conv2d(
                in_channels=16,  # input shape (16, 1, T//4)
                out_channels=16,  # num_filters
                kernel_size=(1, 16),  # filter size
                groups=16,
                bias=False
            ),  # output shape (16, 1, T//4)
            nn.Conv2d(
                in_channels=16,  # input shape (16, 1, T//4)
                out_channels=16,  # num_filters
                kernel_size=(1, 1),  # filter size
                bias=False
            ),  # output shape (16, 1, T//4)
            nn.BatchNorm2d(16),  # output shape (16, 1, T//4)
            nn.ELU(),
            nn.AvgPool2d((1, 8)),  # output shape (16, 1, T//32)
            nn.Dropout(self.drop_out)
        )

        self.out = nn.Linear((16 * 500), classes_num)

    def forward(self, x):
        x = self.block_1(x)
        #print("block1", x.shape)
        x = self.block_2(x)
        #print("block2", x.shape)
        x = self.block_3(x)
        #print("block3", x.shape)

        x = x.view(x.size(0), -1)
        x = self.out(x)
        return F.softmax(x, dim=1), x  # return x for visualization

def resort(arr):
    return np.array([arr[..., item] for item in range(arr.shape[2])])

def data_re_1(filename):
    data0 = resort(scipy.io.loadmat(filename)['epochs_data'])
    data0_1 = data0.reshape([-1, 500, 63])
    return data0_1

def data_re_2(filename):
    data3 = resort(scipy.io.loadmat(filename)['epochs_data'])
    data3_1 = np.delete(data3,63,2)
    data3_2 = data3_1.reshape([-1, 500, 63])
    return data3_2

def mingci(data0, data1, data2, data3, data4, data5, data6, data7, data8, data9, data10, data11, data12, data13, data14, data15):
    data20 = np.vstack((data0, data1))
    data21 = np.vstack((data20, data2))
    data22 = np.vstack((data21, data3))
    data23 = np.vstack((data22, data4))
    data24 = np.vstack((data23, data5))
    data25 = np.vstack((data24, data6))
    data26 = np.vstack((data25, data7))
    data27 = np.vstack((data26, data8))
    data28 = np.vstack((data27, data9))
    data29 = np.vstack((data28, data10))
    data30 = np.vstack((data29, data11))
    data31 = np.vstack((data30, data12))
    data32 = np.vstack((data31, data13))
    data33 = np.vstack((data32, data14))
    data34 = np.vstack((data33, data15))
    return data34

def data_mingci(dir_name, file):
    # 0416-1
    filename = dir_name+"0416-1-ver1-2-1-"+file+"0.mat"
    data_0=data_re_1(filename)
    filename = dir_name+"0416-1-ver1-2-1-"+file+"1.mat"
    data_1=data_re_1(filename)
    filename = dir_name+"0416-1-ver1-2-1-"+file+"2.mat"
    data_2=data_re_1(filename)
    filename = dir_name+"0416-1-ver1-2-1-"+file+"3.mat"
    data_3=data_re_1(filename)
    filename = dir_name+"0416-1-ver1-2-1-"+file+"4.mat"
    data_4=data_re_1(filename)
    filename = dir_name+"0416-1-ver1-2-1-"+file+"5.mat"
    data_5=data_re_1(filename)
    filename = dir_name+"0416-1-ver1-2-1-"+file+"6.mat"
    data_6=data_re_1(filename)
    filename = dir_name+"0416-1-ver1-2-1-"+file+"7.mat"
    data_7=data_re_1(filename)
    filename = dir_name+"0416-1-ver1-2-1-"+file+"8.mat"
    data_8=data_re_1(filename)
    filename = dir_name+"0416-1-ver1-2-1-"+file+"9.mat"
    data_9=data_re_1(filename)
    filename = dir_name+"0416-1-ver1-2-1-"+file+"10.mat"
    data_10=data_re_1(filename)
    filename = dir_name+"0416-1-ver1-2-1-"+file+"11.mat"
    data_11=data_re_1(filename)
    filename = dir_name+"0416-1-ver1-2-1-"+file+"12.mat"
    data_12=data_re_1(filename)
    filename = dir_name+"0416-1-ver1-2-1-"+file+"13.mat"
    data_13=data_re_1(filename)
    filename = dir_name+"0416-1-ver1-2-1-"+file+"14.mat"
    data_14=data_re_1(filename)
    filename = dir_name+"0416-1-ver1-2-1-"+file+"15.mat"
    data_15=data_re_1(filename)
    data0416_1=mingci(data_0, data_1, data_2, data_3, data_4, data_5, data_6, data_7, data_8, data_9, data_10, data_11, data_12, data_13, data_14, data_15)
    # 0417-1
    filename = dir_name+"0417-1-ver1-2-1-"+file+"0.mat"
    data_0=data_re_1(filename)
    filename = dir_name+"0417-1-ver1-2-1-"+file+"1.mat"
    data_1=data_re_1(filename)
    filename = dir_name+"0417-1-ver1-2-1-"+file+"2.mat"
    data_2=data_re_1(filename)
    filename = dir_name+"0417-1-ver1-2-1-"+file+"3.mat"
    data_3=data_re_1(filename)
    filename = dir_name+"0417-1-ver1-2-1-"+file+"4.mat"
    data_4=data_re_1(filename)
    filename = dir_name+"0417-1-ver1-2-1-"+file+"5.mat"
    data_5=data_re_1(filename)
    filename = dir_name+"0417-1-ver1-2-1-"+file+"6.mat"
    data_6=data_re_1(filename)
    filename = dir_name+"0417-1-ver1-2-1-"+file+"7.mat"
    data_7=data_re_1(filename)
    filename = dir_name+"0417-1-ver1-2-1-"+file+"8.mat"
    data_8=data_re_1(filename)
    filename = dir_name+"0417-1-ver1-2-1-"+file+"9.mat"
    data_9=data_re_1(filename)
    filename = dir_name+"0417-1-ver1-2-1-"+file+"10.mat"
    data_10=data_re_1(filename)
    filename = dir_name+"0417-1-ver1-2-1-"+file+"11.mat"
    data_11=data_re_1(filename)
    filename = dir_name+"0417-1-ver1-2-1-"+file+"12.mat"
    data_12=data_re_1(filename)
    filename = dir_name+"0417-1-ver1-2-1-"+file+"13.mat"
    data_13=data_re_1(filename)
    filename = dir_name+"0417-1-ver1-2-1-"+file+"14.mat"
    data_14=data_re_1(filename)
    filename = dir_name+"0417-1-ver1-2-1-"+file+"15.mat"
    data_15=data_re_1(filename)

    data0417_1=mingci(data_0, data_1, data_2, data_3, data_4, data_5, data_6, data_7, data_8, data_9, data_10, data_11, data_12, data_13, data_14, data_15)
    
    # 0417-2
    filename = dir_name+"0417-2-ver1-2-1-"+file+"0.mat"
    data_0=data_re_1(filename)
    filename = dir_name+"0417-2-ver1-2-1-"+file+"1.mat"
    data_1=data_re_1(filename)
    filename = dir_name+"0417-2-ver1-2-1-"+file+"2.mat"
    data_2=data_re_1(filename)
    filename = dir_name+"0417-2-ver1-2-1-"+file+"3.mat"
    data_3=data_re_1(filename)
    filename = dir_name+"0417-2-ver1-2-1-"+file+"4.mat"
    data_4=data_re_1(filename)
    filename = dir_name+"0417-2-ver1-2-1-"+file+"5.mat"
    data_5=data_re_1(filename)
    filename = dir_name+"0417-2-ver1-2-1-"+file+"6.mat"
    data_6=data_re_1(filename)
    filename = dir_name+"0417-2-ver1-2-1-"+file+"7.mat"
    data_7=data_re_1(filename)
    filename = dir_name+"0417-2-ver1-2-1-"+file+"8.mat"
    data_8=data_re_1(filename)
    filename = dir_name+"0417-2-ver1-2-1-"+file+"9.mat"
    data_9=data_re_1(filename)
    filename = dir_name+"0417-2-ver1-2-1-"+file+"10.mat"
    data_10=data_re_1(filename)
    filename = dir_name+"0417-2-ver1-2-1-"+file+"11.mat"
    data_11=data_re_1(filename)
    filename = dir_name+"0417-2-ver1-2-1-"+file+"12.mat"
    data_12=data_re_1(filename)
    filename = dir_name+"0417-2-ver1-2-1-"+file+"13.mat"
    data_13=data_re_1(filename)
    filename = dir_name+"0417-2-ver1-2-1-"+file+"14.mat"
    data_14=data_re_1(filename)
    filename = dir_name+"0417-2-ver1-2-1-"+file+"15.mat"
    data_15=data_re_1(filename)
    data0417_2=mingci(data_0, data_1, data_2, data_3, data_4, data_5, data_6, data_7, data_8, data_9, data_10, data_11, data_12, data_13, data_14, data_15)
    
     # 0418-1
    filename = dir_name+"0418-1-ver1-2-1-"+file+"0.mat"
    data_0=data_re_2(filename)
    filename = dir_name+"0418-1-ver1-2-1-"+file+"1.mat"
    data_1=data_re_2(filename)
    filename = dir_name+"0418-1-ver1-2-1-"+file+"2.mat"
    data_2=data_re_2(filename)
    filename = dir_name+"0418-1-ver1-2-1-"+file+"3.mat"
    data_3=data_re_2(filename)
    filename = dir_name+"0418-1-ver1-2-1-"+file+"4.mat"
    data_4=data_re_2(filename)
    filename = dir_name+"0418-1-ver1-2-1-"+file+"5.mat"
    data_5=data_re_2(filename)
    filename = dir_name+"0418-1-ver1-2-1-"+file+"6.mat"
    data_6=data_re_2(filename)
    filename = dir_name+"0418-1-ver1-2-1-"+file+"7.mat"
    data_7=data_re_2(filename)
    filename = dir_name+"0418-1-ver1-2-1-"+file+"8.mat"
    data_8=data_re_2(filename)
    filename = dir_name+"0418-1-ver1-2-1-"+file+"9.mat"
    data_9=data_re_2(filename)
    filename = dir_name+"0418-1-ver1-2-1-"+file+"10.mat"
    data_10=data_re_2(filename)
    filename = dir_name+"0418-1-ver1-2-1-"+file+"11.mat"
    data_11=data_re_2(filename)
    filename = dir_name+"0418-1-ver1-2-1-"+file+"12.mat"
    data_12=data_re_2(filename)
    filename = dir_name+"0418-1-ver1-2-1-"+file+"13.mat"
    data_13=data_re_2(filename)
    filename = dir_name+"0418-1-ver1-2-1-"+file+"14.mat"
    data_14=data_re_2(filename)
    filename = dir_name+"0418-1-ver1-2-1-"+file+"15.mat"
    data_15=data_re_2(filename)
    data0418_1=mingci(data_0, data_1, data_2, data_3, data_4, data_5, data_6, data_7, data_8, data_9, data_10, data_11, data_12, data_13, data_14, data_15)
    
    # 0419-1
    filename = dir_name+"0419-1-ver1-2-1-"+file+"0.mat"
    data_0=data_re_1(filename)
    filename = dir_name+"0419-1-ver1-2-1-"+file+"1.mat"
    data_1=data_re_1(filename)
    filename = dir_name+"0419-1-ver1-2-1-"+file+"2.mat"
    data_2=data_re_1(filename)
    filename = dir_name+"0419-1-ver1-2-1-"+file+"3.mat"
    data_3=data_re_1(filename)
    filename = dir_name+"0419-1-ver1-2-1-"+file+"4.mat"
    data_4=data_re_1(filename)
    filename = dir_name+"0419-1-ver1-2-1-"+file+"5.mat"
    data_5=data_re_1(filename)
    filename = dir_name+"0419-1-ver1-2-1-"+file+"6.mat"
    data_6=data_re_1(filename)
    filename = dir_name+"0419-1-ver1-2-1-"+file+"7.mat"
    data_7=data_re_1(filename)
    filename = dir_name+"0419-1-ver1-2-1-"+file+"8.mat"
    data_8=data_re_1(filename)
    filename = dir_name+"0419-1-ver1-2-1-"+file+"9.mat"
    data_9=data_re_1(filename)
    filename = dir_name+"0419-1-ver1-2-1-"+file+"10.mat"
    data_10=data_re_1(filename)
    filename = dir_name+"0419-1-ver1-2-1-"+file+"11.mat"
    data_11=data_re_1(filename)
    filename = dir_name+"0419-1-ver1-2-1-"+file+"12.mat"
    data_12=data_re_1(filename)
    filename = dir_name+"0419-1-ver1-2-1-"+file+"13.mat"
    data_13=data_re_1(filename)
    filename = dir_name+"0419-1-ver1-2-1-"+file+"14.mat"
    data_14=data_re_1(filename)
    filename = dir_name+"0419-1-ver1-2-1-"+file+"15.mat"
    data_15=data_re_1(filename)
    data0419_1=mingci(data_0, data_1, data_2, data_3, data_4, data_5, data_6, data_7, data_8, data_9, data_10, data_11, data_12, data_13, data_14, data_15)
    
    # 0419-2
    filename = dir_name+"0419-2-ver1-2-1-"+file+"0.mat"
    data_0=data_re_1(filename)
    filename = dir_name+"0419-2-ver1-2-1-"+file+"1.mat"
    data_1=data_re_1(filename)
    filename = dir_name+"0419-2-ver1-2-1-"+file+"2.mat"
    data_2=data_re_1(filename)
    filename = dir_name+"0419-2-ver1-2-1-"+file+"3.mat"
    data_3=data_re_1(filename)
    filename = dir_name+"0419-2-ver1-2-1-"+file+"4.mat"
    data_4=data_re_1(filename)
    filename = dir_name+"0419-2-ver1-2-1-"+file+"5.mat"
    data_5=data_re_1(filename)
    filename = dir_name+"0419-2-ver1-2-1-"+file+"6.mat"
    data_6=data_re_1(filename)
    filename = dir_name+"0419-2-ver1-2-1-"+file+"7.mat"
    data_7=data_re_1(filename)
    filename = dir_name+"0419-2-ver1-2-1-"+file+"8.mat"
    data_8=data_re_1(filename)
    filename = dir_name+"0419-2-ver1-2-1-"+file+"9.mat"
    data_9=data_re_1(filename)
    filename = dir_name+"0419-2-ver1-2-1-"+file+"10.mat"
    data_10=data_re_1(filename)
    filename = dir_name+"0419-2-ver1-2-1-"+file+"11.mat"
    data_11=data_re_1(filename)
    filename = dir_name+"0419-2-ver1-2-1-"+file+"12.mat"
    data_12=data_re_1(filename)
    filename = dir_name+"0419-2-ver1-2-1-"+file+"13.mat"
    data_13=data_re_1(filename)
    filename = dir_name+"0419-2-ver1-2-1-"+file+"14.mat"
    data_14=data_re_1(filename)
    filename = dir_name+"0419-2-ver1-2-1-"+file+"15.mat"
    data_15=data_re_1(filename)
    data0419_2=mingci(data_0, data_1, data_2, data_3, data_4, data_5, data_6, data_7, data_8, data_9, data_10, data_11, data_12, data_13, data_14, data_15)
    
    # 0420-1
    filename = dir_name+"0420-1-ver1-2-1-"+file+"0.mat"
    data_0=data_re_1(filename)
    filename = dir_name+"0420-1-ver1-2-1-"+file+"1.mat"
    data_1=data_re_1(filename)
    filename = dir_name+"0420-1-ver1-2-1-"+file+"2.mat"
    data_2=data_re_1(filename)
    filename = dir_name+"0420-1-ver1-2-1-"+file+"3.mat"
    data_3=data_re_1(filename)
    filename = dir_name+"0420-1-ver1-2-1-"+file+"4.mat"
    data_4=data_re_1(filename)
    filename = dir_name+"0420-1-ver1-2-1-"+file+"5.mat"
    data_5=data_re_1(filename)
    filename = dir_name+"0420-1-ver1-2-1-"+file+"6.mat"
    data_6=data_re_1(filename)
    filename = dir_name+"0420-1-ver1-2-1-"+file+"7.mat"
    data_7=data_re_1(filename)
    filename = dir_name+"0420-1-ver1-2-1-"+file+"8.mat"
    data_8=data_re_1(filename)
    filename = dir_name+"0420-1-ver1-2-1-"+file+"9.mat"
    data_9=data_re_1(filename)
    filename = dir_name+"0420-1-ver1-2-1-"+file+"10.mat"
    data_10=data_re_1(filename)
    filename = dir_name+"0420-1-ver1-2-1-"+file+"11.mat"
    data_11=data_re_1(filename)
    filename = dir_name+"0420-1-ver1-2-1-"+file+"12.mat"
    data_12=data_re_1(filename)
    filename = dir_name+"0420-1-ver1-2-1-"+file+"13.mat"
    data_13=data_re_1(filename)
    filename = dir_name+"0420-1-ver1-2-1-"+file+"14.mat"
    data_14=data_re_1(filename)
    filename = dir_name+"0420-1-ver1-2-1-"+file+"15.mat"
    data_15=data_re_1(filename)
    data0420_1=mingci(data_0, data_1, data_2, data_3, data_4, data_5, data_6, data_7, data_8, data_9, data_10, data_11, data_12, data_13, data_14, data_15)
    
    # 0420-2
    filename = dir_name+"0420-2-ver1-2-1-"+file+"0.mat"
    data_0=data_re_1(filename)
    filename = dir_name+"0420-2-ver1-2-1-"+file+"1.mat"
    data_1=data_re_1(filename)
    filename = dir_name+"0420-2-ver1-2-1-"+file+"2.mat"
    data_2=data_re_1(filename)
    filename = dir_name+"0420-2-ver1-2-1-"+file+"3.mat"
    data_3=data_re_1(filename)
    filename = dir_name+"0420-2-ver1-2-1-"+file+"4.mat"
    data_4=data_re_1(filename)
    filename = dir_name+"0420-2-ver1-2-1-"+file+"5.mat"
    data_5=data_re_1(filename)
    filename = dir_name+"0420-2-ver1-2-1-"+file+"6.mat"
    data_6=data_re_1(filename)
    filename = dir_name+"0420-2-ver1-2-1-"+file+"7.mat"
    data_7=data_re_1(filename)
    filename = dir_name+"0420-2-ver1-2-1-"+file+"8.mat"
    data_8=data_re_1(filename)
    filename = dir_name+"0420-2-ver1-2-1-"+file+"9.mat"
    data_9=data_re_1(filename)
    filename = dir_name+"0420-2-ver1-2-1-"+file+"10.mat"
    data_10=data_re_1(filename)
    filename = dir_name+"0420-2-ver1-2-1-"+file+"11.mat"
    data_11=data_re_1(filename)
    filename = dir_name+"0420-2-ver1-2-1-"+file+"12.mat"
    data_12=data_re_1(filename)
    filename = dir_name+"0420-2-ver1-2-1-"+file+"13.mat"
    data_13=data_re_1(filename)
    filename = dir_name+"0420-2-ver1-2-1-"+file+"14.mat"
    data_14=data_re_1(filename)
    filename = dir_name+"0420-2-ver1-2-1-"+file+"15.mat"
    data_15=data_re_1(filename)

    data0420_2=mingci(data_0, data_1, data_2, data_3, data_4, data_5, data_6, data_7, data_8, data_9, data_10, data_11, data_12, data_13, data_14, data_15)
    
    # 0421-1
    filename = dir_name+"0421-1-ver1-2-1-"+file+"0.mat"
    data_0=data_re_1(filename)
    filename = dir_name+"0421-1-ver1-2-1-"+file+"1.mat"
    data_1=data_re_1(filename)
    filename = dir_name+"0421-1-ver1-2-1-"+file+"2.mat"
    data_2=data_re_1(filename)
    filename = dir_name+"0421-1-ver1-2-1-"+file+"3.mat"
    data_3=data_re_1(filename)
    filename = dir_name+"0421-1-ver1-2-1-"+file+"4.mat"
    data_4=data_re_1(filename)
    filename = dir_name+"0421-1-ver1-2-1-"+file+"5.mat"
    data_5=data_re_1(filename)
    filename = dir_name+"0421-1-ver1-2-1-"+file+"6.mat"
    data_6=data_re_1(filename)
    filename = dir_name+"0421-1-ver1-2-1-"+file+"7.mat"
    data_7=data_re_1(filename)
    filename = dir_name+"0421-1-ver1-2-1-"+file+"8.mat"
    data_8=data_re_1(filename)
    filename = dir_name+"0421-1-ver1-2-1-"+file+"9.mat"
    data_9=data_re_1(filename)
    filename = dir_name+"0421-1-ver1-2-1-"+file+"10.mat"
    data_10=data_re_1(filename)
    filename = dir_name+"0421-1-ver1-2-1-"+file+"11.mat"
    data_11=data_re_1(filename)
    filename = dir_name+"0421-1-ver1-2-1-"+file+"12.mat"
    data_12=data_re_1(filename)
    filename = dir_name+"0421-1-ver1-2-1-"+file+"13.mat"
    data_13=data_re_1(filename)
    filename = dir_name+"0421-1-ver1-2-1-"+file+"14.mat"
    data_14=data_re_1(filename)
    filename = dir_name+"0421-1-ver1-2-1-"+file+"15.mat"
    data_15=data_re_1(filename)
    
    data0421_1=mingci(data_0, data_1, data_2, data_3, data_4, data_5, data_6, data_7, data_8, data_9, data_10, data_11, data_12, data_13, data_14, data_15)
    
    # 0421-2
    filename = dir_name+"0421-2-ver1-2-1-"+file+"0.mat"
    data_0=data_re_1(filename)
    filename = dir_name+"0421-2-ver1-2-1-"+file+"1.mat"
    data_1=data_re_1(filename)
    filename = dir_name+"0421-2-ver1-2-1-"+file+"2.mat"
    data_2=data_re_1(filename)
    filename = dir_name+"0421-2-ver1-2-1-"+file+"3.mat"
    data_3=data_re_1(filename)
    filename = dir_name+"0421-2-ver1-2-1-"+file+"4.mat"
    data_4=data_re_1(filename)
    filename = dir_name+"0421-2-ver1-2-1-"+file+"5.mat"
    data_5=data_re_1(filename)
    filename = dir_name+"0421-2-ver1-2-1-"+file+"6.mat"
    data_6=data_re_1(filename)
    filename = dir_name+"0421-2-ver1-2-1-"+file+"7.mat"
    data_7=data_re_1(filename)
    filename = dir_name+"0421-2-ver1-2-1-"+file+"8.mat"
    data_8=data_re_1(filename)
    filename = dir_name+"0421-2-ver1-2-1-"+file+"9.mat"
    data_9=data_re_1(filename)
    filename = dir_name+"0421-2-ver1-2-1-"+file+"10.mat"
    data_10=data_re_1(filename)
    filename = dir_name+"0421-2-ver1-2-1-"+file+"11.mat"
    data_11=data_re_1(filename)
    filename = dir_name+"0421-2-ver1-2-1-"+file+"12.mat"
    data_12=data_re_1(filename)
    filename = dir_name+"0421-2-ver1-2-1-"+file+"13.mat"
    data_13=data_re_1(filename)
    filename = dir_name+"0421-2-ver1-2-1-"+file+"14.mat"
    data_14=data_re_1(filename)
    filename = dir_name+"0421-2-ver1-2-1-"+file+"15.mat"
    data_15=data_re_1(filename)

    data0421_2=mingci(data_0, data_1, data_2, data_3, data_4, data_5, data_6, data_7, data_8, data_9, data_10, data_11, data_12, data_13, data_14, data_15)
    
    # 0422-1
    filename = dir_name+"0422-1-ver1-2-1-"+file+"0.mat"
    data_0=data_re_1(filename)
    filename = dir_name+"0422-1-ver1-2-1-"+file+"1.mat"
    data_1=data_re_1(filename)
    filename = dir_name+"0422-1-ver1-2-1-"+file+"2.mat"
    data_2=data_re_1(filename)
    filename = dir_name+"0422-1-ver1-2-1-"+file+"3.mat"
    data_3=data_re_1(filename)
    filename = dir_name+"0422-1-ver1-2-1-"+file+"4.mat"
    data_4=data_re_1(filename)
    filename = dir_name+"0422-1-ver1-2-1-"+file+"5.mat"
    data_5=data_re_1(filename)
    filename = dir_name+"0422-1-ver1-2-1-"+file+"6.mat"
    data_6=data_re_1(filename)
    filename = dir_name+"0422-1-ver1-2-1-"+file+"7.mat"
    data_7=data_re_1(filename)
    filename = dir_name+"0422-1-ver1-2-1-"+file+"8.mat"
    data_8=data_re_1(filename)
    filename = dir_name+"0422-1-ver1-2-1-"+file+"9.mat"
    data_9=data_re_1(filename)
    filename = dir_name+"0422-1-ver1-2-1-"+file+"10.mat"
    data_10=data_re_1(filename)
    filename = dir_name+"0422-1-ver1-2-1-"+file+"11.mat"
    data_11=data_re_1(filename)
    filename = dir_name+"0422-1-ver1-2-1-"+file+"12.mat"
    data_12=data_re_1(filename)
    filename = dir_name+"0422-1-ver1-2-1-"+file+"13.mat"
    data_13=data_re_1(filename)
    filename = dir_name+"0422-1-ver1-2-1-"+file+"14.mat"
    data_14=data_re_1(filename)
    filename = dir_name+"0422-1-ver1-2-1-"+file+"15.mat"
    data_15=data_re_1(filename)

    data0422_1=mingci(data_0, data_1, data_2, data_3, data_4, data_5, data_6, data_7, data_8, data_9, data_10, data_11, data_12, data_13, data_14, data_15)
    
    # 0422-2
    filename = dir_name+"0422-2-ver1-2-1-"+file+"0.mat"
    data_0=data_re_1(filename)
    filename = dir_name+"0422-2-ver1-2-1-"+file+"1.mat"
    data_1=data_re_1(filename)
    filename = dir_name+"0422-2-ver1-2-1-"+file+"2.mat"
    data_2=data_re_1(filename)
    filename = dir_name+"0422-2-ver1-2-1-"+file+"3.mat"
    data_3=data_re_1(filename)
    filename = dir_name+"0422-2-ver1-2-1-"+file+"4.mat"
    data_4=data_re_1(filename)
    filename = dir_name+"0422-2-ver1-2-1-"+file+"5.mat"
    data_5=data_re_1(filename)
    filename = dir_name+"0422-2-ver1-2-1-"+file+"6.mat"
    data_6=data_re_1(filename)
    filename = dir_name+"0422-2-ver1-2-1-"+file+"7.mat"
    data_7=data_re_1(filename)
    filename = dir_name+"0422-2-ver1-2-1-"+file+"8.mat"
    data_8=data_re_1(filename)
    filename = dir_name+"0422-2-ver1-2-1-"+file+"9.mat"
    data_9=data_re_1(filename)
    filename = dir_name+"0422-2-ver1-2-1-"+file+"10.mat"
    data_10=data_re_1(filename)
    filename = dir_name+"0422-2-ver1-2-1-"+file+"11.mat"
    data_11=data_re_1(filename)
    filename = dir_name+"0422-2-ver1-2-1-"+file+"12.mat"
    data_12=data_re_1(filename)
    filename = dir_name+"0422-2-ver1-2-1-"+file+"13.mat"
    data_13=data_re_1(filename)
    filename = dir_name+"0422-2-ver1-2-1-"+file+"14.mat"
    data_14=data_re_1(filename)
    filename = dir_name+"0422-2-ver1-2-1-"+file+"15.mat"
    data_15=data_re_1(filename)

    data0422_2=mingci(data_0, data_1, data_2, data_3, data_4, data_5, data_6, data_7, data_8, data_9, data_10, data_11, data_12, data_13, data_14, data_15)
    
    #0423-1
    filename = dir_name+"0423-1-ver1-2-1-"+file+"0.mat"
    data_0=data_re_1(filename)
    filename = dir_name+"0423-1-ver1-2-1-"+file+"1.mat"
    data_1=data_re_1(filename)
    filename = dir_name+"0423-1-ver1-2-1-"+file+"2.mat"
    data_2=data_re_1(filename)
    filename = dir_name+"0423-1-ver1-2-1-"+file+"3.mat"
    data_3=data_re_1(filename)
    filename = dir_name+"0423-1-ver1-2-1-"+file+"4.mat"
    data_4=data_re_1(filename)
    filename = dir_name+"0423-1-ver1-2-1-"+file+"5.mat"
    data_5=data_re_1(filename)
    filename = dir_name+"0423-1-ver1-2-1-"+file+"6.mat"
    data_6=data_re_1(filename)
    filename = dir_name+"0423-1-ver1-2-1-"+file+"7.mat"
    data_7=data_re_1(filename)
    filename = dir_name+"0423-1-ver1-2-1-"+file+"8.mat"
    data_8=data_re_1(filename)
    filename = dir_name+"0423-1-ver1-2-1-"+file+"9.mat"
    data_9=data_re_1(filename)
    filename = dir_name+"0423-1-ver1-2-1-"+file+"10.mat"
    data_10=data_re_1(filename)
    filename = dir_name+"0423-1-ver1-2-1-"+file+"11.mat"
    data_11=data_re_1(filename)
    filename = dir_name+"0423-1-ver1-2-1-"+file+"12.mat"
    data_12=data_re_1(filename)
    filename = dir_name+"0423-1-ver1-2-1-"+file+"13.mat"
    data_13=data_re_1(filename)
    filename = dir_name+"0423-1-ver1-2-1-"+file+"14.mat"
    data_14=data_re_1(filename)
    filename = dir_name+"0423-1-ver1-2-1-"+file+"15.mat"
    data_15=data_re_1(filename)

    data0423_1=mingci(data_0, data_1, data_2, data_3, data_4, data_5, data_6, data_7, data_8, data_9, data_10, data_11, data_12, data_13, data_14, data_15)
    
    #0423-2
    filename = dir_name+"0423-2-ver1-2-1-"+file+"0.mat"
    data_0=data_re_1(filename)
    filename = dir_name+"0423-2-ver1-2-1-"+file+"1.mat"
    data_1=data_re_1(filename)
    filename = dir_name+"0423-2-ver1-2-1-"+file+"2.mat"
    data_2=data_re_1(filename)
    filename = dir_name+"0423-2-ver1-2-1-"+file+"3.mat"
    data_3=data_re_1(filename)
    filename = dir_name+"0423-2-ver1-2-1-"+file+"4.mat"
    data_4=data_re_1(filename)
    filename = dir_name+"0423-2-ver1-2-1-"+file+"5.mat"
    data_5=data_re_1(filename)
    filename = dir_name+"0423-2-ver1-2-1-"+file+"6.mat"
    data_6=data_re_1(filename)
    filename = dir_name+"0423-2-ver1-2-1-"+file+"7.mat"
    data_7=data_re_1(filename)
    filename = dir_name+"0423-2-ver1-2-1-"+file+"8.mat"
    data_8=data_re_1(filename)
    filename = dir_name+"0423-2-ver1-2-1-"+file+"9.mat"
    data_9=data_re_1(filename)
    filename = dir_name+"0423-2-ver1-2-1-"+file+"10.mat"
    data_10=data_re_1(filename)
    filename = dir_name+"0423-2-ver1-2-1-"+file+"11.mat"
    data_11=data_re_1(filename)
    filename = dir_name+"0423-2-ver1-2-1-"+file+"12.mat"
    data_12=data_re_1(filename)
    filename = dir_name+"0423-2-ver1-2-1-"+file+"13.mat"
    data_13=data_re_1(filename)
    filename = dir_name+"0423-2-ver1-2-1-"+file+"14.mat"
    data_14=data_re_1(filename)
    filename = dir_name+"0423-2-ver1-2-1-"+file+"15.mat"
    data_15=data_re_1(filename)
    data0423_2=mingci(data_0, data_1, data_2, data_3, data_4, data_5, data_6, data_7, data_8, data_9, data_10, data_11, data_12, data_13, data_14, data_15)
    
    data00=np.vstack((data0416_1,data0417_1))
    data01=np.vstack((data00,data0417_2))
    data02=np.vstack((data01,data0418_1))
    data03=np.vstack((data02,data0419_1))
    data04=np.vstack((data03,data0419_2))
    data05=np.vstack((data04,data0420_1))
    data06=np.vstack((data05,data0420_2))
    data07=np.vstack((data06,data0421_1))
    data08=np.vstack((data07,data0421_2))
    data09=np.vstack((data08,data0422_1))
    data10=np.vstack((data09,data0422_2))
    data11=np.vstack((data10,data0423_1))
    epochs = np.vstack((data11,data0423_2))
    epochs = epochs.reshape([-1, 500, 63])
    print("epochs",epochs.shape)
    
    return epochs

#名词数据
#dir_name="../../../opt/yidongyingpan/cmx_bishe/epoch/名詞/足球场/"
#file="zqc-n-"
#epochs_1_b=data_mingci(dir_name, file)

#dir_name="../../../opt/yidongyingpan/cmx_bishe/epoch/名詞/东京塔/"
#file="djt-n-"
#epochs_2_b=data_mingci(dir_name, file)

#dir_name="../../../opt/yidongyingpan/cmx_bishe/epoch/名詞/实验室/"
#file="shysh-n-"
#epochs_3_b=data_mingci(dir_name, file)

#dir_name="../../../opt/yidongyingpan/cmx_bishe/epoch/名詞/爱因斯坦/"
#file="ayst-n-"
#epochs_1_b=data_mingci(dir_name, file)

#dir_name="../../../opt/yidongyingpan/cmx_bishe/epoch/名詞/梅西/"
#file="mx-n-"
#epochs_2_b=data_mingci(dir_name, file)

#dir_name="../../../opt/yidongyingpan/cmx_bishe/epoch/名詞/成龙/"
#file="jc-n-"
#epochs_3_b=data_mingci(dir_name, file)

#dir_name="../../../opt/yidongyingpan/cmx_bishe/epoch/名詞/香蕉/"
#file="bnn-n-"
#epochs_1_b=data_mingci(dir_name, file)

#dir_name="../../../opt/yidongyingpan/cmx_bishe/epoch/名詞/橘子/"
#file="org-n-"
#epochs_2_b=data_mingci(dir_name, file)

#dir_name="../../../opt/yidongyingpan/cmx_bishe/epoch/名詞/西瓜/"
#file="wtm-n-"
#epochs_3_b=data_mingci(dir_name, file)

dir_name="../../../opt/yidongyingpan/cmx_bishe/epoch/名詞/羽毛球/"
file="btm-n-"
epochs_1_b=data_mingci(dir_name, file)

dir_name="../../../opt/yidongyingpan/cmx_bishe/epoch/名詞/游泳/"
file="swm-n-"
epochs_2_b=data_mingci(dir_name, file)

dir_name="../../../opt/yidongyingpan/cmx_bishe/epoch/名詞/篮球/"
file="bsk-n-"
epochs_3_b=data_mingci(dir_name, file)

# Hyper Parameters
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# VISUALIZATION = True
VISUALIZATION = False
CLASSES_NUM = 3
EPOCH = 200
BATCH_SIZE = 32
LR = 3e-4

# get model
eeg_net = EEGNet(classes_num=CLASSES_NUM).to(DEVICE)

optimizer = torch.optim.Adam(eeg_net.parameters(), lr=LR)
loss_func = nn.CrossEntropyLoss().to(DEVICE)

epochs_1_train_b = epochs_1_b[:600,:,:]
epochs_1_test_b = epochs_1_b[600:700,:,:]
print("epochs_1_train_b",epochs_1_train_b.shape)
print("epochs_1_test_b",epochs_1_test_b.shape)
epochs_2_train_b = epochs_2_b[:600,:,:]
epochs_2_test_b = epochs_2_b[600:700,:,:]
print("epochs_2_train_b",epochs_2_train_b.shape)
print("epochs_2_test_b",epochs_2_test_b.shape)
epochs_3_train_b = epochs_3_b[:600,:,:]
epochs_3_test_b = epochs_3_b[600:700,:,:]
print("epochs_3_train_b",epochs_3_train_b.shape)
print("epochs_3_test_b",epochs_3_test_b.shape)


epochs_train= np.vstack((epochs_1_train_b, epochs_2_train_b, epochs_3_train_b))
epochs_test = np.vstack((epochs_1_test_b, epochs_2_test_b, epochs_3_test_b))
print("epochs_train",epochs_train.shape)
print("epochs_test",epochs_test.shape)

train_labels = np.zeros(1800, dtype=np.int64)
train_labels[600:1200] = 1
train_labels[1200:1800] = 2

test_labels = np.zeros(300, dtype=np.int64)
test_labels[100:200] = 1
test_labels[200:300] = 2

train_x = torch.unsqueeze(torch.from_numpy(epochs_train).type(torch.float),dim=1)
test_x = torch.unsqueeze(torch.from_numpy(epochs_test).type(torch.float),dim=1)

train_y = torch.from_numpy((train_labels))
test_y = torch.from_numpy((test_labels))

print("train_x",train_x.shape)
print("test_x",test_x.shape)
print("train_y",train_y.shape)
print("test_y",test_y.shape)

dataset = Data.TensorDataset(train_x, train_y)
dataloader = Data.DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

plt.ion()

# start train
print('use {}'.format("cuda" if torch.cuda.is_available() else "cpu"))

for epoch in range(EPOCH):
    for step, (b_x, b_y) in enumerate(dataloader):
        b_x, b_y = b_x.to(DEVICE), b_y.to(DEVICE)
        output = eeg_net(b_x)[1]
        loss = loss_func(output, b_y.long())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 5 == 0:
          with torch.no_grad():
            test_out, last_layer = eeg_net(test_x.to(DEVICE))
            pred_y = torch.max(test_out.cpu(), 1)[1].data.numpy()
            test_accuracy = float((pred_y == test_y.data.numpy()).astype(int).sum()) / float(test_y.size(0))

    print('Epoch: ', epoch, '| train loss: %.4f' % loss.cpu().data.numpy(),'| test accuracy: %.3f' % test_accuracy)

plt.ioff()