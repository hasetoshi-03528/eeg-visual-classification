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
        # print("block1", x.shape)
        x = self.block_2(x)
        # print("block2", x.shape)
        x = self.block_3(x)
        # print("block3", x.shape)

        x = x.view(x.size(0), -1)
        x = self.out(x)
        return F.softmax(x, dim=1), x  # return x for visualization

def resort(arr):
    return np.array([arr[..., item] for item in range(arr.shape[2])])

dir_name="../../../opt/yidongyingpan/cmx_bishe/epoch/建筑/足球场/"
file="zqc"

filename = dir_name+"0416-1-ver1-2-1-"+file+".mat"			# data shape: 551*50*63
data0 = resort(scipy.io.loadmat(filename)['epochs_data'])
data0_1 = data0.reshape([-1, 500, 63])
filename = dir_name+"0417-1-ver1-2-1-"+file+".mat"
data1 = resort(scipy.io.loadmat(filename)['epochs_data'])
data1_1 = data1.reshape([-1, 500, 63])
filename = dir_name+"0417-2-ver1-2-1-"+file+".mat"
data2 = resort(scipy.io.loadmat(filename)['epochs_data'])
data2_1 = data2.reshape([-1, 500, 63])
filename = dir_name+"0418-1-ver1-2-1-"+file+".mat"			# data shape: 551*50*64		eog
data3 = resort(scipy.io.loadmat(filename)['epochs_data'])
data3_1 = np.delete(data3,63,2)
data3_2 = data3_1.reshape([-1, 500, 63])
filename = dir_name+"0419-1-ver1-2-1-"+file+".mat"
data4 = resort(scipy.io.loadmat(filename)['epochs_data'])
data4_1 = data4.reshape([-1, 500, 63])
filename = dir_name+"0419-2-ver1-2-1-"+file+".mat"
data5 = resort(scipy.io.loadmat(filename)['epochs_data'])
data5_1 = data5.reshape([-1, 500, 63])
filename = dir_name+"0420-1-ver1-2-1-"+file+".mat"			# data shape: 551*37*63		数据样本: 576
data6 = resort(scipy.io.loadmat(filename)['epochs_data'])
data6_1 = data6.reshape([-1, 500, 63])
filename = dir_name+"0420-2-ver1-2-1-"+file+".mat"
data7 = resort(scipy.io.loadmat(filename)['epochs_data'])
data7_1 = data7.reshape([-1, 500, 63])
filename = dir_name+"0421-1-ver1-2-1-"+file+".mat"
data8 = resort(scipy.io.loadmat(filename)['epochs_data'])
data8_1 = data8.reshape([-1, 500, 63])
filename = dir_name+"0421-2-ver1-2-1-"+file+".mat"
data9 = resort(scipy.io.loadmat(filename)['epochs_data'])
data9_1 = data9.reshape([-1, 500, 63])
filename = dir_name+"0422-1-ver1-2-1-"+file+".mat"
data10= resort(scipy.io.loadmat(filename)['epochs_data'])
data10_1= data10.reshape([-1, 500, 63])
filename = dir_name+"0422-2-ver1-2-1-"+file+".mat"
data11= resort(scipy.io.loadmat(filename)['epochs_data'])
data11_1= data11.reshape([-1, 500, 63])
filename = dir_name+"0423-1-ver1-2-1-"+file+".mat"
data12= resort(scipy.io.loadmat(filename)['epochs_data'])
data12_1= data12.reshape([-1, 500, 63])
filename = dir_name+"0423-2-ver1-2-1-"+file+".mat"
data13= resort(scipy.io.loadmat(filename)['epochs_data'])
data13_1= data13.reshape([-1, 500, 63])

data14 = np.vstack((data0_1 , data1_1))
data15 = np.vstack((data14, data2_1))
data16 = np.vstack((data15, data3_2))
data17 = np.vstack((data16, data4_1))
data18 = np.vstack((data17, data5_1))
data19 = np.vstack((data18, data6_1))
data20 = np.vstack((data19, data7_1))
data21 = np.vstack((data20, data8_1))
data22 = np.vstack((data21, data9_1))
data23 = np.vstack((data22, data10_1))
data24 = np.vstack((data23, data11_1))
data25 = np.vstack((data24, data12_1))
epochs_1 = np.vstack((data25, data13_1))

print(epochs_1.shape)
#print(epochs)
epochs_1 = epochs_1.reshape([-1, 500, 63])
print("epochs_1",epochs_1.shape)

dir_name="../../../opt/yidongyingpan/cmx_bishe/epoch/建筑/东京塔/"
file="djt"

filename = dir_name+"0416-1-ver1-2-1-"+file+".mat"			# data shape: 551*50*63
data0 = resort(scipy.io.loadmat(filename)['epochs_data'])
data0_1 = data0.reshape([-1, 500, 63])
filename = dir_name+"0417-1-ver1-2-1-"+file+".mat"
data1 = resort(scipy.io.loadmat(filename)['epochs_data'])
data1_1 = data1.reshape([-1, 500, 63])
filename = dir_name+"0417-2-ver1-2-1-"+file+".mat"
data2 = resort(scipy.io.loadmat(filename)['epochs_data'])
data2_1 = data2.reshape([-1, 500, 63])
filename = dir_name+"0418-1-ver1-2-1-"+file+".mat"			# data shape: 551*50*64		eog
data3 = resort(scipy.io.loadmat(filename)['epochs_data'])
data3_1 = np.delete(data3,63,2)
data3_2 = data3_1.reshape([-1, 500, 63])
filename = dir_name+"0419-1-ver1-2-1-"+file+".mat"
data4 = resort(scipy.io.loadmat(filename)['epochs_data'])
data4_1 = data4.reshape([-1, 500, 63])
filename = dir_name+"0419-2-ver1-2-1-"+file+".mat"
data5 = resort(scipy.io.loadmat(filename)['epochs_data'])
data5_1 = data5.reshape([-1, 500, 63])
filename = dir_name+"0420-1-ver1-2-1-"+file+".mat"			# data shape: 551*37*63		数据样本: 576
data6 = resort(scipy.io.loadmat(filename)['epochs_data'])
data6_1 = data6.reshape([-1, 500, 63])
filename = dir_name+"0420-2-ver1-2-1-"+file+".mat"
data7 = resort(scipy.io.loadmat(filename)['epochs_data'])
data7_1 = data7.reshape([-1, 500, 63])
filename = dir_name+"0421-1-ver1-2-1-"+file+".mat"
data8 = resort(scipy.io.loadmat(filename)['epochs_data'])
data8_1 = data8.reshape([-1, 500, 63])
filename = dir_name+"0421-2-ver1-2-1-"+file+".mat"
data9 = resort(scipy.io.loadmat(filename)['epochs_data'])
data9_1 = data9.reshape([-1, 500, 63])
filename = dir_name+"0422-1-ver1-2-1-"+file+".mat"
data10= resort(scipy.io.loadmat(filename)['epochs_data'])
data10_1= data10.reshape([-1, 500, 63])
filename = dir_name+"0422-2-ver1-2-1-"+file+".mat"
data11= resort(scipy.io.loadmat(filename)['epochs_data'])
data11_1= data11.reshape([-1, 500, 63])
filename = dir_name+"0423-1-ver1-2-1-"+file+".mat"
data12= resort(scipy.io.loadmat(filename)['epochs_data'])
data12_1= data12.reshape([-1, 500, 63])
filename = dir_name+"0423-2-ver1-2-1-"+file+".mat"
data13= resort(scipy.io.loadmat(filename)['epochs_data'])
data13_1= data13.reshape([-1, 500, 63])

data14 = np.vstack((data0_1 , data1_1))
data15 = np.vstack((data14, data2_1))
data16 = np.vstack((data15, data3_2))
data17 = np.vstack((data16, data4_1))
data18 = np.vstack((data17, data5_1))
data19 = np.vstack((data18, data6_1))
data20 = np.vstack((data19, data7_1))
data21 = np.vstack((data20, data8_1))
data22 = np.vstack((data21, data9_1))
data23 = np.vstack((data22, data10_1))
data24 = np.vstack((data23, data11_1))
data25 = np.vstack((data24, data12_1))
epochs_2 = np.vstack((data25, data13_1))
epochs_2 = np.vstack((epochs_1,epochs_2))
epochs_2 = epochs_2.reshape([-1, 500, 63])

print("epochs_2",epochs_2.shape)

dir_name="../../../opt/yidongyingpan/cmx_bishe/epoch/建筑/实验室/"
file="shysh"

filename = dir_name+"0420-1-ver1-2-1-"+file+".mat"
data6 = resort(scipy.io.loadmat(filename)['epochs_data'])
print("data6 shape",data6.shape)
filename = dir_name+"0418-1-ver1-2-1-"+file+".mat"			# data shape: 551*50*64
data3 = resort(scipy.io.loadmat(filename)['epochs_data'])
print("data3 shape",data3.shape)

filename = dir_name+"0416-1-ver1-2-1-"+file+".mat"			# data shape: 551*50*63
data0 = resort(scipy.io.loadmat(filename)['epochs_data'])
data0_1 = data0.reshape([-1, 500, 63])
filename = dir_name+"0417-1-ver1-2-1-"+file+".mat"
data1 = resort(scipy.io.loadmat(filename)['epochs_data'])
data1_1 = data1.reshape([-1, 500, 63])
filename = dir_name+"0417-2-ver1-2-1-"+file+".mat"
data2 = resort(scipy.io.loadmat(filename)['epochs_data'])
data2_1 = data2.reshape([-1, 500, 63])
filename = dir_name+"0418-1-ver1-2-1-"+file+".mat"			# data shape: 551*50*64		eog
data3 = resort(scipy.io.loadmat(filename)['epochs_data'])
data3_1 = np.delete(data3,63,2)
data3_2 = data3_1.reshape([-1, 500, 63])
filename = dir_name+"0419-1-ver1-2-1-"+file+".mat"
data4 = resort(scipy.io.loadmat(filename)['epochs_data'])
data4_1 = data4.reshape([-1, 500, 63])
filename = dir_name+"0419-2-ver1-2-1-"+file+".mat"
data5 = resort(scipy.io.loadmat(filename)['epochs_data'])
data5_1 = data5.reshape([-1, 500, 63])
filename = dir_name+"0420-1-ver1-2-1-"+file+".mat"			# data shape: 551*37*63		数据样本: 576
data6 = resort(scipy.io.loadmat(filename)['epochs_data'])
data6_1 = data6.reshape([-1, 500, 63])
filename = dir_name+"0420-2-ver1-2-1-"+file+".mat"
data7 = resort(scipy.io.loadmat(filename)['epochs_data'])
data7_1 = data7.reshape([-1, 500, 63])
filename = dir_name+"0421-1-ver1-2-1-"+file+".mat"
data8 = resort(scipy.io.loadmat(filename)['epochs_data'])
data8_1 = data8.reshape([-1, 500, 63])
filename = dir_name+"0421-2-ver1-2-1-"+file+".mat"
data9 = resort(scipy.io.loadmat(filename)['epochs_data'])
data9_1 = data9.reshape([-1, 500, 63])
filename = dir_name+"0422-1-ver1-2-1-"+file+".mat"
data10= resort(scipy.io.loadmat(filename)['epochs_data'])
data10_1= data10.reshape([-1, 500, 63])
filename = dir_name+"0422-2-ver1-2-1-"+file+".mat"
data11= resort(scipy.io.loadmat(filename)['epochs_data'])
data11_1= data11.reshape([-1, 500, 63])
filename = dir_name+"0423-1-ver1-2-1-"+file+".mat"
data12= resort(scipy.io.loadmat(filename)['epochs_data'])
data12_1= data12.reshape([-1, 500, 63])
filename = dir_name+"0423-2-ver1-2-1-"+file+".mat"
data13= resort(scipy.io.loadmat(filename)['epochs_data'])
data13_1= data13.reshape([-1, 500, 63])

data14 = np.vstack((data0_1 , data1_1))
data15 = np.vstack((data14, data2_1))
data16 = np.vstack((data15, data3_2))
data17 = np.vstack((data16, data4_1))
data18 = np.vstack((data17, data5_1))
data19 = np.vstack((data18, data6_1))
data20 = np.vstack((data19, data7_1))
data21 = np.vstack((data20, data8_1))
data22 = np.vstack((data21, data9_1))
data23 = np.vstack((data22, data10_1))
data24 = np.vstack((data23, data11_1))
data25 = np.vstack((data24, data12_1))
epochs_3 = np.vstack((data25, data13_1))
epochs_3 = epochs_3.reshape([-1, 500, 63])
epochs_3 = np.vstack((epochs_2, epochs_3)) 

print("epochs_3",epochs_3.shape)

dir_name="../../../opt/yidongyingpan/cmx_bishe/epoch/人物/爱因斯坦/"
file="ayst"

filename = dir_name+"0420-1-ver1-2-1-"+file+".mat"
data6 = resort(scipy.io.loadmat(filename)['epochs_data'])
print("data6 shape",data6.shape)
filename = dir_name+"0418-1-ver1-2-1-"+file+".mat"			# data shape: 551*50*64
data3 = resort(scipy.io.loadmat(filename)['epochs_data'])
print("data3 shape",data3.shape)

filename = dir_name+"0416-1-ver1-2-1-"+file+".mat"			# data shape: 551*50*63
data0 = resort(scipy.io.loadmat(filename)['epochs_data'])
data0_1 = data0.reshape([-1, 500, 63])
filename = dir_name+"0417-1-ver1-2-1-"+file+".mat"
data1 = resort(scipy.io.loadmat(filename)['epochs_data'])
data1_1 = data1.reshape([-1, 500, 63])
filename = dir_name+"0417-2-ver1-2-1-"+file+".mat"
data2 = resort(scipy.io.loadmat(filename)['epochs_data'])
data2_1 = data2.reshape([-1, 500, 63])
filename = dir_name+"0418-1-ver1-2-1-"+file+".mat"			# data shape: 551*50*64		eog
data3 = resort(scipy.io.loadmat(filename)['epochs_data'])
data3_1 = np.delete(data3,63,2)
data3_2 = data3_1.reshape([-1, 500, 63])
filename = dir_name+"0419-1-ver1-2-1-"+file+".mat"
data4 = resort(scipy.io.loadmat(filename)['epochs_data'])
data4_1 = data4.reshape([-1, 500, 63])
filename = dir_name+"0419-2-ver1-2-1-"+file+".mat"
data5 = resort(scipy.io.loadmat(filename)['epochs_data'])
data5_1 = data5.reshape([-1, 500, 63])
filename = dir_name+"0420-1-ver1-2-1-"+file+".mat"			# data shape: 551*37*63		数据样本: 576
data6 = resort(scipy.io.loadmat(filename)['epochs_data'])
data6_1 = data6.reshape([-1, 500, 63])
filename = dir_name+"0420-2-ver1-2-1-"+file+".mat"
data7 = resort(scipy.io.loadmat(filename)['epochs_data'])
data7_1 = data7.reshape([-1, 500, 63])
filename = dir_name+"0421-1-ver1-2-1-"+file+".mat"
data8 = resort(scipy.io.loadmat(filename)['epochs_data'])
data8_1 = data8.reshape([-1, 500, 63])
filename = dir_name+"0421-2-ver1-2-1-"+file+".mat"
data9 = resort(scipy.io.loadmat(filename)['epochs_data'])
data9_1 = data9.reshape([-1, 500, 63])
filename = dir_name+"0422-1-ver1-2-1-"+file+".mat"
data10= resort(scipy.io.loadmat(filename)['epochs_data'])
data10_1= data10.reshape([-1, 500, 63])
filename = dir_name+"0422-2-ver1-2-1-"+file+".mat"
data11= resort(scipy.io.loadmat(filename)['epochs_data'])
data11_1= data11.reshape([-1, 500, 63])
filename = dir_name+"0423-1-ver1-2-1-"+file+".mat"
data12= resort(scipy.io.loadmat(filename)['epochs_data'])
data12_1= data12.reshape([-1, 500, 63])
filename = dir_name+"0423-2-ver1-2-1-"+file+".mat"
data13= resort(scipy.io.loadmat(filename)['epochs_data'])
data13_1= data13.reshape([-1, 500, 63])

data14 = np.vstack((data0_1 , data1_1))
data15 = np.vstack((data14, data2_1))
data16 = np.vstack((data15, data3_2))
data17 = np.vstack((data16, data4_1))
data18 = np.vstack((data17, data5_1))
data19 = np.vstack((data18, data6_1))
data20 = np.vstack((data19, data7_1))
data21 = np.vstack((data20, data8_1))
data22 = np.vstack((data21, data9_1))
data23 = np.vstack((data22, data10_1))
data24 = np.vstack((data23, data11_1))
data25 = np.vstack((data24, data12_1))
epochs_4 = np.vstack((data25, data13_1))

print(epochs_4.shape)
#print(epochs)
epochs_4 = epochs_4.reshape([-1, 500, 63])
print("epochs_4",epochs_4.shape)

dir_name="../../../opt/yidongyingpan/cmx_bishe/epoch/人物/梅西/"
file="mx"

filename = dir_name+"0420-1-ver1-2-1-"+file+".mat"
data6 = resort(scipy.io.loadmat(filename)['epochs_data'])
print("data6 shape",data6.shape)
filename = dir_name+"0418-1-ver1-2-1-"+file+".mat"			# data shape: 551*50*64
data3 = resort(scipy.io.loadmat(filename)['epochs_data'])
print("data3 shape",data3.shape)

filename = dir_name+"0416-1-ver1-2-1-"+file+".mat"			# data shape: 551*50*63
data0 = resort(scipy.io.loadmat(filename)['epochs_data'])
data0_1 = data0.reshape([-1, 500, 63])
filename = dir_name+"0417-1-ver1-2-1-"+file+".mat"
data1 = resort(scipy.io.loadmat(filename)['epochs_data'])
data1_1 = data1.reshape([-1, 500, 63])
filename = dir_name+"0417-2-ver1-2-1-"+file+".mat"
data2 = resort(scipy.io.loadmat(filename)['epochs_data'])
data2_1 = data2.reshape([-1, 500, 63])
filename = dir_name+"0418-1-ver1-2-1-"+file+".mat"			# data shape: 551*50*64		eog
data3 = resort(scipy.io.loadmat(filename)['epochs_data'])
data3_1 = np.delete(data3,63,2)
data3_2 = data3_1.reshape([-1, 500, 63])
filename = dir_name+"0419-1-ver1-2-1-"+file+".mat"
data4 = resort(scipy.io.loadmat(filename)['epochs_data'])
data4_1 = data4.reshape([-1, 500, 63])
filename = dir_name+"0419-2-ver1-2-1-"+file+".mat"
data5 = resort(scipy.io.loadmat(filename)['epochs_data'])
data5_1 = data5.reshape([-1, 500, 63])
filename = dir_name+"0420-1-ver1-2-1-"+file+".mat"			# data shape: 551*37*63		数据样本: 576
data6 = resort(scipy.io.loadmat(filename)['epochs_data'])
data6_1 = data6.reshape([-1, 500, 63])
filename = dir_name+"0420-2-ver1-2-1-"+file+".mat"
data7 = resort(scipy.io.loadmat(filename)['epochs_data'])
data7_1 = data7.reshape([-1, 500, 63])
filename = dir_name+"0421-1-ver1-2-1-"+file+".mat"
data8 = resort(scipy.io.loadmat(filename)['epochs_data'])
data8_1 = data8.reshape([-1, 500, 63])
filename = dir_name+"0421-2-ver1-2-1-"+file+".mat"
data9 = resort(scipy.io.loadmat(filename)['epochs_data'])
data9_1 = data9.reshape([-1, 500, 63])
filename = dir_name+"0422-1-ver1-2-1-"+file+".mat"
data10= resort(scipy.io.loadmat(filename)['epochs_data'])
data10_1= data10.reshape([-1, 500, 63])
filename = dir_name+"0422-2-ver1-2-1-"+file+".mat"
data11= resort(scipy.io.loadmat(filename)['epochs_data'])
data11_1= data11.reshape([-1, 500, 63])
filename = dir_name+"0423-1-ver1-2-1-"+file+".mat"
data12= resort(scipy.io.loadmat(filename)['epochs_data'])
data12_1= data12.reshape([-1, 500, 63])
filename = dir_name+"0423-2-ver1-2-1-"+file+".mat"
data13= resort(scipy.io.loadmat(filename)['epochs_data'])
data13_1= data13.reshape([-1, 500, 63])

data14 = np.vstack((data0_1 , data1_1))
data15 = np.vstack((data14, data2_1))
data16 = np.vstack((data15, data3_2))
data17 = np.vstack((data16, data4_1))
data18 = np.vstack((data17, data5_1))
data19 = np.vstack((data18, data6_1))
data20 = np.vstack((data19, data7_1))
data21 = np.vstack((data20, data8_1))
data22 = np.vstack((data21, data9_1))
data23 = np.vstack((data22, data10_1))
data24 = np.vstack((data23, data11_1))
data25 = np.vstack((data24, data12_1))
epochs_5 = np.vstack((data25, data13_1))

epochs_5 = np.vstack((epochs_4, epochs_5))
#print(epochs)
epochs_5 = epochs_5.reshape([-1, 500, 63])
print("epochs_5",epochs_5.shape)

dir_name="../../../opt/yidongyingpan/cmx_bishe/epoch/人物/成龙/"
file="jc"

filename = dir_name+"0420-1-ver1-2-1-"+file+".mat"
data6 = resort(scipy.io.loadmat(filename)['epochs_data'])
print("data6 shape",data6.shape)
filename = dir_name+"0418-1-ver1-2-1-"+file+".mat"			# data shape: 551*50*64
data3 = resort(scipy.io.loadmat(filename)['epochs_data'])
print("data3 shape",data3.shape)

filename = dir_name+"0416-1-ver1-2-1-"+file+".mat"			# data shape: 551*50*63
data0 = resort(scipy.io.loadmat(filename)['epochs_data'])
data0_1 = data0.reshape([-1, 500, 63])
filename = dir_name+"0417-1-ver1-2-1-"+file+".mat"
data1 = resort(scipy.io.loadmat(filename)['epochs_data'])
data1_1 = data1.reshape([-1, 500, 63])
filename = dir_name+"0417-2-ver1-2-1-"+file+".mat"
data2 = resort(scipy.io.loadmat(filename)['epochs_data'])
data2_1 = data2.reshape([-1, 500, 63])
filename = dir_name+"0418-1-ver1-2-1-"+file+".mat"			# data shape: 551*50*64		eog
data3 = resort(scipy.io.loadmat(filename)['epochs_data'])
data3_1 = np.delete(data3,63,2)
data3_2 = data3_1.reshape([-1, 500, 63])
filename = dir_name+"0419-1-ver1-2-1-"+file+".mat"
data4 = resort(scipy.io.loadmat(filename)['epochs_data'])
data4_1 = data4.reshape([-1, 500, 63])
filename = dir_name+"0419-2-ver1-2-1-"+file+".mat"
data5 = resort(scipy.io.loadmat(filename)['epochs_data'])
data5_1 = data5.reshape([-1, 500, 63])
filename = dir_name+"0420-1-ver1-2-1-"+file+".mat"			# data shape: 551*37*63		数据样本: 576
data6 = resort(scipy.io.loadmat(filename)['epochs_data'])
data6_1 = data6.reshape([-1, 500, 63])
filename = dir_name+"0420-2-ver1-2-1-"+file+".mat"
data7 = resort(scipy.io.loadmat(filename)['epochs_data'])
data7_1 = data7.reshape([-1, 500, 63])
filename = dir_name+"0421-1-ver1-2-1-"+file+".mat"
data8 = resort(scipy.io.loadmat(filename)['epochs_data'])
data8_1 = data8.reshape([-1, 500, 63])
filename = dir_name+"0421-2-ver1-2-1-"+file+".mat"
data9 = resort(scipy.io.loadmat(filename)['epochs_data'])
data9_1 = data9.reshape([-1, 500, 63])
filename = dir_name+"0422-1-ver1-2-1-"+file+".mat"
data10= resort(scipy.io.loadmat(filename)['epochs_data'])
data10_1= data10.reshape([-1, 500, 63])
filename = dir_name+"0422-2-ver1-2-1-"+file+".mat"
data11= resort(scipy.io.loadmat(filename)['epochs_data'])
data11_1= data11.reshape([-1, 500, 63])
filename = dir_name+"0423-1-ver1-2-1-"+file+".mat"
data12= resort(scipy.io.loadmat(filename)['epochs_data'])
data12_1= data12.reshape([-1, 500, 63])
filename = dir_name+"0423-2-ver1-2-1-"+file+".mat"
data13= resort(scipy.io.loadmat(filename)['epochs_data'])
data13_1= data13.reshape([-1, 500, 63])

data14 = np.vstack((data0_1 , data1_1))
data15 = np.vstack((data14, data2_1))
data16 = np.vstack((data15, data3_2))
data17 = np.vstack((data16, data4_1))
data18 = np.vstack((data17, data5_1))
data19 = np.vstack((data18, data6_1))
data20 = np.vstack((data19, data7_1))
data21 = np.vstack((data20, data8_1))
data22 = np.vstack((data21, data9_1))
data23 = np.vstack((data22, data10_1))
data24 = np.vstack((data23, data11_1))
data25 = np.vstack((data24, data12_1))
epochs_6 = np.vstack((data25, data13_1))

epochs_6 = np.vstack((epochs_5, epochs_6))
epochs_6 = epochs_6.reshape([-1, 500, 63])
print("epochs_6",epochs_6.shape)


dir_name="../../../opt/yidongyingpan/cmx_bishe/epoch/水果/香蕉/"
file="bnn"

filename = dir_name+"0420-1-ver1-2-1-"+file+".mat"
data6 = resort(scipy.io.loadmat(filename)['epochs_data'])
print("data6 shape",data6.shape)
filename = dir_name+"0418-1-ver1-2-1-"+file+".mat"			# data shape: 551*50*64
data3 = resort(scipy.io.loadmat(filename)['epochs_data'])
print("data3 shape",data3.shape)

filename = dir_name+"0416-1-ver1-2-1-"+file+".mat"			# data shape: 551*50*63
data0 = resort(scipy.io.loadmat(filename)['epochs_data'])
data0_1 = data0.reshape([-1, 500, 63])
filename = dir_name+"0417-1-ver1-2-1-"+file+".mat"
data1 = resort(scipy.io.loadmat(filename)['epochs_data'])
data1_1 = data1.reshape([-1, 500, 63])
filename = dir_name+"0417-2-ver1-2-1-"+file+".mat"
data2 = resort(scipy.io.loadmat(filename)['epochs_data'])
data2_1 = data2.reshape([-1, 500, 63])
filename = dir_name+"0418-1-ver1-2-1-"+file+".mat"			# data shape: 551*50*64		eog
data3 = resort(scipy.io.loadmat(filename)['epochs_data'])
data3_1 = np.delete(data3,63,2)
data3_2 = data3_1.reshape([-1, 500, 63])
filename = dir_name+"0419-1-ver1-2-1-"+file+".mat"
data4 = resort(scipy.io.loadmat(filename)['epochs_data'])
data4_1 = data4.reshape([-1, 500, 63])
filename = dir_name+"0419-2-ver1-2-1-"+file+".mat"
data5 = resort(scipy.io.loadmat(filename)['epochs_data'])
data5_1 = data5.reshape([-1, 500, 63])
filename = dir_name+"0420-1-ver1-2-1-"+file+".mat"			# data shape: 551*37*63		数据样本: 576
data6 = resort(scipy.io.loadmat(filename)['epochs_data'])
data6_1 = data6.reshape([-1, 500, 63])
filename = dir_name+"0420-2-ver1-2-1-"+file+".mat"
data7 = resort(scipy.io.loadmat(filename)['epochs_data'])
data7_1 = data7.reshape([-1, 500, 63])
filename = dir_name+"0421-1-ver1-2-1-"+file+".mat"
data8 = resort(scipy.io.loadmat(filename)['epochs_data'])
data8_1 = data8.reshape([-1, 500, 63])
filename = dir_name+"0421-2-ver1-2-1-"+file+".mat"
data9 = resort(scipy.io.loadmat(filename)['epochs_data'])
data9_1 = data9.reshape([-1, 500, 63])
filename = dir_name+"0422-1-ver1-2-1-"+file+".mat"
data10= resort(scipy.io.loadmat(filename)['epochs_data'])
data10_1= data10.reshape([-1, 500, 63])
filename = dir_name+"0422-2-ver1-2-1-"+file+".mat"
data11= resort(scipy.io.loadmat(filename)['epochs_data'])
data11_1= data11.reshape([-1, 500, 63])
filename = dir_name+"0423-1-ver1-2-1-"+file+".mat"
data12= resort(scipy.io.loadmat(filename)['epochs_data'])
data12_1= data12.reshape([-1, 500, 63])
filename = dir_name+"0423-2-ver1-2-1-"+file+".mat"
data13= resort(scipy.io.loadmat(filename)['epochs_data'])
data13_1= data13.reshape([-1, 500, 63])

data14 = np.vstack((data0_1 , data1_1))
data15 = np.vstack((data14, data2_1))
data16 = np.vstack((data15, data3_2))
data17 = np.vstack((data16, data4_1))
data18 = np.vstack((data17, data5_1))
data19 = np.vstack((data18, data6_1))
data20 = np.vstack((data19, data7_1))
data21 = np.vstack((data20, data8_1))
data22 = np.vstack((data21, data9_1))
data23 = np.vstack((data22, data10_1))
data24 = np.vstack((data23, data11_1))
data25 = np.vstack((data24, data12_1))
epochs_7 = np.vstack((data25, data13_1))

print(epochs_7.shape)
#print(epochs)
epochs_7 = epochs_7.reshape([-1, 500, 63])
print("epochs_7",epochs_7.shape)

dir_name="../../../opt/yidongyingpan/cmx_bishe/epoch/水果/橘子/"
file="org"

filename = dir_name+"0420-1-ver1-2-1-"+file+".mat"
data6 = resort(scipy.io.loadmat(filename)['epochs_data'])
print("data6 shape",data6.shape)
filename = dir_name+"0418-1-ver1-2-1-"+file+".mat"			# data shape: 551*50*64
data3 = resort(scipy.io.loadmat(filename)['epochs_data'])
print("data3 shape",data3.shape)

filename = dir_name+"0416-1-ver1-2-1-"+file+".mat"			# data shape: 551*50*63
data0 = resort(scipy.io.loadmat(filename)['epochs_data'])
data0_1 = data0.reshape([-1, 500, 63])
filename = dir_name+"0417-1-ver1-2-1-"+file+".mat"
data1 = resort(scipy.io.loadmat(filename)['epochs_data'])
data1_1 = data1.reshape([-1, 500, 63])
filename = dir_name+"0417-2-ver1-2-1-"+file+".mat"
data2 = resort(scipy.io.loadmat(filename)['epochs_data'])
data2_1 = data2.reshape([-1, 500, 63])
filename = dir_name+"0418-1-ver1-2-1-"+file+".mat"			# data shape: 551*50*64		eog
data3 = resort(scipy.io.loadmat(filename)['epochs_data'])
data3_1 = np.delete(data3,63,2)
data3_2 = data3_1.reshape([-1, 500, 63])
filename = dir_name+"0419-1-ver1-2-1-"+file+".mat"
data4 = resort(scipy.io.loadmat(filename)['epochs_data'])
data4_1 = data4.reshape([-1, 500, 63])
filename = dir_name+"0419-2-ver1-2-1-"+file+".mat"
data5 = resort(scipy.io.loadmat(filename)['epochs_data'])
data5_1 = data5.reshape([-1, 500, 63])
filename = dir_name+"0420-1-ver1-2-1-"+file+".mat"			# data shape: 551*37*63		数据样本: 576
data6 = resort(scipy.io.loadmat(filename)['epochs_data'])
data6_1 = data6.reshape([-1, 500, 63])
filename = dir_name+"0420-2-ver1-2-1-"+file+".mat"
data7 = resort(scipy.io.loadmat(filename)['epochs_data'])
data7_1 = data7.reshape([-1, 500, 63])
filename = dir_name+"0421-1-ver1-2-1-"+file+".mat"
data8 = resort(scipy.io.loadmat(filename)['epochs_data'])
data8_1 = data8.reshape([-1, 500, 63])
filename = dir_name+"0421-2-ver1-2-1-"+file+".mat"
data9 = resort(scipy.io.loadmat(filename)['epochs_data'])
data9_1 = data9.reshape([-1, 500, 63])
filename = dir_name+"0422-1-ver1-2-1-"+file+".mat"
data10= resort(scipy.io.loadmat(filename)['epochs_data'])
data10_1= data10.reshape([-1, 500, 63])
filename = dir_name+"0422-2-ver1-2-1-"+file+".mat"
data11= resort(scipy.io.loadmat(filename)['epochs_data'])
data11_1= data11.reshape([-1, 500, 63])
filename = dir_name+"0423-1-ver1-2-1-"+file+".mat"
data12= resort(scipy.io.loadmat(filename)['epochs_data'])
data12_1= data12.reshape([-1, 500, 63])
filename = dir_name+"0423-2-ver1-2-1-"+file+".mat"
data13= resort(scipy.io.loadmat(filename)['epochs_data'])
data13_1= data13.reshape([-1, 500, 63])

data14 = np.vstack((data0_1 , data1_1))
data15 = np.vstack((data14, data2_1))
data16 = np.vstack((data15, data3_2))
data17 = np.vstack((data16, data4_1))
data18 = np.vstack((data17, data5_1))
data19 = np.vstack((data18, data6_1))
data20 = np.vstack((data19, data7_1))
data21 = np.vstack((data20, data8_1))
data22 = np.vstack((data21, data9_1))
data23 = np.vstack((data22, data10_1))
data24 = np.vstack((data23, data11_1))
data25 = np.vstack((data24, data12_1))
epochs_8 = np.vstack((data25, data13_1))

epochs_8 = np.vstack((epochs_7, epochs_8))
epochs_8 = epochs_8.reshape([-1, 500, 63])
print("epochs_8",epochs_8.shape)

dir_name="../../../opt/yidongyingpan/cmx_bishe/epoch/水果/西瓜/"
file="wtm"

filename = dir_name+"0420-1-ver1-2-1-"+file+".mat"
data6 = resort(scipy.io.loadmat(filename)['epochs_data'])
print("data6 shape",data6.shape)
filename = dir_name+"0418-1-ver1-2-1-"+file+".mat"			# data shape: 551*50*64
data3 = resort(scipy.io.loadmat(filename)['epochs_data'])
print("data3 shape",data3.shape)

filename = dir_name+"0416-1-ver1-2-1-"+file+".mat"			# data shape: 551*50*63
data0 = resort(scipy.io.loadmat(filename)['epochs_data'])
data0_1 = data0.reshape([-1, 500, 63])
filename = dir_name+"0417-1-ver1-2-1-"+file+".mat"
data1 = resort(scipy.io.loadmat(filename)['epochs_data'])
data1_1 = data1.reshape([-1, 500, 63])
filename = dir_name+"0417-2-ver1-2-1-"+file+".mat"
data2 = resort(scipy.io.loadmat(filename)['epochs_data'])
data2_1 = data2.reshape([-1, 500, 63])
filename = dir_name+"0418-1-ver1-2-1-"+file+".mat"			# data shape: 551*50*64		eog
data3 = resort(scipy.io.loadmat(filename)['epochs_data'])
data3_1 = np.delete(data3,63,2)
data3_2 = data3_1.reshape([-1, 500, 63])
filename = dir_name+"0419-1-ver1-2-1-"+file+".mat"
data4 = resort(scipy.io.loadmat(filename)['epochs_data'])
data4_1 = data4.reshape([-1, 500, 63])
filename = dir_name+"0419-2-ver1-2-1-"+file+".mat"
data5 = resort(scipy.io.loadmat(filename)['epochs_data'])
data5_1 = data5.reshape([-1, 500, 63])
filename = dir_name+"0420-1-ver1-2-1-"+file+".mat"			# data shape: 551*37*63		数据样本: 576
data6 = resort(scipy.io.loadmat(filename)['epochs_data'])
data6_1 = data6.reshape([-1, 500, 63])
filename = dir_name+"0420-2-ver1-2-1-"+file+".mat"
data7 = resort(scipy.io.loadmat(filename)['epochs_data'])
data7_1 = data7.reshape([-1, 500, 63])
filename = dir_name+"0421-1-ver1-2-1-"+file+".mat"
data8 = resort(scipy.io.loadmat(filename)['epochs_data'])
data8_1 = data8.reshape([-1, 500, 63])
filename = dir_name+"0421-2-ver1-2-1-"+file+".mat"
data9 = resort(scipy.io.loadmat(filename)['epochs_data'])
data9_1 = data9.reshape([-1, 500, 63])
filename = dir_name+"0422-1-ver1-2-1-"+file+".mat"
data10= resort(scipy.io.loadmat(filename)['epochs_data'])
data10_1= data10.reshape([-1, 500, 63])
filename = dir_name+"0422-2-ver1-2-1-"+file+".mat"
data11= resort(scipy.io.loadmat(filename)['epochs_data'])
data11_1= data11.reshape([-1, 500, 63])
filename = dir_name+"0423-1-ver1-2-1-"+file+".mat"
data12= resort(scipy.io.loadmat(filename)['epochs_data'])
data12_1= data12.reshape([-1, 500, 63])
filename = dir_name+"0423-2-ver1-2-1-"+file+".mat"
data13= resort(scipy.io.loadmat(filename)['epochs_data'])
data13_1= data13.reshape([-1, 500, 63])

data14 = np.vstack((data0_1 , data1_1))
data15 = np.vstack((data14, data2_1))
data16 = np.vstack((data15, data3_2))
data17 = np.vstack((data16, data4_1))
data18 = np.vstack((data17, data5_1))
data19 = np.vstack((data18, data6_1))
data20 = np.vstack((data19, data7_1))
data21 = np.vstack((data20, data8_1))
data22 = np.vstack((data21, data9_1))
data23 = np.vstack((data22, data10_1))
data24 = np.vstack((data23, data11_1))
data25 = np.vstack((data24, data12_1))
epochs_9 = np.vstack((data25, data13_1))

epochs_9 = np.vstack((epochs_8, epochs_9))
epochs_9 = epochs_9.reshape([-1, 500, 63])
print("epochs_9",epochs_9.shape)

dir_name="../../../opt/yidongyingpan/cmx_bishe/epoch/运动/羽毛球/"
file="btm"

filename = dir_name+"0420-1-ver1-2-1-"+file+".mat"
data6 = resort(scipy.io.loadmat(filename)['epochs_data'])
print("data6 shape",data6.shape)
filename = dir_name+"0418-1-ver1-2-1-"+file+".mat"			# data shape: 551*50*64
data3 = resort(scipy.io.loadmat(filename)['epochs_data'])
print("data3 shape",data3.shape)

filename = dir_name+"0416-1-ver1-2-1-"+file+".mat"			# data shape: 551*50*63
data0 = resort(scipy.io.loadmat(filename)['epochs_data'])
data0_1 = data0.reshape([-1, 500, 63])
filename = dir_name+"0417-1-ver1-2-1-"+file+".mat"
data1 = resort(scipy.io.loadmat(filename)['epochs_data'])
data1_1 = data1.reshape([-1, 500, 63])
filename = dir_name+"0417-2-ver1-2-1-"+file+".mat"
data2 = resort(scipy.io.loadmat(filename)['epochs_data'])
data2_1 = data2.reshape([-1, 500, 63])
filename = dir_name+"0418-1-ver1-2-1-"+file+".mat"			# data shape: 551*50*64		eog
data3 = resort(scipy.io.loadmat(filename)['epochs_data'])
data3_1 = np.delete(data3,63,2)
data3_2 = data3_1.reshape([-1, 500, 63])
filename = dir_name+"0419-1-ver1-2-1-"+file+".mat"
data4 = resort(scipy.io.loadmat(filename)['epochs_data'])
data4_1 = data4.reshape([-1, 500, 63])
filename = dir_name+"0419-2-ver1-2-1-"+file+".mat"
data5 = resort(scipy.io.loadmat(filename)['epochs_data'])
data5_1 = data5.reshape([-1, 500, 63])
filename = dir_name+"0420-1-ver1-2-1-"+file+".mat"			# data shape: 551*37*63		数据样本: 576
data6 = resort(scipy.io.loadmat(filename)['epochs_data'])
data6_1 = data6.reshape([-1, 500, 63])
filename = dir_name+"0420-2-ver1-2-1-"+file+".mat"
data7 = resort(scipy.io.loadmat(filename)['epochs_data'])
data7_1 = data7.reshape([-1, 500, 63])
filename = dir_name+"0421-1-ver1-2-1-"+file+".mat"
data8 = resort(scipy.io.loadmat(filename)['epochs_data'])
data8_1 = data8.reshape([-1, 500, 63])
filename = dir_name+"0421-2-ver1-2-1-"+file+".mat"
data9 = resort(scipy.io.loadmat(filename)['epochs_data'])
data9_1 = data9.reshape([-1, 500, 63])
filename = dir_name+"0422-1-ver1-2-1-"+file+".mat"
data10= resort(scipy.io.loadmat(filename)['epochs_data'])
data10_1= data10.reshape([-1, 500, 63])
filename = dir_name+"0422-2-ver1-2-1-"+file+".mat"
data11= resort(scipy.io.loadmat(filename)['epochs_data'])
data11_1= data11.reshape([-1, 500, 63])
filename = dir_name+"0423-1-ver1-2-1-"+file+".mat"
data12= resort(scipy.io.loadmat(filename)['epochs_data'])
data12_1= data12.reshape([-1, 500, 63])
filename = dir_name+"0423-2-ver1-2-1-"+file+".mat"
data13= resort(scipy.io.loadmat(filename)['epochs_data'])
data13_1= data13.reshape([-1, 500, 63])

data14 = np.vstack((data0_1 , data1_1))
data15 = np.vstack((data14, data2_1))
data16 = np.vstack((data15, data3_2))
data17 = np.vstack((data16, data4_1))
data18 = np.vstack((data17, data5_1))
data19 = np.vstack((data18, data6_1))
data20 = np.vstack((data19, data7_1))
data21 = np.vstack((data20, data8_1))
data22 = np.vstack((data21, data9_1))
data23 = np.vstack((data22, data10_1))
data24 = np.vstack((data23, data11_1))
data25 = np.vstack((data24, data12_1))
epochs_10 = np.vstack((data25, data13_1))

print(epochs_10.shape)
#print(epochs)
epochs_10 = epochs_10.reshape([-1, 500, 63])
print("epochs_10",epochs_10.shape)

dir_name="../../../opt/yidongyingpan/cmx_bishe/epoch/运动/游泳/"
file="swm"

filename = dir_name+"0420-1-ver1-2-1-"+file+".mat"
data6 = resort(scipy.io.loadmat(filename)['epochs_data'])
print("data6 shape",data6.shape)
filename = dir_name+"0418-1-ver1-2-1-"+file+".mat"			# data shape: 551*50*64
data3 = resort(scipy.io.loadmat(filename)['epochs_data'])
print("data3 shape",data3.shape)

filename = dir_name+"0416-1-ver1-2-1-"+file+".mat"			# data shape: 551*50*63
data0 = resort(scipy.io.loadmat(filename)['epochs_data'])
data0_1 = data0.reshape([-1, 500, 63])
filename = dir_name+"0417-1-ver1-2-1-"+file+".mat"
data1 = resort(scipy.io.loadmat(filename)['epochs_data'])
data1_1 = data1.reshape([-1, 500, 63])
filename = dir_name+"0417-2-ver1-2-1-"+file+".mat"
data2 = resort(scipy.io.loadmat(filename)['epochs_data'])
data2_1 = data2.reshape([-1, 500, 63])
filename = dir_name+"0418-1-ver1-2-1-"+file+".mat"			# data shape: 551*50*64		eog
data3 = resort(scipy.io.loadmat(filename)['epochs_data'])
data3_1 = np.delete(data3,63,2)
data3_2 = data3_1.reshape([-1, 500, 63])
filename = dir_name+"0419-1-ver1-2-1-"+file+".mat"
data4 = resort(scipy.io.loadmat(filename)['epochs_data'])
data4_1 = data4.reshape([-1, 500, 63])
filename = dir_name+"0419-2-ver1-2-1-"+file+".mat"
data5 = resort(scipy.io.loadmat(filename)['epochs_data'])
data5_1 = data5.reshape([-1, 500, 63])
filename = dir_name+"0420-1-ver1-2-1-"+file+".mat"			# data shape: 551*37*63		数据样本: 576
data6 = resort(scipy.io.loadmat(filename)['epochs_data'])
data6_1 = data6.reshape([-1, 500, 63])
filename = dir_name+"0420-2-ver1-2-1-"+file+".mat"
data7 = resort(scipy.io.loadmat(filename)['epochs_data'])
data7_1 = data7.reshape([-1, 500, 63])
filename = dir_name+"0421-1-ver1-2-1-"+file+".mat"
data8 = resort(scipy.io.loadmat(filename)['epochs_data'])
data8_1 = data8.reshape([-1, 500, 63])
filename = dir_name+"0421-2-ver1-2-1-"+file+".mat"
data9 = resort(scipy.io.loadmat(filename)['epochs_data'])
data9_1 = data9.reshape([-1, 500, 63])
filename = dir_name+"0422-1-ver1-2-1-"+file+".mat"
data10= resort(scipy.io.loadmat(filename)['epochs_data'])
data10_1= data10.reshape([-1, 500, 63])
filename = dir_name+"0422-2-ver1-2-1-"+file+".mat"
data11= resort(scipy.io.loadmat(filename)['epochs_data'])
data11_1= data11.reshape([-1, 500, 63])
filename = dir_name+"0423-1-ver1-2-1-"+file+".mat"
data12= resort(scipy.io.loadmat(filename)['epochs_data'])
data12_1= data12.reshape([-1, 500, 63])
filename = dir_name+"0423-2-ver1-2-1-"+file+".mat"
data13= resort(scipy.io.loadmat(filename)['epochs_data'])
data13_1= data13.reshape([-1, 500, 63])

data14 = np.vstack((data0_1 , data1_1))
data15 = np.vstack((data14, data2_1))
data16 = np.vstack((data15, data3_2))
data17 = np.vstack((data16, data4_1))
data18 = np.vstack((data17, data5_1))
data19 = np.vstack((data18, data6_1))
data20 = np.vstack((data19, data7_1))
data21 = np.vstack((data20, data8_1))
data22 = np.vstack((data21, data9_1))
data23 = np.vstack((data22, data10_1))
data24 = np.vstack((data23, data11_1))
data25 = np.vstack((data24, data12_1))
epochs_11 = np.vstack((data25, data13_1))

epochs_11 = np.vstack((epochs_10, epochs_11))
epochs_11 = epochs_11.reshape([-1, 500, 63])
print("epochs_11",epochs_11.shape)

dir_name="../../../opt/yidongyingpan/cmx_bishe/epoch/运动/篮球/"
file="bsk"

filename = dir_name+"0420-1-ver1-2-1-"+file+".mat"
data6 = resort(scipy.io.loadmat(filename)['epochs_data'])
print("data6 shape",data6.shape)
filename = dir_name+"0418-1-ver1-2-1-"+file+".mat"			# data shape: 551*50*64
data3 = resort(scipy.io.loadmat(filename)['epochs_data'])
print("data3 shape",data3.shape)

filename = dir_name+"0416-1-ver1-2-1-"+file+".mat"			# data shape: 551*50*63
data0 = resort(scipy.io.loadmat(filename)['epochs_data'])
data0_1 = data0.reshape([-1, 500, 63])
filename = dir_name+"0417-1-ver1-2-1-"+file+".mat"
data1 = resort(scipy.io.loadmat(filename)['epochs_data'])
data1_1 = data1.reshape([-1, 500, 63])
filename = dir_name+"0417-2-ver1-2-1-"+file+".mat"
data2 = resort(scipy.io.loadmat(filename)['epochs_data'])
data2_1 = data2.reshape([-1, 500, 63])
filename = dir_name+"0418-1-ver1-2-1-"+file+".mat"			# data shape: 551*50*64		eog
data3 = resort(scipy.io.loadmat(filename)['epochs_data'])
data3_1 = np.delete(data3,63,2)
data3_2 = data3_1.reshape([-1, 500, 63])
filename = dir_name+"0419-1-ver1-2-1-"+file+".mat"
data4 = resort(scipy.io.loadmat(filename)['epochs_data'])
data4_1 = data4.reshape([-1, 500, 63])
filename = dir_name+"0419-2-ver1-2-1-"+file+".mat"
data5 = resort(scipy.io.loadmat(filename)['epochs_data'])
data5_1 = data5.reshape([-1, 500, 63])
filename = dir_name+"0420-1-ver1-2-1-"+file+".mat"			# data shape: 551*37*63		数据样本: 576
data6 = resort(scipy.io.loadmat(filename)['epochs_data'])
data6_1 = data6.reshape([-1, 500, 63])
filename = dir_name+"0420-2-ver1-2-1-"+file+".mat"
data7 = resort(scipy.io.loadmat(filename)['epochs_data'])
data7_1 = data7.reshape([-1, 500, 63])
filename = dir_name+"0421-1-ver1-2-1-"+file+".mat"
data8 = resort(scipy.io.loadmat(filename)['epochs_data'])
data8_1 = data8.reshape([-1, 500, 63])
filename = dir_name+"0421-2-ver1-2-1-"+file+".mat"
data9 = resort(scipy.io.loadmat(filename)['epochs_data'])
data9_1 = data9.reshape([-1, 500, 63])
filename = dir_name+"0422-1-ver1-2-1-"+file+".mat"
data10= resort(scipy.io.loadmat(filename)['epochs_data'])
data10_1= data10.reshape([-1, 500, 63])
filename = dir_name+"0422-2-ver1-2-1-"+file+".mat"
data11= resort(scipy.io.loadmat(filename)['epochs_data'])
data11_1= data11.reshape([-1, 500, 63])
filename = dir_name+"0423-1-ver1-2-1-"+file+".mat"
data12= resort(scipy.io.loadmat(filename)['epochs_data'])
data12_1= data12.reshape([-1, 500, 63])
filename = dir_name+"0423-2-ver1-2-1-"+file+".mat"
data13= resort(scipy.io.loadmat(filename)['epochs_data'])
data13_1= data13.reshape([-1, 500, 63])

data14 = np.vstack((data0_1 , data1_1))
data15 = np.vstack((data14, data2_1))
data16 = np.vstack((data15, data3_2))
data17 = np.vstack((data16, data4_1))
data18 = np.vstack((data17, data5_1))
data19 = np.vstack((data18, data6_1))
data20 = np.vstack((data19, data7_1))
data21 = np.vstack((data20, data8_1))
data22 = np.vstack((data21, data9_1))
data23 = np.vstack((data22, data10_1))
data24 = np.vstack((data23, data11_1))
data25 = np.vstack((data24, data12_1))
epochs_12 = np.vstack((data25, data13_1))

epochs_12 = np.vstack((epochs_11, epochs_12))
epochs_12 = epochs_12.reshape([-1, 500, 63])
print("epochs_12",epochs_12.shape)

# Hyper Parameters
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# VISUALIZATION = True
VISUALIZATION = False
CLASSES_NUM = 4
EPOCH = 200
BATCH_SIZE = 40
LR = 3e-4

# get model
eeg_net = EEGNet(classes_num=CLASSES_NUM).to(DEVICE)

optimizer = torch.optim.Adam(eeg_net.parameters(), lr=LR)
loss_func = nn.CrossEntropyLoss().to(DEVICE)

print("epochs_3",epochs_3.shape)
print("epochs_6",epochs_6.shape)
print("epochs_9",epochs_9.shape)
print("epochs_12",epochs_12.shape)

epochs_3_train = epochs_3[:1750,:,:]
epochs_3_test = epochs_3[1750:2100,:,:]
print("epochs_3_train",epochs_3_train.shape)
print("epochs_3_test",epochs_3_test.shape)

epochs_6_train = epochs_6[:1750,:,:]
epochs_6_test = epochs_6[1750:2100,:,:]
print("epochs_6_train",epochs_6_train.shape)
print("epochs_6_test",epochs_6_test.shape)

epochs_9_train = epochs_9[:1750,:,:]
epochs_9_test = epochs_9[1750:2100,:,:]
print("epochs_9_train",epochs_9_train.shape)
print("epochs_9_test",epochs_9_test.shape)

epochs_12_train = epochs_12[:1750,:,:]
epochs_12_test = epochs_12[1750:2100,:,:]
print("epochs_12_train",epochs_12_train.shape)
print("epochs_12_test",epochs_12_test.shape)


epochs_train= np.vstack((epochs_3_train,epochs_6_train, epochs_9_train,epochs_12_train))
epochs_test = np.vstack((epochs_3_test, epochs_6_test, epochs_9_test, epochs_12_test))
print("epochs_train",epochs_train.shape)
print("epochs_test",epochs_test.shape)

train_labels = np.zeros(7000, dtype=np.int64)
train_labels[1750:3500] = 1
train_labels[3500:5250] = 2
train_labels[5250:70000] = 3

test_labels = np.zeros(1400, dtype=np.int64)
test_labels[350:700] = 1
test_labels[700:1050] = 2
test_labels[1050:1400] = 3

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