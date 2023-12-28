from model.Res2Unet_baseV2 import MSFAUnet_baseV2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class PPIBlur(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(PPIBlur, self).__init__()
        kernel = [[1, 2, 2, 2, 1],
                  [2, 4, 4, 4, 2],
                  [2, 4, 4, 4, 2],
                  [2, 4, 4, 4, 2],
                  [1, 2, 2, 2, 1]]
        kernel = np.array(kernel,dtype=float)
        kernel *= 1 / 64
        w,h = kernel.shape

        kernel = torch.FloatTensor(kernel).expand(out_channels, in_channels, h, w)
        self.weight = nn.Parameter(data=kernel, requires_grad=False)
        self.padding = w//2

    def forward(self, x):

        x = F.conv2d(x,self.weight,padding=self.padding)

        return x

    def blur(self, tensor_image):
        kernel = [[1, 2, 2, 2, 1],
                  [2, 4, 4, 4, 2],
                  [2, 4, 4, 4, 2],
                  [2, 4, 4, 4, 2],
                  [1, 2, 2, 2, 1]]
        kernel*=1/64
        kernel = np.array(list).shape
        min_batch = tensor_image.size()[0]
        channels = tensor_image.size()[1]
        out_channel = channels
        kernel = torch.FloatTensor(kernel).expand(out_channel, channels, 5, 5)
        self.weight = nn.Parameter(data=kernel, requires_grad=False)

        return F.conv2d(tensor_image, self.weight, 1, 1)

class SobelBlur(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(SobelBlur, self).__init__()
        kernel_x = [[-1, 0, 1],
                    [-2, 0, 2],
                    [-1, 0, 1]]

        kernel_y = [[-1, -2, -1],
                    [0, 0, 0],
                    [1, 2, 1]]
        kernel_x = np.array(kernel_x,dtype=float)
        kernel_y = np.array(kernel_y,dtype=float)
        w,h = kernel_x.shape

        kernel_x = torch.FloatTensor(kernel_x).expand(out_channels, in_channels, h, w)
        kernel_y = torch.FloatTensor(kernel_y).expand(out_channels, in_channels, h, w)
        self.weight_x = nn.Parameter(data=kernel_x, requires_grad=False)
        self.weight_y = nn.Parameter(data=kernel_y, requires_grad=False)
        self.padding = w//2

    def forward(self, x):
        x_gx = F.conv2d(x,self.weight_x,padding=self.padding)
        x_gy = F.conv2d(x,self.weight_y,padding=self.padding)
        x = x_gx+x_gy
        return x
class PPIG(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(PPIG, self).__init__()
        channels = 8
        self.convs = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=channels, kernel_size=9, padding=4),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=7, padding=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=channels, out_channels=1, kernel_size=5, padding=2)
        )
        # 传统ppi
        self.ppiBlur = PPIBlur(in_channels,out_channels)
        # 边缘过滤器
        self.sobelBlur = SobelBlur(in_channels,out_channels)

    def forward(self,x):
        residual = self.convs(x)
        x = self.ppiBlur(x)
        x = x+residual
        x = self.sobelBlur(x)
        # repeat8通道
        #x = x.unsqueeze(1)
        x = x.repeat(1,8,1,1)
        return x
        #----------------计算传统PPI

class Res2Unet_PPIG(nn.Module):
    def __init__(self,in_channels,out_channels,input_shape,sigmoid_scale):
        super(Res2Unet_PPIG, self,).__init__()
        self.res2_unet = MSFAUnet_baseV2(in_channels,out_channels,input_shape,sigmoid_scale)
        self.ppig = PPIG(in_channels=1,out_channels=1)
        self.fusion = nn.Conv2d(in_channels,out_channels,kernel_size=3,padding=1)
    def forward(self,x_1,x_8):
        #8通道直接使用Res2Unet计算
        x_8 = self.res2_unet(x_8)
        # 单通道提取边缘后复制为8通道
        x_ppig = self.ppig(x_1)
        x_8 = x_8+x_ppig
        x_8 = self.fusion(x_8)
        return x_8


if __name__ == '__main__':
    batch_size = 2
    in_channels = 8
    out_channels = 8
    height = 480
    width = 636
    input_tensor_8 = torch.randn(batch_size, 8, height, width)
    input_tensor_1 = torch.randn(batch_size, 1, height, width)
    #input_tensor = torch.arange(0,196,dtype = torch.float).reshape(2,2,7,7)

    print("输入形状：\n",input_tensor_8, input_tensor_8.shape)
    net = Res2Unet_PPIG(in_channels, out_channels, input_tensor_8.shape,1.001)
    y = net(input_tensor_1,input_tensor_8)
    #ppi_bur = SobelBlur(in_channels,out_channels)
    print("网络形状为\n", net)
    #y = ppi_bur(input_tensor)
    print("输出形状：\n",y,y.shape)
