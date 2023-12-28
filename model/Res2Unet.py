import math

import torch
import torch.nn as nn
import torch.nn.functional as F
class SE_block(nn.Module):
    def __init__(self,in_channels,ratio = 4):
        super(SE_block, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels,in_channels//4),
            nn.ReLU(),
            nn.Linear(in_channels//4,in_channels)
        )
        self.sigmoid = nn.Sigmoid()
    def forward(self,x):
        residual = x
        x = self.avg_pool(x)
        x = x.view(x.size(0),-1)
        x = self.fc(x)
        x = x.view(x.size(0),-1,1,1)
        x = self.sigmoid(x)
        x = x*residual
        return x


class Bottle2neck(nn.Module):
    expansion = 4

    def __init__(self, in_channels,out_channels, stride=1,scale=6,base_width=32):
        """ Constructor
        Args:
            inplanes: input channel dimensionality
            planes: output channel dimensionality
            stride: conv stride. Replaces pooling layer.
            downsample: None when stride = 1
            baseWidth: basic width of conv3x3
            scale: number of scale.
            type: 'normal': normal set. 'stage': first block of a new stage.
        """
        super(Bottle2neck, self).__init__()

        width = int(math.floor(out_channels * (base_width / 64.0)))
        self.conv1 = nn.Conv2d(in_channels, width * scale, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width * scale)
        # 3*3卷积个数
        if scale == 1:
            self.nums = 1
        else:
            self.nums = scale - 1

        convs = []
        bns = []
        for i in range(self.nums):
            convs.append(nn.Conv2d(width, width, kernel_size=3, stride=stride, padding=1, bias=False))
            bns.append(nn.BatchNorm2d(width))
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)
        # 1*1 卷积
        self.conv3 = nn.Conv2d(width * scale, in_channels, kernel_size=1, bias=False)

        self.bn3 = nn.BatchNorm2d(in_channels)
        self.relu = nn.LeakyReLU(inplace=True)
        self.scale = scale
        self.se = SE_block(in_channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        b,c,h,w = out.shape
        spx = torch.split(out, c//self.scale, dim=1)
        #out = []
        sp = torch.zeros(out.shape)
        for i in range(0,self.scale):
            if i ==0:
                out = spx[i]
                continue
            elif i==1:
                sp = self.convs[i-1](spx[i])
                sp = self.relu(self.bns[i-1](sp))
            else:
                sp = self.convs[i-1](spx[i]+sp)
                sp =self.relu(self.bns[i-1](sp))
            out = torch.cat((sp,out),dim=1)
        #out = torch.cat(out,dim=1)
        out = self.conv3(out)
        out = self.se(out)
        # 残差
        out = x+out
        return out

class BaseLayer(nn.Module):
    def __init__(self,in_channels,out_channels,res_number,sample_type,factor,kernel_size=3,padding=1):
        super(BaseLayer, self).__init__()
        #out_channels//=2
        self.conv = nn.Conv2d(in_channels,out_channels,kernel_size=kernel_size,padding=padding)
        self.res2net = nn.ModuleList([
            Bottle2neck(out_channels,out_channels,scale=6) for _ in range(res_number)
        ])
        self.sample_type = sample_type
        if sample_type == 'up':
            self.sample = nn.PixelUnshuffle(downscale_factor=factor)
        elif sample_type == 'down':
            self.sample = nn.PixelShuffle(upscale_factor=factor)
        else:
            pass

    def forward(self,x):
        x = self.conv(x)
        for res2net in self.res2net:
            x = res2net(x)
        if self.sample_type != 'None':
            x1 = self.sample(x)
            if self.sample_type == 'down':
                return x1
            return x,x1
        return x

class MSFAUnet(nn.Module):

    def __init__(self,in_channels,out_channels,input_shape,sigmoid_scale):
        """

        :param in_channels: 8
        :param out_channels: 8
        :param input_shape: b,c,w,h
        """
        super(MSFAUnet, self).__init__()
        self.sigmoid_scale = nn.Parameter(torch.tensor(sigmoid_scale))
        # 8->32,8->128
        self.layer1 = BaseLayer(in_channels,32,res_number=2,sample_type='up',factor=2)
        #self.sample1 = nn.PixelShuffle

        in_channels = 32
        in_channels*=4 #128
        # 128->64;128->256
        self.layer2 = BaseLayer(in_channels,in_channels//2,res_number=1,sample_type='up',factor=2)

        in_channels *= 2#256
        # 256->128,256->512
        self.layer3 = BaseLayer(in_channels,in_channels//2,res_number=1,sample_type='up',factor=2)

        in_channels *=2#512
        #512->256
        self.layer4  = BaseLayer(in_channels,in_channels//2,res_number=1,sample_type='None',factor=2)

        #上采样
        in_channels //=2
        # 256->256,256->64
        self.layer5  = BaseLayer(in_channels,in_channels,res_number=1,sample_type='down',factor=2,kernel_size=1,padding=0)

        in_channels = 192
        #192->128->32
        self.layer6  = BaseLayer(in_channels,128,res_number=1,sample_type='down',factor=2)
        in_channels = 96 #(32+64)
        #96->48->12
        self.layer7  = BaseLayer(in_channels,48,res_number=1,sample_type='down',factor=2)
        in_channels = 44 #(32+12)

        self.conv2 = nn.Sequential(
            # 64->32
            nn.Conv2d(in_channels,32,kernel_size=3,padding=1,stride=1),
            # 32->8
            nn.Conv2d(32,out_channels,kernel_size=3,padding=1,stride=1)
        )

    def forward(self,x):
        #x = self.Mykernel(x)
        out1_1,out1_2 = self.layer1(x)
        out2_1,out2_2 = self.layer2(out1_2)
        pad = nn.ZeroPad2d((1,0,0,0))
        out2_2 = pad(out2_2)
        out3_1,out3_2 = self.layer3(out2_2)

        out = self.layer4(out3_2)
        out= self.layer5(out)
        out = torch.cat((out,out3_1),dim=1)
        out = self.layer6(out)
        out = out[:,:,:,2:]
        out = torch.cat((out,out2_1),dim=1)
        out = self.layer7(out)
        out = torch.cat((out,out1_1),dim=1)

        out = self.conv2(out)
        # ----------------------------------
        out = torch.sigmoid(out)*self.sigmoid_scale
        return out

    def get_scale(self):
        return self.sigmoid_scale


if __name__ == '__main__':
    batch_size = 1
    in_channels = 8
    out_channels =8
    height = 480
    width = 636
    input_tensor = torch.randn(batch_size, in_channels, height, width)
    print("输入形状：\n",input_tensor.shape)
    net = MSFAUnet(in_channels, out_channels, input_tensor.shape,1.001)
    print("网络形状为\n",net)
    y = net(input_tensor)
    print("输出形状：\n",y.shape)