import torch
import torch.nn as nn


class ChannelAttention(nn.Module):
    def __init__(self, in_channels):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_channels, in_channels // 4, 1)
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv2d(in_channels // 4, in_channels, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class ECAblock(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size=3,padding=1,stride=1):
        super(ECAblock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels,out_channels,kernel_size=kernel_size,padding=padding,stride=stride),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=out_channels,out_channels=out_channels,kernel_size=kernel_size,stride=1,padding=padding),
            nn.BatchNorm2d(out_channels)
        )
        self.channel_wise_min = ChannelAttention(out_channels)

        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Sequential(
            nn.Linear(out_channels,out_channels),
            nn.ReLU(inplace=True),
            nn.Linear(out_channels,out_channels)
        )
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)

        # 改变通道数和大小
        self.pooling = nn.MaxPool2d(kernel_size,stride=stride,padding=padding)
        # 1*1卷积
        #self.conv1_1 = nn.Conv2d(in_channels,out_channels,1)
        #self.conv3= nn.Conv2d(in_channels,out_channels,kernel_size=kernel_size,padding=padding,stride=stride)

    def forward(self,x):

        out = self.conv1(x)
        out = self.conv2(out)
        b,c,w,h = out.shape

        out_1 = self.channel_wise_min(out)
        out_2 =  self.global_avg_pool(out)

        out_1 = out_1.view(out_1.size(0),-1)
        out_1 = self.fc(out_1)

        out_2 = out_2.view(out_2.size(0),-1)
        out_2 = self.fc(out_2)

        out_3 = torch.add(out_1,out_2)
        out_3 = self.sigmoid(out_3)
        out_3 = out_3.view(out_3.size(0),-1,1,1)
        out_3 = out_3.expand_as(out)

        out = out*out_3

        x = self.pooling(x)
        #x = self.conv1_1(x)
        x = torch.add(out,x)
        x = self.relu(x)
        return x

class Stagemodule(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size,padding):
        super(Stagemodule, self).__init__()
        # 构建3个eac块
        if in_channels<=8:
            out = in_channels
        else:
            out = in_channels//2
        self.conv1 = nn.Conv2d(in_channels,out,1)
        self.eca_conv = nn.ModuleList([
            ECAblock(out, out, kernel_size=kernel_size,padding=padding,stride=stride)
            for in_channels,stride in zip([out,out,out],[2,1,1])
        ])
        self.conv2 = nn.Conv2d(out,out_channels, 1)
    def forward(self,x):
        x = self.conv1(x)
        for eca_block in self.eca_conv:
            x = eca_block(x)
        x = self.conv2(x)
        return x

class LinerMapping(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(LinerMapping, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Linear(in_channels,out_channels)

    def forward(self,x):

        x = self.avg_pool(x)

        x = x.view(x.size(0),-1)
        x =self.fc(x)
        x = x.view(x.size(0),-1,1,1)
        return x

class FEmodule(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(FEmodule, self).__init__()

        #out_channels_list=[64,128,256,384,512,768]
        out_channels_list=[16,32,64,128,256,512]
        kernel_size = [13,11,9,7,5,3]
        padding = [6,5,4,3,2,1]
        self.stage = nn.ModuleList([
            Stagemodule(in_channels,out_channels,kernel_size,padding) for in_channels,out_channels,kernel_size,padding in zip([in_channels,16,32,64,128,256],out_channels_list,kernel_size,padding)
        ])
        self.fc_mapping = nn.ModuleList([
            LinerMapping(in_channel,out_channels=in_channels) for in_channel in out_channels_list
        ])

        self.conv = nn.ModuleList([
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1) for _ in range(8)
        ])

    def forward(self,x):
        outputs_stage = []
        x_p = x
        for stage in self.stage:
            x_p = stage(x_p)
            outputs_stage.append(x_p)
        outputs = [fc(xi) for fc,xi in zip(self.fc_mapping,outputs_stage)]

        x1,x2,x3,x4,x5,x6= [x for x in outputs]

        x1 = x * x1
        x1 = self.conv[0](x1)

        x2 = x * x2
        x2 = self.conv[1](x2)

        temp = self.conv[4](x)
        x3 = x3*temp
        x3 = self.conv[2](x3)

        temp = self.conv[5](x)
        x4 = x4 * temp
        x4 = self.conv[3](x4)

        temp = self.conv[6](x)
        x5 = x5 * temp

        temp = self.conv[7](x)
        x6 = x6 * temp
        x = torch.cat((x1,x2,x3,x4,x5,x6),dim=1)
        return x

class HSRModule(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size):
        super(HSRModule, self).__init__()
        self.conv7_7= nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 7, padding=3),
            nn.ReLU(inplace=True)
        )
        self.conv5_5 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 5, padding=2),
            nn.ReLU(inplace=True)
        )
        #------------------------------------
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels,in_channels,kernel_size,padding=1),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size, padding=1),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size, padding=1),
            nn.ReLU(inplace=True)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels,in_channels,kernel_size,padding=1),
            nn.ReLU(inplace=True)
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels,in_channels,kernel_size,padding=1),
            nn.ReLU(inplace=True)
        )
        self.conv6 = nn.Conv2d(in_channels,out_channels,kernel_size,padding=1)
        self.conv7 = nn.Conv2d(in_channels,out_channels,kernel_size,padding=1)

    def forward(self,x):
        residual1 = x
        residual2 = self.conv7_7(x)
        residual3 = self.conv5_5(x)
        x = torch.add(residual1,residual2)
        x = torch.add(x,residual3)
        #----------------------------------
        x = self.conv1(x)

        out2 = self.conv2(x)

        out3 = self.conv3(out2)

        out3 = torch.add(out3,out2)

        out3 = self.conv4(out3)

        x = torch.add(x,out3)
        x = self.conv5(x)

        out2 = self.conv6(x)
        out3 = self.conv7(x)
        x = (out2+out3)/2.0
        return x
class SCRBnetV3(nn.Module):
    def __init__(self,in_channels,out_channels, masks, fe_number=6, scale=1.001):
        super(SCRBnetV3, self).__init__()
        self.fe_number = fe_number
        self.fe = FEmodule(in_channels,out_channels)

        self.hsr = HSRModule(in_channels*self.fe_number, out_channels,kernel_size=3)
        # 加入可学习的缩放参数
        #self.scale = nn.Parameter(torch.tensor(scale))
        self.scale = scale
        self.masks = masks

    def forward(self, x):
        N = x.shape[0]
        masks = self.masks.expand(N, -1, -1, -1)

        x1 = self.fe(x)
        x1 = self.hsr(x1)

        # 将数据映射到(0,1)区间
        x1 = torch.sigmoid(x1)*self.scale

        x1[masks] = (x1[masks]+x[masks])*0.5
        return x1

if __name__ == '__main__':
    import numpy as np
    batch_size = 2
    in_channels = 8
    height = 480
    width = 636
    input_tensor = torch.randn(batch_size, in_channels, height, width)
    masks = np.load("../masks.npy")
    masks_tensor = torch.from_numpy(masks)
    #print(fe)
    net = SCRBnetV3(in_channels, out_channels=8,masks=masks_tensor,scale=1.001)

    from thop import profile

    input = torch.randn(1, 8, 480, 636)
    flops, params = profile(net, inputs=(input,))
    print('flops:{}'.format(flops))
    print('params:{}'.format(params))

    # print(net)
    # y = net(input_tensor)
    # print(y.shape)




