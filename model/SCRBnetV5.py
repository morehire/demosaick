import numpy as np
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
            nn.Conv2d(in_channels=out_channels,out_channels=out_channels,kernel_size=kernel_size,stride=1,padding=1),
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
        self.pooling = nn.MaxPool2d(kernel_size, stride=stride, padding=padding)
        # 1*1卷积
        self.conv1_1 = nn.Conv2d(in_channels, out_channels, 1)
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

        #x = self.conv3(x)
        x = self.pooling(x)
        x = self.conv1_1(x)
        x = torch.add(out,x)
        x = self.relu(x)
        return x


class Stagemodule(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(Stagemodule, self).__init__()
        # 构建3个eac块
        self.eca_conv = nn.ModuleList([
            ECAblock(in_channels, out_channels, kernel_size=3,padding=1,stride=stride)
            for in_channels,stride in zip([in_channels,out_channels,out_channels],[2,1,1])
        ])
    def forward(self,x):
        for eca_block in self.eca_conv:
            x = eca_block(x)
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
    def __init__(self,in_channels,out_channels_list,fc_out_channels,convList_inchannels,convList_outchannels):
        super(FEmodule, self).__init__()

        out_channels_list=out_channels_list
        fc_out_channels = fc_out_channels

        self.stage = nn.ModuleList([
            Stagemodule(in_channels,out_channels) for in_channels,out_channels in zip([in_channels,64,128,256,384,512],out_channels_list)
        ])
        self.fc_mapping = nn.ModuleList([
            LinerMapping(in_channel,out_channels=out_channel) for in_channel,out_channel in zip(out_channels_list,fc_out_channels)
        ])
        self.conv = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
            for in_channels,out_channels in zip(convList_inchannels,convList_outchannels)
        ])

    def forward(self,x):
        outputs_stage = []
        x_p = x
        for stage in self.stage:
            x_p = stage(x_p)
            outputs_stage.append(x_p)
        outputs = [fc(xi) for fc,xi in zip(self.fc_mapping,outputs_stage)]

        x1,x2,x3,x4,x5,x6 = [x for x in outputs]

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
        x6 = x6*temp

        x = torch.cat((x1,x2,x3,x4,x5,x6),dim=1)
        return x

class HSRModule(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size):
        super(HSRModule, self).__init__()
        out = 128
        self.conv1_1 = nn.Conv2d(in_channels,out,1)
        self.conv1 = nn.Sequential(
            nn.Conv2d(out,out//4,kernel_size,padding=1),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out//4, out//4, kernel_size, padding=1),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(out//4, out//4, kernel_size, padding=1),
            nn.ReLU(inplace=True)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(out//4,out//4,kernel_size,padding=1),
            nn.ReLU(inplace=True)
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(out//4,out_channels,kernel_size,padding=1),
            nn.ReLU(inplace=True)
        )
        self.conv6 = nn.Conv2d(out_channels,out_channels,kernel_size,padding=1)
        self.conv7 = nn.Conv2d(out_channels,out_channels,kernel_size,padding=1)

    def forward(self,x):
        x = self.conv1_1(x)
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

class SCRBnetV5(nn.Module):
    def __init__(self,in_channels,out_channels, masks, scale=1.001):
        super(SCRBnetV5, self).__init__()
        #self.fe_number = fe_number
        # 定义stage模块的输出通道
        out_channels_list = [64, 128, 256, 384, 512, 768]
        # 定义6个全连接的输出通道
        fc_out_channels = [1, 1, 8, 16, 32, 64]
        # 定义 8 个卷积操作的输入输出通道
        convList_inchannels = [1, 1, 8, 16, 1, 1, 1, 1]
        convList_outchannels = [8, 8, 16, 16, 8, 16, 32, 64]
        self.fe = FEmodule(in_channels,out_channels_list,fc_out_channels,convList_inchannels,convList_outchannels)
        #self.hsr = HSRModule(in_channels*self.fe_number, out_channels,kernel_size=3)
        self.hsr = HSRModule(144, out_channels,kernel_size=3)
        self.scale = scale
        self.masks = masks

    def forward(self, x):
        N = x.shape[0]
        masks = self.masks.expand(N, -1, -1, -1)
        x = self.fe(x)
        x = self.hsr(x)
        x = torch.sigmoid(x)*self.scale

        #x1[masks] = (x1[masks]+x[masks])*0.5
        return x


if __name__ == '__main__':
    from thop import profile
    batch_size = 4
    in_channels = 1
    height = 480
    width = 636
    masks = np.load("../masks.npy")
    masks_tensor = torch.from_numpy(masks)
    input_tensor = torch.randn(batch_size, in_channels, height, width)
    #fe = FEmodule(in_channels, 64)
    #print(fe)
    net = SCRBnetV5(in_channels=1, out_channels=8,masks=masks_tensor)
    input = torch.randn(1, 1, 480, 636)
    flops, params = profile(net, inputs=(input,))
    print('flops:{}'.format(flops))
    print('params:{}'.format(params))

    # 定义stage模块的输出通道
    out_channels_list = [64, 128, 256, 384, 512, 768]
    # 定义6个全连接的输出通道
    fc_out_channels = [1, 1, 8, 16, 32, 64]
    # 定义 8 个卷积操作的输入输出通道
    convList_inchannels = [1, 1, 8, 16, 1, 1, 1, 1]
    convList_outchannels = [8, 8, 16, 16, 8, 16, 32, 64]
    fe = FEmodule(1,out_channels_list,fc_out_channels,convList_inchannels,convList_outchannels)
    flops, params = profile(fe, inputs=(input,))
    print('fe_flops:{}'.format(flops))
    print('fe_params:{}'.format(params))

    hsr = HSRModule(144, 8,3)
    input = torch.randn(1, 144, 480, 636)
    flops, params = profile(hsr, inputs=(input,))
    print('fe_flops:{}'.format(flops))
    print('fe_params:{}'.format(params))
    print(net)
    y = net(input_tensor)
    print(y.shape)




