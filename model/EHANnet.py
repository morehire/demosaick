import torch
import torch.nn as nn


class ResidualChannelAttentionBlock(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(ResidualChannelAttentionBlock, self).__init__()
        self.conv_layer1 =nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        )
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.conv_layer2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        )
        self.sigmoid = nn.Sigmoid()
    def forward(self,x):
        out1 = self.conv_layer1(x)
        out2 = self.sigmoid(out1)
        out2 = self.conv_layer2(out2)
        out2 = self.sigmoid(out2)
        out1 = torch.multiply(out2,out1)
        x = torch.add(x,out1)
        return x

class LayerAttention(nn.Module):
    """
    层注意力模块
    """
    def __init__(self,alpha=1):
        super(LayerAttention, self).__init__()
        self.softmax  = nn.Softmax(dim=-1)
        # 尺度因子
        self.alpha = nn.Parameter(torch.tensor(alpha,dtype=torch.float32))

    def forward(self,x):
        (b,n,c,h,w) = x.shape
        # (n,hwc)
        out = x.view(b,n,-1)
        out_t = out.transpose(1, 2)

        #(b,n,n)
        # 批量矩阵乘法
        out_t = torch.bmm(out,out_t)
        #(b,n,n) 归一化
        out_t = self.softmax(out_t)
        #(b,n,hwc)
        # 批量矩阵乘法
        out = torch.bmm(out_t,out)
        out = out.view((b,n,c,h,w))
        x = x+out*self.alpha
        x = x.view(b,n*c,h,w)
        return x

class ChannelSpatialAttention(nn.Module):
    def __init__(self,in_channels,beta=1):
        super(ChannelSpatialAttention, self).__init__()
        self.softmax = nn.Softmax(dim=-1)
        self.beta = nn.Parameter(torch.tensor(beta,dtype=torch.float32))
        # 3D卷积
        self.conv3d = nn.Conv3d(1,1,kernel_size=(3,3,3),stride=1,padding=1)

    def forward(self,x):
        (b, c, h, w) = x.shape
        # 将通道维当做deep维，使用3D卷积
        out = x.unsqueeze(1)

        out = self.conv3d(out)
        out = self.softmax(out)

        # 变回原形状
        out = out.view(b,-1,h,w)
        # 尺度因子
        out  = self.beta*out
        out = torch.multiply(out,x)
        x = out + x
        return x

class EHANnet(nn.Module):

    def __init__(self,in_channels,out_channels,alpha,beta,rc_block_num=3,rg_numbers = 3):
        """

        :param in_channels:
        :param out_channels:
        :param alpha: lam层尺度因子，可学习参数，初始为0
        :param beta: csam层尺度因子，可学习参数，初始为0
        :param rc_block_num: 每个rg模块中rcab块的个数，默认为3
        :param rg_numbers: 模型中rg块的个数，默认为3
        """
        super(EHANnet, self).__init__()
        # RG模块的数量
        self.block_num = rg_numbers
        # LAM
        self.lam = LayerAttention(alpha)
        # 1*1的卷积将lam层扩充后的通道改为原来来的通道数，便于残差连接
        self.lam_conv = nn.Conv2d(out_channels*self.block_num,out_channels,1)

        # shallow feature extraction
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)

        # rg_modules，每一个RG中包含rc_block_num个RCAB
        self.rg_layer1 = self._make_layer(ResidualChannelAttentionBlock, out_channels, rc_block_num)
        self.rg_conv1 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        # rg_modules
        self.rg_layer2 = self._make_layer(ResidualChannelAttentionBlock, out_channels, rc_block_num)
        self.rg_conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        # rg_modules
        self.rg_layer3 = self._make_layer(ResidualChannelAttentionBlock, out_channels, rc_block_num)
        self.rg_conv3 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        # csam
        self.csam = ChannelSpatialAttention(out_channels,beta)
        # ---------------------------------
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        # Upscale
        #self.trans_conv = nn.ConvTranspose2d()
        self.conv3 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)

        #


    def _make_layer(self, block, in_channel, rc_numbers):
        layers = []
        for i in range(rc_numbers):
            layers.append(block(in_channel, in_channel))
        return nn.Sequential(*layers)

    def forward(self, x):
        #(h,w,c) = x.shape
        # residual 1
        x = self.conv1(x)
        #-------------------------
        rg1_out = self.rg_layer1(x)
        rg1_out = self.rg_conv1(rg1_out)
        rg1_out = torch.add(x, rg1_out)
        #-------------------------
        rg2_out = self.rg_layer2(rg1_out)
        rg2_out = self.rg_conv2(rg2_out)
        rg2_out = torch.add(rg2_out, rg1_out)
        #--------------------------
        rg3_out = self.rg_layer2(rg2_out)
        rg3_out = self.rg_conv3(rg3_out)
        rg3_out = torch.add(rg3_out, rg2_out)

        lam_x = torch.stack((rg1_out,rg2_out,rg3_out),dim=1)
        lam_x = self.lam(lam_x)
        # residual 2
        lam_x = self.lam_conv(lam_x)
        #---------------------------------------
        rg3_out = self.conv2(rg3_out)
        #csam
        # residual 3
        rg3_out = self.csam(rg3_out)

        x = torch.add(x,lam_x)
        x = torch.add(x, rg3_out)
        #--------------------------------------
        x = self.conv3(x)

        # 数据后处理
        x = torch.sigmoid(x)

        return x
if __name__ == '__main__':
    batch_size = 4
    in_channels = 8
    height = 480
    width = 512
    input_tensor = torch.randn(batch_size, in_channels, height, width)
    print(input_tensor.shape)
    net = EHANnet(in_channels, out_channels=8,alpha=1,beta=1)
    y = net(input_tensor)
    print(y.shape)


