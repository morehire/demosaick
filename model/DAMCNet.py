import torch.nn as nn
import torch


class SourceEncoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SourceEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=True),
            nn.LeakyReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=True),
            nn.LeakyReLU(),
        )

    def forward(self, x):
        return self.encoder(x)


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(DoubleConv, self).__init__()
        padding = kernel_size // 2
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
            nn.LeakyReLU(0.2),
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=padding),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(0.2)
        )

    def forward(self, x):
        return self.conv(x)


class ChannelAttention(nn.Module):
    def __init__(self, in_channels):
        super(ChannelAttention, self).__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // 2, bias=True),
            nn.ReLU(),
            nn.Linear(in_channels // 2, in_channels, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        avg_out = self.global_avg_pool(x)
        avg_out = avg_out.view(avg_out.size(0), -1)
        fc_out = self.fc(avg_out)
        fc_out = fc_out.view(fc_out.size(0), -1, 1, 1)
        fc_out = fc_out.expand_as(x)
        return x*fc_out


class MCCA(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_sizes):
        super(MCCA, self).__init__()
        self.double_convs = nn.ModuleList([
            DoubleConv(in_channels, out_channels, kernel_size)
            for kernel_size in range(kernel_sizes, 1, -2)
        ])
        self.channel_attention = ChannelAttention(out_channels)

    def forward(self, x):
        outputs = [double_conv(x) for double_conv in self.double_convs]
        x = torch.sum(torch.stack(outputs), dim=0)
        x = self.channel_attention(x)
        return x


class DAMC(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DAMC, self).__init__()
        self.source_encoder = SourceEncoder(in_channels, out_channels)
        self.up_mcca3 = MCCA(out_channels, out_channels, 3)
        self.up_mcca5 = MCCA(out_channels, out_channels, 5)
        self.up_mcca7 = MCCA(out_channels, out_channels, 7)
        self.up_mcca9 = MCCA(out_channels, out_channels, 9)
        self.and_mcca9s = nn.Sequential(
            MCCA(out_channels, out_channels, 9),
            MCCA(out_channels, out_channels, 9)
        )
        self.down_mcca9 = MCCA(out_channels, out_channels, 9)
        self.down_mcca7 = MCCA(out_channels, out_channels, 7)
        self.down_mcca5 = MCCA(out_channels, out_channels, 5)
        self.down_mcca3 = MCCA(out_channels, out_channels, 3)

        self.softplus = nn.Softplus(threshold=40)

    def forward(self, x):
        x = self.source_encoder(x)
        up_x3 = self.up_mcca3(x)
        up_x5 = self.up_mcca5(up_x3)
        input_up_x7 = x+up_x3+up_x5
        up_x7 = self.up_mcca7(input_up_x7)
        up_x9 = self.up_mcca9(up_x7)
        and_x9 = self.and_mcca9s(up_x9)
        down_x9 = self.down_mcca9(and_x9)
        down_x7 = self.down_mcca7(down_x9)
        input_down_x5 = up_x3+down_x7
        down_x5 = self.down_mcca5(input_down_x5)
        input_down_x3 = up_x3+down_x5
        down_x3 = self.down_mcca3(input_down_x3)

        x = self.softplus(down_x3)
        x = x/torch.max(x)
        return x


if __name__ == "__main__":
    batch_size = 16
    in_channels = 8
    height = 128
    width = 128
    input_tensor = torch.abs(torch.randn(batch_size, in_channels, height, width))

    # # 创建不同类型的 MCCA 模块
    # mcca9 = MCCA(in_channels, out_channels=64, kernel_sizes=9)
    # mcca7 = MCCA(in_channels, out_channels=64, kernel_sizes=7)
    # mcca5 = MCCA(in_channels, out_channels=64, kernel_sizes=5)
    # mcca3 = MCCA(in_channels, out_channels=64, kernel_sizes=3)
    #
    # # 在 MCCA 模块上执行前向传播
    # output9 = mcca9(input_tensor)
    # output7 = mcca7(input_tensor)
    # output5 = mcca5(input_tensor)
    # output3 = mcca3(input_tensor)
    #
    # # 输出结果形状
    # print("Input shape:", input_tensor.shape)
    # print("MCCA9 output shape:", output9.shape)
    # print("MCCA7 output shape:", output7.shape)
    # print("MCCA5 output shape:", output5.shape)
    # print("MCCA3 output shape:", output3.shape)

    damc = DAMC(3, 8)
    output = damc(input_tensor)
    print("min: ", torch.min(output))
    print("max: ", torch.max(output))
    print("DAMC output shape:", output.shape)