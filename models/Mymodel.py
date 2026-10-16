import torch as th
import torch.nn as nn

class Mask(nn.Module) :
    """
    思路：通过卷积+sigmoid激活函数生成注意力图
    作用：生成注意力图，作为掩膜“罩在”训练出来的残差上
    参考论文：SPANet
    """
    def __init__(self, in_channels : int):
        super().__init__()
        self.mask = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 4, kernel_size = 3, padding = 1), 
            nn.ReLU(inplace = True),
            # 把所有通道压缩成一个，作为掩膜表示有雨的概率
            nn.Conv2d(in_channels // 4, 1, kernel_size = 1), 
            nn.Sigmoid()
        )
    def forward(self, x) :
        return self.mask(x)

class SADBlock(nn.Module) :
    """
    思路：用空洞卷积扩大感受野，并采用生成的掩膜乘上得到的残差雨层解决内容被抹去的问题
    作用：残差卷积块，得到雨层
    参考论文：SPANet
    """
    def __init__(self, channels : int) :
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size = 3, padding = 1), 
            nn.InstanceNorm2d(channels), 
            nn.ReLU(inplace = True), 
            nn.Conv2d(channels, channels, kernel_size = 3, padding = 2, dilation = 2), 
            nn.InstanceNorm2d(channels), 
        )
        self.mask = Mask(channels)
        self.relu = nn.ReLU(inplace = True)

    def forward(self, x) :
        rain = self.conv(x)
        mask = self.mask(x)
        return self.relu(x + rain * mask)

class Mymodel(nn.Module) :
    """
    思路：多阶段去雨，每个阶段把上一阶段的结果和初始有雨图cat在一起做残差网络，多次迭代得到去雨图
    作用：去雨
    参考论文：PReNet
    """
    def __init__(self, channels : int = 48, num_blocks : int = 4, iterations : int = 3) :
        super().__init__()
        self.channels = channels
        self.iterations = iterations

        self.entry = nn.Conv2d(6, channels, kernel_size = 3, padding = 1)
        self.blocks = nn.ModuleList()
        for _ in range(num_blocks) :
            self.blocks.append(SADBlock(channels))
        self.exit = nn.Conv2d(channels, 3, kernel_size = 3, padding = 1)

    def forward(self, x_input) :
        x = x_input
        for _ in range(self.iterations) :
            x_in = th.cat([x, x_input], dim = 1)
            feat = self.entry(x_in)
            for bolck in self.blocks :
                feat = bolck(feat)
            feat = self.exit(feat)
            x = x + feat
        return x
