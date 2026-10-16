import torch as th
import torch.nn as nn

class ResBlock(nn.Module) :
    def __init__(self, in_channels, out_channels) :
        super(ResBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias = False), 
            nn.InstanceNorm2d(out_channels), 
            nn.ReLU(inplace = True), 
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias = False), 
            nn.InstanceNorm2d(out_channels), 
        )
        self.relu = nn.ReLU(inplace = False)

    def forward(self, x) :
        return self.relu(x + self.conv(x))

class DerainNet(nn.Module):
    def __init__(self, channels : int = 32, num : int = 4):
        super().__init__()
        self.entry = nn.Conv2d(3, channels, 3, 1, 1)
        self.pool = nn.MaxPool2d(2, 2)
        self.downs1 = nn.ModuleList()
        self.downs2 = nn.ModuleList()
        self.downs3 = nn.ModuleList()
        for i in range(num) :
            self.downs1.append(ResBlock(channels, channels))
        for i in range(num) :
            self.downs2.append(ResBlock(channels, channels))
        for i in range(num) :
            self.downs3.append(ResBlock(channels, channels))
        self.up2 = nn.ConvTranspose2d(channels, channels, 2, 2)
        self.up4 = nn.ConvTranspose2d(channels, channels, 4, 4)
        self.exit = nn.Sequential(
            nn.Conv2d(3 * channels, 3 * channels, 3, 1, 1), 
            nn.InstanceNorm2d(3 * channels), 
            nn.ReLU(inplace = True), 
            nn.Conv2d(3 * channels, 3, 3, 1, 1)
        )

    def forward(self, x):
        f = self.entry(x)
        f1 = f
        for down in self.downs1:
            f1 = down(f1)
        f1 = f1 + f

        d2 = self.pool(f)
        f2 = d2
        for down in self.downs2 : 
            f2 = down(f2)
        f2 = self.up2(f2 + d2)

        d3 = self.pool(d2)
        f3 = d3
        for down in self.downs3 : 
            f3 = down(f3)
        f3 = self.up4(f3 + d3)

        fa = self.exit(th.cat([f1, f2, f3], dim = 1))
        
        return x - fa
