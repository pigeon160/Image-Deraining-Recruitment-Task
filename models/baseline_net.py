import torch.nn as nn

class Block(nn.Module) :
    def __init__(self, in_channels, out_channels) :
        super(Block, self).__init__()
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

class BaselineNet(nn.Module):
    def __init__(self, in_channels = 3, out_channels = 3, channels = 64, num = 4):
        super(BaselineNet, self).__init__()
        self.blocks = nn.ModuleList()
        self.entry = nn.Conv2d(in_channels, channels, 3, 1, 1)
        for i in range(num) :
            self.blocks.append(Block(channels, channels))
        self.exit = nn.Conv2d(channels, out_channels, 3, 1, 1)

    def forward(self, x):
        s = x
        x = self.entry(x)
        for block in self.blocks:
            x = block(x)
        x = self.exit(x)
        return s - x
