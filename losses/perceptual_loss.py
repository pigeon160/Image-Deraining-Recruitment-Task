import torch as th
import torch.nn as nn
from torchvision.models import vgg19, VGG19_Weights

class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        vgg = vgg19(weights = VGG19_Weights.IMAGENET1K_V1).features
        self.slice = vgg[:35]
        self.slice.eval()
        for p in self.slice.parameters():
            p.requires_grad = False
        self.mean = th.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        self.std = th.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        self.criterion = nn.L1Loss()

    def forward(self, generated, ground_truth):
        gen_norm = (generated - self.mean) / self.std
        gt_norm = (ground_truth - self.mean) / self.std

        feat_gen = self.slice(gen_norm)
        feat_gt = self.slice(gt_norm)

        return self.criterion(feat_gen, feat_gt)