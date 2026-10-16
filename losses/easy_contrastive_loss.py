import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg19, VGG19_Weights

class EasyContrastiveLoss(nn.Module) :
  def __init__(self) :
    super(EasyContrastiveLoss, self).__init__()
    vgg = vgg19(weights = VGG19_Weights.IMAGENET1K_V1).features
    self.slice = vgg[:35]
    self.slice.eval()
    for p in self.slice.parameters() :
      p.requires_grad = False
    self.pool = nn.AdaptiveAvgPool2d(1)
    self.mean = th.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    self.std = th.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

  def get_vector(self, img) :
    img_norm = (img - self.mean) / self.std
    
    feat = self.slice(img_norm)
    v = self.pool(feat).flatten(1)

    return F.normalize(v, p = 2, dim = 1)

  def forward(self, pred, target, rainy) :
    v_pred = self.get_vector(pred)
    v_target = self.get_vector(target)
    v_rainy = self.get_vector(rainy)

    cos_target = (v_pred * v_target).sum(dim = 1).mean()
    cos_rainy = (v_pred * v_rainy).sum(dim = 1).mean()

    loss = cos_rainy - cos_target
    return loss