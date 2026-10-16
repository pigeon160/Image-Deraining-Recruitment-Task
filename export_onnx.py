import os
import torch as th
from dataset import DerainDataset
from models.Mymodel import Mymodel
from torch.utils.data import DataLoader

def transform() :

  device = th.device("cpu")
  model = Mymodel().to(device)
  model.eval()
  n = sum(p.numel() for p in model.parameters())
  print(f"参数量：{n} | 模型大小：{(n * 4 / 1024 / 1024):.2f} MB")
  data_dir = "./data/Rain200L"
  ds = DerainDataset(data_dir, is_train = False)
  rainy, _, _ = next(iter(DataLoader(ds, batch_size = 1, shuffle = False)))

  path = "checkpoints/mymodel.onnx"
  os.makedirs("checkpoints", exist_ok = True)

  th.onnx.export(
    model, rainy, path, 
    export_params = True, 
    opset_version = 14, 
    do_constant_folding = True, 
    input_names = ["rainy_image"], 
    output_names = ["derained_image"], 
    dynamic_axes = {
      "rainy_image" : {0 : "batch", 2 : "H", 3 : "W"}, 
      "derained_image" : {0 : "batch", 2 : "H", 3 : "W"}
    }, 
  )

if __name__ == "__main__" :
  transform()