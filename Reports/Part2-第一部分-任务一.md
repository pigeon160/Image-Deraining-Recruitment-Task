$\quad$ 1. ```  parser.add_argument("--model", type=str, default="baseline", choices=["baseline", "unet", "derainnet", "mymodel"], help="Model to use")```

$\quad$ 2.创建 ```model/DerainNet.py``` 和 ```model/Mymodel.py``` ，内容先用 ```model/baseline.py``` 套皮占位，并在 ```main.py``` 中加入 

```
from models.DerainNet import DerainNet
from models.Mymodel import Mymodel
```

$\quad$ 3.

```
 if args.model == "baseline":
 	model = BaselineNet().to(device)
 elif args.model == "unet":
 	model = UNet().to(device)
 elif args.model == "derainnet" :
 	model = DerainNet().to(device)
 elif args.model == "mymodel" :
 	model = Mymodel().to(device)
```

另外，因为本部分规定**损失函数只能使用 L2 Loss** ，因此把 ```pixel_loss_fn``` 改为 ```MSEloss```

```
# pixel_loss_fn = nn.L1Loss()
pixel_loss_fn = nn.MSELoss()
```