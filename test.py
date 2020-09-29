import torch
import numpy as np
a = torch.tensor(1.9, dtype=torch.float,requires_grad=False)
a = a.cuda()
print(a)
