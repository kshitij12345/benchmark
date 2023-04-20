# # MAML Model
from torchbenchmark.models.functorch_maml_omniglot import Model
from torch._dynamo.utils import counters
import torch

counters.clear()
m = Model("train", 'cpu')

m.train()
print(counters)

# CIFAR-10
from torchbenchmark.models.functorch_dp_cifar10 import Model
from torch._dynamo.utils import counters

counters.clear()
m = Model("train", 'cpu')  # OOM on CUDA

m.train()
print(counters)

# VmapHessian
from userbenchmark.functorch.vmap_hessian_fc import VmapHessianFC
from torch._dynamo.utils import counters
import torch

counters.clear()
m = VmapHessianFC(device='cpu')  # Error on CUDA
m.run()
print(counters)
