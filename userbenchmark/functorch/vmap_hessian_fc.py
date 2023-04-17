import torch
import torch.nn as nn
from torch.func import vmap, jacfwd, jacrev
from .util import BenchmarkCase

# batched hessians of fully connected layers is a popular quantity
# in physics-related models.
# This test case is from https://github.com/pytorch/functorch/issues/989
# We haven't been able to get the full model yet, so, this test case
# is going into the functorch userbenchmark instead of torchbenchmark.

from torch._dynamo import allow_in_graph
from functools import wraps

def traceable(f):
    f = allow_in_graph(f)

    @wraps(f)
    def wrapper(*args, **kwargs):
        return f(*args, **kwargs)

    return wrapper

class VmapHessianFC(BenchmarkCase):
    def __init__(self):
        device = 'cpu'
        D1 = 2  # x, y
        D2 = 3  # u, v, p
        B = 10
        x = torch.randn(B, D1).to(device)

        model = nn.Sequential(
            nn.Linear(D1, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, D2),
        ).to(device)

        self.model = model
        self.x = x

    def name(self):
        return 'vmap_hessian_fc_cuda'

    def run(self):
        params_and_buffers = (dict(self.model.named_parameters()), dict(self.model.named_buffers()))

        def predict(params_and_buffers, x):
            # out = self.model(x)
            out = torch.func.functional_call(self.model, params_and_buffers, x)
            return out, out

        fn = vmap(
            jacfwd(jacrev(predict, argnums=1, has_aux=True), argnums=1, has_aux=True),
            in_dims=(None, 0),
        )

        fn = torch.compile(traceable(fn))
        hessian, pred = fn(params_and_buffers, self.x)
