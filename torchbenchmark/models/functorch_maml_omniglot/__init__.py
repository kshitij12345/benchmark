import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from functorch import make_functional_with_buffers
from torch.func import vmap, grad
import functools
from pathlib import Path
from typing import Tuple

from ...util.model import BenchmarkModel
from torchbenchmark.tasks import OTHER
import copy

from torch._dynamo import allow_in_graph
from functools import wraps

# def make_functional_with_buffers(mod, disable_autograd_tracking=False):
#     params_dict = dict(mod.named_parameters())
#     params_names = params_dict.keys()
#     params_values = tuple(params_dict.values())

#     buffers_dict = dict(mod.named_buffers())
#     buffers_names = buffers_dict.keys()
#     buffers_values = tuple(buffers_dict.values())
    
#     stateless_mod = copy.deepcopy(mod)
#     stateless_mod.to('meta')

#     def fmodel(new_params_values, new_buffers_values, *args, **kwargs):
#         new_params_dict = {name: value for name, value in zip(params_names, new_params_values)}
#         new_buffers_dict = {name: value for name, value in zip(buffers_names, new_buffers_values)}
#         return torch.func.functional_call(stateless_mod, (new_params_dict, new_buffers_dict), args, kwargs)
  
#     if disable_autograd_tracking:
#         params_values = torch.utils._pytree.tree_map(torch.Tensor.detach, params_values)
#     return fmodel, params_values, buffers_values

def traceable(f):
    f = allow_in_graph(f)

    @wraps(f)
    def wrapper(*args, **kwargs):
        return f(*args, **kwargs)

    return wrapper

def loss_for_task(net, n_inner_iter, x_spt, y_spt, x_qry, y_qry):
    params, buffers, fnet = net
    querysz = x_qry.size(0)

    def compute_loss(new_params, buffers, x, y):
        logits = fnet(new_params, buffers, x)
        loss = F.cross_entropy(logits, y)
        return loss

    new_params = params
    for _ in range(n_inner_iter):
        grads = grad(compute_loss)(new_params, buffers, x_spt, y_spt)
        new_params = [p - g * 1e-1 for p, g, in zip(new_params, grads)]

    # The final set of adapted parameters will induce some
    # final loss and accuracy on the query dataset.
    # These will be used to update the model's meta-parameters.
    qry_logits = fnet(new_params, buffers, x_qry)
    qry_loss = F.cross_entropy(qry_logits, y_qry)
    qry_acc = (qry_logits.argmax(
        dim=1) == y_qry).sum() / querysz

    return qry_loss, qry_acc


class Model(BenchmarkModel):
    task = OTHER.OTHER_TASKS
    DEFAULT_TRAIN_BSIZE = 1
    DEFAULT_EVAL_BSIZE = 1
    ALLOW_CUSTOMIZE_BSIZE = False

    def __init__(self, test, device, jit=False, batch_size=None, extra_args=[]):
        super().__init__(test=test, device=device, jit=jit, batch_size=batch_size, extra_args=extra_args)

        n_way = 5
        inplace_relu = True
        net = nn.Sequential(
            nn.Conv2d(1, 64, 3),
            nn.BatchNorm2d(64, affine=True, track_running_stats=False),
            nn.ReLU(inplace=inplace_relu),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 64, 3),
            nn.BatchNorm2d(64, affine=True, track_running_stats=False),
            nn.ReLU(inplace=inplace_relu),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 64, 3),
            nn.BatchNorm2d(64, affine=True, track_running_stats=False),
            nn.ReLU(inplace=inplace_relu),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(64, n_way)).to(device)

        self.model = net

        root = str(Path(__file__).parent.parent)
        self.meta_inputs = torch.load(f'{root}/functorch_maml_omniglot/batch.pt')
        self.meta_inputs = tuple([torch.from_numpy(i).to(self.device) for i in self.meta_inputs])
        self.example_inputs = (self.meta_inputs[0][0],)

    def get_module(self):
        return self.model, self.example_inputs

    def train(self):
        model = self.model
        model.train()
        fnet, params, buffers = make_functional_with_buffers(self.model)
        net = (params, buffers, fnet)
        meta_opt = optim.Adam(params, lr=1e-3)

        # Sample a batch of support and query images and labels.
        x_spt, y_spt, x_qry, y_qry = self.meta_inputs
        task_num, setsz, c_, h, w = x_spt.size()

        n_inner_iter = 5
        meta_opt.zero_grad()

        net = self.model
        params = dict(self.model.named_parameters())
        buffers = dict(self.model.named_buffers())
        params_and_buffers = (params, buffers)

        def loss_for_task_compliant(n_inner_iter, params_and_buffers, x_spt, y_spt, x_qry, y_qry):
            params, buffers = params_and_buffers
            querysz = x_qry.size(0)

            def compute_loss(new_params, buffers, x, y):
                logits = torch.func.functional_call(net, (new_params, buffers), x)
                loss = F.cross_entropy(logits, y)
                return loss

            new_params = params
            for _ in range(n_inner_iter):
                grads = grad(compute_loss)(new_params, buffers, x_spt, y_spt)
                new_params = {k: p - g * 1e-1 for (k, p), (kg, g) in zip(new_params.items(), grads.items())}

            # The final set of adapted parameters will induce some
            # final loss and accuracy on the query dataset.
            # These will be used to update the model's meta-parameters.
            qry_logits = torch.func.functional_call(net, (new_params, buffers), x_qry)
            qry_loss = F.cross_entropy(qry_logits, y_qry)
            qry_acc = (qry_logits.argmax(
                dim=1) == y_qry).sum() / querysz

            return qry_loss, qry_acc

        # In parallel, trains one model per task. There is a support (x, y)
        # for each task and a query (x, y) for each task.
        # compute_loss_for_task = functools.partial(loss_for_task, net, n_inner_iter)
        fn = vmap(loss_for_task_compliant, in_dims=(None, None, 0, 0, 0, 0))
        fn = torch.compile(traceable(fn))
        qry_losses, qry_accs = fn(n_inner_iter, params_and_buffers, x_spt, y_spt, x_qry, y_qry)

        # Compute the maml loss by summing together the returned losses.
        qry_losses.sum().backward()

        meta_opt.step()

    def eval(self) -> Tuple[torch.Tensor]:
        model, (example_input,) = self.get_module()
        model.eval()
        with torch.no_grad():
            out = model(example_input)
        return (out, )
