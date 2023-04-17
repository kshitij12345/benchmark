# """test.py
# Setup and Run hub models.

# Make sure to enable an https proxy if necessary, or the setup steps may hang.
# """
# # This file shows how to use the benchmark suite from user end.
# import gc
# import os
# import unittest
# from unittest.mock import patch

# import torch
# from torchbenchmark import _list_model_paths, ModelTask, get_metadata_from_yaml
# from torchbenchmark.util.metadata_utils import skip_by_metadata
# from components._impl.tasks import base as base_task


# class FunctorchModelTask(ModelTask):
#     @base_task.run_in_worker(scoped=True)
#     @staticmethod
#     def check_functorch_grad() -> None:
#         instance = globals()["model"]
#         import functorch
#         import torch.utils._pytree as pytree
#         from torch.testing._internal.common_utils import is_iterable_of_tensors


#         model, inputs = instance.get_module()
#         fn_model, params, buffers = functorch.make_functional_with_buffers(model)
#         if hasattr(instance, "HF_MODEL"):
#             # `inputs` from `get_module` don't have labels,
#             # hence we get loss = `None`. So we pull
#             # the inputs with labels manually.
#             inputs = instance.example_inputs

#         def fn(params, buffers, inputs):
#             # HF models return loss in the result dictionary/structure
#             if hasattr(instance, "HF_MODEL"):
#                 results = fn_model(params, buffers, **inputs)
#                 return results.loss

#             results = fn_model(params, buffers, *inputs)
#             results, _ = pytree.tree_flatten(results)
#             results = pytree.tree_map(lambda x : x.sum() if isinstance(x, torch.Tensor) else x, results)
#             return sum([res.sum() for res in results])

#         grads = functorch.grad(fn, argnums=(0,))(params, buffers, inputs)
#         assert len(grads[0]) == len(params)


#     @base_task.run_in_worker(scoped=True)
#     @staticmethod
#     def check_functorch_jacrev() -> None:
#         instance = globals()["model"]
#         import functorch
#         import torch.utils._pytree as pytree
#         from torch.testing._internal.common_utils import is_iterable_of_tensors


#         model, inputs = instance.get_module()
#         fn_model, params, buffers = functorch.make_functional_with_buffers(model)
#         if hasattr(instance, "HF_MODEL"):
#             # `inputs` from `get_module` don't have labels,
#             # hence we get loss = `None`. So we pull
#             # the inputs with labels manually.
#             inputs = instance.example_inputs

#         def fn(params, buffers, inputs):
#             # HF models return loss in the result dictionary/structure
#             if hasattr(instance, "HF_MODEL"):
#                 results = fn_model(params, buffers, **inputs)
#                 return results.loss

#             results = fn_model(params, buffers, *inputs)
#             results, _ = pytree.tree_flatten(results)
#             results = pytree.tree_map(lambda x : x.sum() if isinstance(x, torch.Tensor) else x, results)
#             return sum([res.sum() for res in results])

#         grads = functorch.jacrev(fn, argnums=(0,))(params, buffers, inputs)
#         # assert len(grads[0]) == len(params)

#     @base_task.run_in_worker(scoped=True)
#     @staticmethod
#     def check_functorch_jvp() -> None:
#         instance = globals()["model"]
#         import functorch
#         import torch.utils._pytree as pytree
#         from torch.testing._internal.common_utils import is_iterable_of_tensors


#         model, inputs = instance.get_module()
#         fn_model, params, buffers = functorch.make_functional_with_buffers(model)
#         if hasattr(instance, "HF_MODEL"):
#             # `inputs` from `get_module` don't have labels,
#             # hence we get loss = `None`. So we pull
#             # the inputs with labels manually.
#             inputs = instance.example_inputs

#         def fn(params, buffers, inputs):
#             # HF models return loss in the result dictionary/structure
#             if hasattr(instance, "HF_MODEL"):
#                 results = fn_model(params, buffers, **inputs)
#                 return results.loss

#             results = fn_model(params, buffers, *inputs)
#             results, _ = pytree.tree_flatten(results)
#             results = pytree.tree_map(lambda x : x.sum() if isinstance(x, torch.Tensor) else x, results)
#             return sum([res.sum() for res in results])

#         functorch.jvp(fn, (params, buffers, inputs), (params, buffers, inputs))

#     @base_task.run_in_worker(scoped=True)
#     @staticmethod
#     def check_functorch_vmap() -> None:
#         instance = globals()["model"]
#         import functorch
#         import torch.utils._pytree as pytree
#         from torch.testing._internal.common_utils import is_iterable_of_tensors


#         model, inputs = instance.get_module()
#         fn_model, params, buffers = functorch.make_functional_with_buffers(model)
#         if hasattr(instance, "HF_MODEL"):
#             # `inputs` from `get_module` don't have labels,
#             # hence we get loss = `None`. So we pull
#             # the inputs with labels manually.
#             inputs = instance.example_inputs

#         def fn(params, buffers, inputs):
#             # HF models return loss in the result dictionary/structure
#             if hasattr(instance, "HF_MODEL"):
#                 results = fn_model(params, buffers, **inputs)
#                 return results.loss

#             results = fn_model(params, buffers, *inputs)
#             results, _ = pytree.tree_flatten(results)
#             results = pytree.tree_map(lambda x : x.sum() if isinstance(x, torch.Tensor) else x, results)
#             return sum([res.sum() for res in results])

#         def expand_input(t):
#             if isinstance(t, torch.Tensor):
#                 return t.expand(2, *t.shape)
#             return t
#         inputs = pytree.tree_map(expand_input, inputs)

#         functorch.vmap(fn, in_dims=(None, None, 0), randomness='same')(params, buffers, inputs)
        

# # Some of the models have very heavyweight setup, so we have to set a very
# # generous limit. That said, we don't want the entire test suite to hang if
# # a single test encounters an extreme failure, so we give up after 5 a test
# # is unresponsive to 5 minutes. (Note: this does not require that the entire
# # test case completes in 5 minutes. It requires that if the worker is
# # unresponsive for 5 minutes the parent will presume it dead / incapacitated.)
# TIMEOUT = 300  # Seconds

# class TestBenchmark(unittest.TestCase):

#     def setUp(self):
#         gc.collect()

#     def tearDown(self):
#         gc.collect()

#     def test_fx_profile(self):
#         try:
#             from torch.fx.interpreter import Interpreter
#         except ImportError:  # older versions of PyTorch
#             raise unittest.SkipTest("Requires torch>=1.8")
#         from fx_profile import main, ProfileAggregate
#         with patch.object(ProfileAggregate, "save") as mock_save:
#             # just run one model to make sure things aren't completely broken
#             main(["--repeat=1", "--filter=pytorch_struct", "--device=cpu"])
#             self.assertGreaterEqual(mock_save.call_count, 1)

# def _load_test(path, device):
#     def check_functorch(self):
#         task = FunctorchModelTask(path, timeout=TIMEOUT)
#         with task.watch_cuda_memory(skip=(device != "cuda"), assert_equal=self.assertEqual):
#             try:
#                 task.make_model_instance(test="train", device=device, jit=False, batch_size=1)
#                 # task.check_functorch_grad()
#                 # task.check_functorch_jacrev()
#                 # task.check_functorch_jvp()
#                 task.check_functorch_vmap()
#                 task.del_model_instance()
#             except NotImplementedError:
#                 self.skipTest(f'Method check_device on {device} is not implemented, skipping...')


#     name = os.path.basename(path)
#     metadata = get_metadata_from_yaml(path)
#     for fn, fn_name in zip([check_functorch],
#                            ["check_functorch"]):
#         # set exclude list based on metadata
#         setattr(TestBenchmark, f'test_{name}_{fn_name}_{device}',
#                 (unittest.skipIf(skip_by_metadata(test=fn_name, device=device,\
#                                                   jit=False, extra_args=[], metadata=metadata), "This test is skipped by its metadata")(fn)))


# def _load_tests():
#     devices = ['cpu']
#     if torch.cuda.is_available():
#         devices.append('cuda')
#     if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
#         devices.append('mps')

#     for path in _list_model_paths():
#         # TODO: skipping quantized tests for now due to BC-breaking changes for prepare
#         # api, enable after PyTorch 1.13 release
#         if "quantized" in path:
#             continue
#         for device in devices:
#             _load_test(path, device)


# _load_tests()
# if __name__ == '__main__':
#     unittest.main()

# CIFAR10
# Works
# from torchbenchmark.models.functorch_dp_cifar10 import Model

# m = Model("train", 'cpu')

# m.train()
# print("Successful")

# MAML
from torchbenchmark.models.functorch_maml_omniglot import Model
# import torch
m = Model("train", 'cpu')

m.train()
print("Successful")

import torch

class Net(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 64, 3)
        self.linear = torch.nn.Linear(28, 28)
    
    def forward(self, x):
        # return self.linear(x)
        return self.conv(x)

net = Net()

x = torch.randn(64, 1, 28, 28)

from torch.func import functional_call
from torch._dynamo import allow_in_graph
from functools import wraps

def traceable(f):
    f = allow_in_graph(f)

    @wraps(f)
    def wrapper(*args, **kwargs):
        return f(*args, **kwargs)

    return wrapper

def f(x, params, buffers):
    def pred(x):
        return functional_call(net, (params, buffers), x)

    return torch.func.vmap(pred)(x)

# f(x, dict(net.named_parameters()), dict(net.named_buffers()))

# torch.compile(traceable(f))(x, dict(net.named_parameters()), dict(net.named_buffers()))

# Errors
# from userbenchmark.functorch.vmap_hessian_fc import VmapHessianFC
# # import torch
# m = VmapHessianFC()
# m.run()

# import torch
# from torch import nn
# from torch._dynamo import allow_in_graph
# from functools import wraps
# from torch.func import jacfwd, jacrev, vmap, vjp, jvp, grad

# def traceable(f):
#     f = allow_in_graph(f)

#     @wraps(f)
#     def wrapper(*args, **kwargs):
#         return f(*args, **kwargs)

#     return wrapper


# model = nn.Sequential(
#             nn.Linear(3, 512),
#             nn.ReLU(),
#             nn.Linear(512, 3),
#         )

# x = torch.randn(10, 3)

# def predict(params_and_buffers, x):
#     return torch.func.functional_call(model, params_and_buffers, x)

# params_and_buffers = (dict(model.named_parameters()), dict(model.named_buffers()))

# fn = vmap(predict, in_dims=(None, 0))
# expected = fn(params_and_buffers, x)
# actual = torch.compile(traceable(fn))(params_and_buffers, x)

# torch.testing.assert_close(actual, expected)

