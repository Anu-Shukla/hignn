import sys
import torch
from torch.autograd.functional import jacobian

sys.path.append("../python")
from HIGNN.model_structure import Two_body_net

m = torch.load(
    "../python/Saved_Model/Unbounded/HIGNN_nn_2body.pkl",
    map_location="cpu",
    weights_only=False,
)
print(type(m))
print(m.keys() if hasattr(m, "keys") else m)

x = torch.randn(5, 3, requires_grad=True)
print(m(x).grad_fn)

print(jacobian(m, x).shape)
