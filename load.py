from safetensors import safe_open
from IPython import embed


tensors = {}
with safe_open("quantized_model/model.safetensors", framework="pt", device=0) as f:
    tensor = f.get_tensor('model.layers.0.mlp.up_proj.qweight')
    embed()