import numpy as np
from IPython import embed


y_path = 'postprocess/model.layers.0.self_attn_k_before_rope.npy'
y = np.load(y_path)

x_path = 'postprocess/model.layers.0.self_attn_input.npy'
x = np.load(x_path)

w_path = 'postprocess/model.layers.0.self_attn_kweight.npy'
w = np.load(w_path)

z_path = 'postprocess/model.layers.0.self_attn_kzeros.npy'
z = np.load(z_path)

s_path = 'postprocess/model.layers.0.self_attn_kscales.npy'
s = np.load(s_path)

W = (w - z - 1) * s
Y = x @ W
error = np.abs(y - Y)
print(error.max(), error.mean())
# 0.0027529796575436194
