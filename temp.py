import numpy as np
from IPython import embed


prefix = 'postprocess/model.layers.0.mlp_'
s = np.load(f'{prefix}upscales.npy')
z = np.load(f'{prefix}upzeros.npy')
w = np.load(f'{prefix}upweight.npy')

y = np.fromfile('dequantize/0_mlp.up_proj.dat', dtype=np.uint32)
y = y.reshape((4096, 11008))
y.dtype = np.float32

Y = (w - z - 1) * s
diff = np.abs(Y - y)
print(diff)
embed()