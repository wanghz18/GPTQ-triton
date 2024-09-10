import numpy as np
from IPython import embed
import os


layers = ['0', '31']
tp = ['self_attn', 'mlp']
name = {
    'self_attn' : ['q', 'k', 'v', 'o'],
    'mlp': ['up', 'gate', 'down'],
}

def compress_uint32_to_uint4(data: "uint32") -> "uint4":
    data = data.astype(np.uint8)
    data = data.reshape(-1, 2)
    data = (data[:, 0] << 4) + data[:, 1]
    return data

for x in layers:
    for y in tp:
        names = name[y]
        for it in names:
            prefix = f'model.layers.{x}.{y}_{it}'
            w = np.load(f'postprocess/{prefix}weight.npy')
            z = np.load(f'postprocess/{prefix}zeros.npy')
            s = np.load(f'postprocess/{prefix}scales.npy')
            z = np.array(z, dtype=np.float16)
            w = w.astype(np.uint32)
            if not os.path.exists(f'block_wise/{prefix}'):
                os.mkdir(f'block_wise/{prefix}')
            for row in range(0, w.shape[0], 32):
                for column in range(0, w.shape[1], 32):
                    block = []
                    block_scales = s[row, column : column + 32]
                    block_zeros = -(z[row, column : column + 32] + 1) * s[row, column : column + 32]
                    block_qweight = w[row : row + 32, column : column + 32]
                    block_qweight = compress_uint32_to_uint4(block_qweight)

                    block.append (block_scales.flatten().view(np.uint8))
                    block.append(block_zeros.flatten().view(np.uint8))
                    block.append(block_qweight.flatten().view(np.uint8))
                    result_array = np.concatenate(block)
                    filename = f'block_wise/{prefix}/{row}_{column}.dat'
                    result_array.tofile(filename)
