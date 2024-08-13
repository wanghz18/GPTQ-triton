import os
import numpy as np


prefix = 'save'
files = os.listdir(prefix)
# print(files)
for it in files:
    filename = f'{prefix}/{it}'
    data = np.load(filename)
    if 'embed_tokens' in it or 'norm' in it or 'rope' in it or 'lm_head' in it:
        np.save(f'postprocess/{it}', data)
        print(it, data.shape)
    elif 'weight' in it:
        new_data = []
        row = data.shape[0]
        for i in range(row):
            item = data[i, :]
            for j in range(8):
                new_data.append((item >> (4 * j)) & 0xf)
        new_data = np.array(new_data)
        np.save(f'postprocess/{it}', new_data)
        print(it, new_data.shape)
    elif 'zero' in it:
        new_data = np.zeros((data.shape[0], data.shape[1] * 8))
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                num = data[i, j]
                for k in range(8):
                    quant_num = (num >> (4 * k)) & 0xf
                    new_data[i, 8 * j + k] = quant_num
        
        res = []
        row = new_data.shape[0]
        for i in range(row):
            for _ in range(128):
                res.append(new_data[i, :])
        new_data = np.array(res)
        np.save(f'postprocess/{it}', new_data)
        print(it, new_data.shape)
    elif 'scale' in it:
        row = data.shape[0]
        new_data = []
        for i in range(row):
            for _ in range(128):
                new_data.append(data[i, :])
        new_data = np.array(new_data)
        np.save(f'postprocess/{it}', new_data)
        print(it, new_data.shape)
    else:
        np.save(f'postprocess/{it}', data)
        print(it, data.shape)