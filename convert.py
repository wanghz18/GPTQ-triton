import os
import numpy as np


def convert_npy_to_dat(directory):
    # 遍历指定目录
    for filename in os.listdir(directory):
        # 检查文件是否为.npy格式
        if filename.endswith('.npy'):
            # 检查文件名是否包含特定的关键字
            if ('weight' not in filename and
                'scale' not in filename and
                'zero' not in filename) or ('norm' in filename):
                # 构造完整的文件路径
                npy_file_path = os.path.join(directory, filename)
                # 读取.npy文件
                data = np.load(npy_file_path)
                # 构造输出的.dat文件名
                dat_file_name = filename.replace('.npy', '.dat')
                dat_file_path = os.path.join(directory, dat_file_name)
                # 将数据写入.dat文件
                # data.tofile(dat_file_path)
                # print(f'Converted: {npy_file_path} to {dat_file_path}')

                # 获取元数据信息
                shape = data.shape
                dtype = data.dtype

                # 构造同名的.c文件名
                c_file_name = filename.replace('.npy', '.c')
                c_file_path = os.path.join(directory, c_file_name)

                # 写入元数据到.c文件
                with open(c_file_path, 'w') as c_file:
                    c_file.write(f"/*\n")
                    c_file.write(f" * File: {c_file_name}\n")
                    c_file.write(f" * Converted from: {filename}\n")
                    c_file.write(f" * Dimensions: {shape}\n")
                    c_file.write(f" * Data type: {dtype}\n")
                    c_file.write(f" */\n")

                print(f'Metadata written to: {c_file_path}')

                # 如果数据维度 >= 2 且后两维都整除 32
                if len(shape) >= 2 and shape[-1] % 32 == 0 and shape[-2] % 32 == 0:
                    # 重新排列为 32x32 的块格式
                    blockwise_shape = shape[:-2] + (shape[-2] // 32, 32, shape[-1] // 32, 32)
                    blockwise_data = data.reshape(blockwise_shape).swapaxes(-2, -3)
                    blockwise_data = np.ascontiguousarray(blockwise_data)

                    # 构造块文件名
                    blockwise_file_name = filename.replace('.npy', '_blockwise.dat')
                    blockwise_file_path = os.path.join(directory, blockwise_file_name)

                    # 将块数据写入新文件
                    blockwise_data.tofile(blockwise_file_path)
                    print(f'Blockwise converted: {npy_file_path} to {blockwise_file_path}')


# 指定你的目录
directory = 'postprocess'
convert_npy_to_dat(directory)
