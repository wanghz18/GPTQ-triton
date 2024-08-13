from typing import List, Tuple
import numpy as np
import os

##################################################
#       以下是可以修改的部分，制定量化后数的排序        #
##################################################

weight_shape = (4096, 11008)
weight_name = "ffn_wgate_layer_0" # 向量名
weight_blockwise_shape = (weight_shape[0] // 32, weight_shape[1] // 32)  # 以block为基本单元看weight

# 文件路径设置
qweight_file = f"weight_{weight_name}.dat"
zeros_file = f"zeros_{weight_name}.dat"
scales_file = f"scales_{weight_name}.dat"
save_file = f"gptq_{weight_name}.dat"
load_path = "E:/bf16 data/data_blockwise_batching_bf16_decoding"
save_path = "E:/bf16 data/data_blockwise_batching_bf16_decoding"
qweight_file = os.path.join(load_path, qweight_file)
zeros_file = os.path.join(load_path, zeros_file)
scales_file = os.path.join(load_path, scales_file)
save_name = os.path.join(save_path, save_file)

"""
说明：
因为GPTQ排数，16个量化后的block对应5个量化前的block的大小
因此我们以16个量化后的block为基本单元，制定量化后数的排序
order为一个二维list，表示基本单元间的顺序
order[i]为一个list，表示一个基本单元16个block的内部排序
order[i][j]为一个tuple，表示一个量化后block的位置，如weight[0:32,0:32]的位置为(0, 0)，weight[32:64,0:32]的位置为(1, 0)
当需要插入一个全0矩阵时，可以表示如下：
order[i] = None
order[i][j] = (None, None)
"""

"""
例子：
按blockwise为基本单元的row-major排列：
"""
order = []
# 外层循环row
for x in range(weight_blockwise_shape[0]):
    # 内层循环column
    for y in range(weight_blockwise_shape[1]):
        order.append((x, y))
order = [order[i:i + 16] for i in range(0, len(order), 16)]

"""
例子：
按blockwise为基本单元的column-major排列：
"""
order = []
# 外层循环column
for y in range(weight_blockwise_shape[1]):
    # 内层循环row
    for x in range(weight_blockwise_shape[0]):
        order.append((x, y))
order = [order[i:i + 16] for i in range(0, len(order), 16)]  # reshape为16个block为一个基本单元

"""
例子：
按blockwise为基本单元的column-major排列并且在末尾添加一列block的0：
"""
order = []
# 外层循环column
for y in range(weight_blockwise_shape[1]):
    # 内层循环row
    for x in range(weight_blockwise_shape[0]):
        order.append((x, y))
for x in range(weight_blockwise_shape[0]):
    order.append((None, None))
order = [order[i:i + 16] for i in range(0, len(order), 16)]  # reshape为16个block为一个基本单元

"""
例子：
按那天和闫博讨论说的8*8 mesh排列：（这部分比较难理解，有需要可以找我讨论，不需要看懂这个例子）
"""
order = []
# 外两层相当于循环4*8个PE核
# 外层循环row的四个PE核
for x in range(4):
    # 内层循环column的8个PE核
    for y in range(8):
        x_step = weight_shape[0] // 4
        y_step = weight_shape[1] // 8
        # PE核负责的区域
        PE_x_start = x * x_step  # 可取到
        PE_x_end = (x + 1) * x_step  # 取不到
        PE_y_start = y * y_step  # 可取到
        PE_y_end = (y + 1) * y_step  # 取不到
        PE_x_mid = (PE_x_start + PE_x_end) // 2  # 中间位置
        # PE核负责的区域（block_id）
        PE_block_x_start = PE_x_start // 32
        PE_block_x_end = PE_x_end // 32
        PE_block_y_start = PE_y_start // 32
        PE_block_y_end = PE_y_end // 32
        PE_block_x_mid = PE_x_mid // 32
        # mid上半部分的8个数和下半部分的8个数拼一个单元
        for x_inner in range(0, PE_block_x_mid - PE_block_x_start, 8):
            for y_inner in range(0, PE_block_y_end - PE_block_y_start):
                # 16个block
                # 上半个：[PE_block_x_start+x_inner:PE_block_x_start+x_inner+8,
                # PE_block_y_start+y_inner:PE_block_y_start+y_inner+1]
                # 下半个：[PE_block_x_mid+x_inner:PE_block_mid+x_inner+8,
                # PE_block_y_start+y_inner:PE_block_y_start+y_inner+1]
                basic_unit_block16 = []
                for i in range(8):
                    basic_unit_block16.append((PE_block_x_start + x_inner + i, PE_block_y_start + y_inner))
                for i in range(8):
                    basic_unit_block16.append((PE_block_x_mid + x_inner + i, PE_block_y_start + y_inner))
                order.append(basic_unit_block16)

##################################################
#              以下是不要轻易修改的部分              #
##################################################

# gptq配置
groupsize = 128
bitwidth = 4
numbers_per_uint32 = 32 // bitwidth

# weight的衍生形状
qweight_shape = (weight_shape[0], weight_shape[1])  # qweight的形状
zeros_shape = (weight_shape[0] // groupsize, weight_shape[1])  # zeros的形状
scales_shape = (weight_shape[0] // groupsize, weight_shape[1])

# 读取文件并reshape
qweight = np.fromfile(qweight_file, dtype=np.uint32)
qweight = qweight.reshape(qweight_shape)
zeros = np.fromfile(zeros_file, dtype=np.uint32)
zeros = zeros.reshape(zeros_shape)
scales = np.fromfile(scales_file, dtype=np.float16)
scales = scales.reshape(scales_shape)


# zeropoint从int4变fp32
def convert_block_zeros(zeros: "int4", scales: "fp16") -> "fp16":
    "gptq: w = s(w - z) = sw + (-sz)"
    # 期望32个zeros和32个scales相乘
    zeros = zeros.flatten()
    scales = scales.flatten()
    assert zeros.size == 32
    assert scales.size == 32
    zeros = zeros.astype(np.float16)
    return -scales * zeros


def compress_uint32_to_uint4(data: "uint32") -> "uint4":
    data = data.astype(np.uint8)
    data = data.reshape(-1, 2)
    data = data[:, 0] << 4 + data[:, 1]
    return data


def focus_block(block_id_x, block_id_y):
    block_qweight = qweight[block_id_x * 32: (block_id_x + 1) * 32, block_id_y * 32: (block_id_y + 1) * 32]  # 32*32个数
    block_qweight = compress_uint32_to_uint4(block_qweight)
    zeros_id_x = block_id_x * 32 // groupsize
    block_zeros = zeros[zeros_id_x, block_id_y * 32: (block_id_y + 1) * 32]  # 32个数
    block_scales = scales[zeros_id_x, block_id_y * 32: (block_id_y + 1) * 32]  # 32个数
    block_zeros = convert_block_zeros(block_zeros, block_scales)
    return block_scales, block_zeros, block_qweight


def empty_block() -> np.array:
    empty_qweight = np.zeros(32 * 32 // 2, dtype=np.uint8)
    empty_zeros = np.zeros(32 * 32 // 2, dtype=np.float16)
    empty_scales = np.zeros(32 * 32 // 2, dtype=np.float16)
    return empty_scales, empty_zeros, empty_qweight


def block16_unquant_to_block5_quant(block16: List[Tuple[int | None, int | None]]) -> np.array:
    assert len(block16) == 16, "the length of block16 is not 16"
    block5 = []
    for i, j in block16:
        if i is None or j is None:
            block_scales, block_zeros, block_qweight = empty_block()
        else:
            block_scales, block_zeros, block_qweight = focus_block(i, j)
        block5.append(block_scales.flatten().view(np.uint8))
        block5.append(block_zeros.flatten().view(np.uint8))
        block5.append(block_qweight.flatten().view(np.uint8))
    block5_array = np.concatenate(block5)
    assert block5_array.size == 5 * 32 * 32 * 2, "the shape of block5 is not 5 * 32 * 32 * 2 Bytes"
    return block5_array


def build_unquant_to_quant(order: List[List[Tuple[int, int]]]) -> np.array:
    result_array = []
    for block16 in order:
        if block16 is None:
            block5_array = block16_unquant_to_block5_quant([(None, None) for _ in range(16)])
        else:
            block5_array = block16_unquant_to_block5_quant(block16)
        result_array.append(block5_array)
    result_array = np.concatenate(result_array)
    return result_array


result = build_unquant_to_quant(order)
result.tofile(save_name)
