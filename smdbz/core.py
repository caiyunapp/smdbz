import struct
import copy

import numpy as np
from matplotlib import image
from scipy import sparse
from tqdm import tqdm


class CSRMatrix:
    """
    CSRMatrix 是一个用于表示稀疏矩阵的类，使用压缩稀疏行（Compressed Sparse Row，CSR）格式存储数据。
    """

    def __init__(self, data: list, indices: list, indptr: list):
        """
        初始化 CSRMatrix 实例。

        参数:
        data (list): 非零元素的列表。
        indices (list): 非零元素的列索引列表。
        indptr (list): 行索引指针列表，指示每行开始和结束的位置。
        """
        self.data = data
        self.indices = indices
        self.indptr = indptr

    def get(self, row: int, col: int):
        """
        获取指定行和列的元素。

        参数:
        row (int): 行索引。
        col (int): 列索引。

        返回:
        int: 指定位置的元素值。如果该位置没有元素，则返回 0。
        """
        start = self.indptr[row]
        end = self.indptr[row + 1]
        for i in range(start, end):
            if self.indices[i] == col:
                return self.data[i]
        return 0

    def __getitem__(self, key):
        """
        重载索引操作符，使得可以使用 matrix[row, col] 的方式获取元素。

        参数:
        key (tuple): 包含行索引和列索引的元组。

        返回:
        int: 指定位置的元素值。如果该位置没有元素，则返回 0。
        """
        row, col = key
        return self.get(row, col)

    def __setitem__(self, key: tuple, value: int):
        """
        重载赋值操作符，使得可以使用 matrix[row, col] = value 的方式设置元素。

        参数:
        key (tuple): 包含行索引和列索引的元组。
        value (int): 要设置的值。

        异常:
        IndexError: 如果指定的索引超出范围。
        """
        row, col = key
        start = self.indptr[row]
        end = self.indptr[row + 1]
        for i in range(start, end):
            if self.indices[i] == col:
                self.data[i] = value
                return
        raise IndexError("index out of range")


def read_dbz_idx(fp):
    """
    读取 dbz 文件并返回一个 numpy 数组。

    该函数首先使用 image.imread 函数读取指定的 dbz 文件，然后将读取的结果乘以 255 并转换为 uint8 类型的 numpy 数组。

    参数:
    fp (str): 要读取的 dbz 文件的路径。

    返回:
    numpy.ndarray: 从 dbz 文件中读取的数据构造的 numpy 数组。
    """
    array = (image.imread(fp) * 255).astype(np.uint8)
    return array


def sparsize(fps, progress_bar=True):
    """
    对给定的文件路径列表中的 dbz 文件进行稀疏化处理。

    该函数首先读取第一个 dbz 文件，并将其复制为 dbz_array_sum 和 dbz_arrays。
    然后，对文件路径列表中的其余文件，读取每个文件并将其添加到 dbz_array_sum，并将其添加到 dbz_arrays 列表中。
    最后，找出 dbz_array_sum 中非零元素的位置，将 dbz_array_sum 转换为 CSR 矩阵，并返回一个包含结果的字典。

    参数:
    fps (list): dbz 文件的路径列表。
    progress_bar (bool, 可选): 是否显示进度条。默认为 True。

    返回:
    dict: 包含以下键的字典：
        - "indices": CSR 矩阵的索引。
        - "indptr": CSR 矩阵的指针。
        - "nozi": 非零元素的位置。
        - "dbz_arrays": dbz 文件的 numpy 数组列表。
        - "width": dbz_array_sum 的宽度。
        - "height": dbz_array_sum 的高度。
    """
    # 读取第一个 dbz 文件
    fp0 = fps[0]
    dbz_array = read_dbz_idx(fp0)
    # 初始化 dbz_array_sum 和 dbz_arrays
    dbz_array_sum = copy.deepcopy(dbz_array)
    dbz_arrays = [dbz_array]

    # 根据是否需要显示进度条来选择迭代器
    if progress_bar:
        iterator = tqdm(fps[1:], desc="Sparsizing")
    else:
        iterator = fps[1:]

    # 对文件路径列表中的其余文件进行处理
    for fp in iterator:
        array = read_dbz_idx(fp)
        dbz_array_sum += array
        dbz_arrays.append(array)

    # 找出 dbz_array_sum 中非零元素的位置
    nozi = np.where(dbz_array_sum != 0)
    # 将 dbz_array_sum 转换为 CSR 矩阵
    sm = sparse.csr_matrix(dbz_array_sum)

    # 构造结果字典并返回
    result = {
        "indices": sm.indices,
        "indptr": sm.indptr,
        "nozi": nozi,
        "dbz_arrays": dbz_arrays,
        "width": dbz_array_sum.shape[1],
        "height": dbz_array_sum.shape[0],
    }
    return result

def generate_smdbz(dbzfps, outfp, progress_bar=True):
    """
    生成数组字节并写入到指定的输出文件中。

    该函数首先将输入的 dbz 文件进行稀疏化处理，然后将处理结果的各个部分（包括索引、指针、非零元素的索引和 dbz 数组）转换为字节格式。
    然后，对每个非零元素，将其转换为一个字节，并将所有的字节添加到一个字节数组中。
    最后，将所有的字节（包括帧长度、索引长度、指针长度、索引、指针和字节数组）写入到指定的输出文件中。

    参数:
    dbzfps (list): dbz 文件的路径列表。
    outfp (str): 输出文件的路径。
    progress_bar (bool, 可选): 是否显示进度条。默认为 True。

    返回:
    bool: 如果操作成功，则返回 True；否则返回 False。
    """
    # 对 dbz 文件进行稀疏化处理
    sparse_result = sparsize(dbzfps)

    if sparse_result is None:
        return False

    # 获取稀疏化处理结果的各个部分
    indices = sparse_result["indices"]
    indptr = sparse_result["indptr"]
    nozi = sparse_result["nozi"]
    dbz_arrays = sparse_result["dbz_arrays"]

    # 计算非零元素的数量和帧长度
    length = len(nozi[0])
    frame_length = len(dbzfps)
    # 将帧长度转换为字节
    frame_length_byte = struct.pack("i", frame_length)

    # 将索引长度和指针长度转换为字节
    indices_length_byte = struct.pack("i", len(indices))
    indptr_length_byte = struct.pack("i", len(indptr))

    # 将索引和指针转换为字节
    indices_byte = struct.pack(f"{len(indices)}i", *indices)
    indptr_byte = struct.pack(f"{len(indptr)}i", *indptr)

    # 初始化字节数组和数字列表
    dbz_nozarray = bytearray()
    numbers = [0] * (frame_length // 2 + frame_length % 2)

    # 根据是否需要显示进度条来选择迭代器
    if progress_bar:
        iterator = tqdm(range(length), desc="Generating bytes")
    else:
        iterator = range(length)

    # 对每个非零元素进行处理
    for i in iterator:
        for j in range(0, len(dbzfps), 2):
            try:
                # 将两个非零元素合并为一个字节
                single_byte = (
                    dbz_arrays[j][nozi[0][i], nozi[1][i]] << 4
                    | dbz_arrays[j + 1][nozi[0][i], nozi[1][i]]
                )
            except IndexError:
                # 如果只有一个非零元素，则直接使用该元素
                single_byte = dbz_arrays[j][nozi[0][i], nozi[1][i]]
            numbers[j // 2] = single_byte
        # 将生成的字节添加到字节数组中
        dbz_nozarray.extend(
            struct.pack(f"{int(frame_length/2 + frame_length % 2)}B", *numbers)
        )

    # 将所有的字节合并为一个结果
    result = (
        frame_length_byte
        + indices_length_byte
        + indptr_length_byte
        + indices_byte
        + indptr_byte
        + dbz_nozarray
    )

    # 将结果写入到指定的输出文件中
    with open(outfp, "wb") as f:
        f.write(result)

    return True


def read_smdbz(fp: str):
    """
    读取 smdbz 文件并返回一个 CSRMatrix 对象。

    该函数首先打开文件并读取内容，然后解析文件内容以获取各种参数，包括帧长度、高度、宽度、索引长度、指针长度、索引、指针和值。
    最后，使用这些参数创建一个 CSRMatrix 对象并返回。

    参数:
    fp (str): 要读取的文件的路径。

    返回:
    CSRMatrix: 从文件中读取的数据构造的 CSRMatrix 对象。
    """
    with open(fp, "rb") as f:
        content = f.read()

    offset = 0
    # 解析帧长度
    frame_length = struct.unpack("i", content[offset : offset + 4])[0]
    offset += 4
    # 解析索引长度
    indices_length = struct.unpack("i", content[offset : offset + 4])[0]
    offset += 4
    # 解析指针长度
    indptr_length = struct.unpack("i", content[offset : offset + 4])[0]
    offset += 4
    # 解析索引
    indices = struct.unpack(
        f"{indices_length}i", content[offset : offset + 4 * indices_length]
    )
    offset += 4 * indices_length
    # 解析指针
    indptr = struct.unpack(
        f"{indptr_length}i", content[offset : offset + 4 * indptr_length]
    )
    offset += 4 * indptr_length
    # 初始化值列表
    values = []
    # 解析值
    for _ in range(indices_length):
        dbz_nozarray_raw = struct.unpack(
            f"{int(frame_length/2)}B",
            content[offset : offset + int(frame_length / 2)],
        )

        single_value = []
        offset += int(frame_length / 2)

        # 对每个原始值进行处理
        for v in dbz_nozarray_raw:
            v1 = v >> 4  # 获取高4位
            v2 = v & 0x0F  # 获取低4位

            single_value.append(v1)
            single_value.append(v2)

        # 处理剩余的值
        remain = frame_length % 2
        if remain > 0:
            remain_nozarray_raw = struct.unpack(
                f"{remain}B", content[offset : offset + remain]
            )
            offset += remain
            single_value.append(remain_nozarray_raw[0])

        values.append(single_value)

    # 使用解析的参数创建 CSRMatrix 对象并返回
    return CSRMatrix(values, indices, indptr)