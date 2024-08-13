import numpy as np
import gzip
from struct import unpack


def read_images(path: str) -> np.array:
    """
    读取 MNIST 图片数据文件。
    
    参数:
    path (str): 图片数据文件的路径。
    
    返回:
    numpy.ndarray: 包含图片数据的 NumPy 数组。
"""
    with gzip.open(path, "rb") as f:
        # 解析前 16 个字节，获取魔数、图像数量、行数和列数
        magic, num_images, rows, cols = unpack('>IIII', f.read(16))
        # 读取剩余的数据，每个像素占用一个字节
        images = np.frombuffer(f.read(), dtype=np.uint8).reshape(num_images, rows, cols)
        
        print(f"file: {path}, Magic Number: {magic}")
        print(f"image nums: {num_images}, col: {cols}, row: {rows}")
        print(f"{images.shape = }\n")
        
        return images


def read_labels(path: str) -> np.array:
    """
    读取 MNIST 标签数据文件。
    
    参数:
    path (str): 标签数据文件的路径。
    
    返回:
    numpy.ndarray: 包含标签数据的 NumPy 数组。
    """
    with gzip.open(path, "rb") as f:
        # 解析前 8 个字节，获取魔数和标签数量
        magic, num_labels = unpack('>II', f.read(8))
        # 读取剩余的数据，每个标签占用一个字节
        label_data = np.frombuffer(f.read(), dtype=np.uint8)
        
        print(f"file: {path}, Magic Number: {magic}")
        print(f"label nums: {num_labels}")
        print(f"{label_data.shape = }\n")
        
        return label_data