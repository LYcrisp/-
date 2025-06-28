import numpy as np 
import scipy.sparse as sp

import torch.utils.data as data

import pickle

"""
构造 PyTorch 的 Dataset 类，用于测试阶段的数据加载，主要用于评估推荐模型性能，测试时和模型的输出做对比
"""

class TstData(data.Dataset):
    def __init__(self, coomat, trnMat):
        super(TstData, self).__init__()
        """
        初始化函数，用于构建推荐系统测试阶段的数据集。

        参数：
        - coomat: 测试集的稀疏矩阵（COO格式），记录用户在测试集中的正样本交互
        - trnMat: 训练集的稀疏矩阵，用于标记用户已知的交互记录（用于过滤推荐候选）

        功能：
        - 构建测试用户列表和对应测试正样本位置（商品ID）
        - 构造训练交互矩阵（CSR格式），方便快速判断某物品是否已训练中出现
        """
        # 将训练集转换为CSR格式，并将非零位置变为1.0（标记交互过的item）
        self.csrmat = (trnMat.tocsr() != 0) * 1.0

        # 初始化一个长度为用户数的列表，每个位置存该用户在测试集中的正样本item列表
        tstLocs = [None] * coomat.shape[0]
        tstUsrs = set()  # 记录出现过测试数据的用户集合

        for i in range(len(coomat.data)):
            row = coomat.row[i]  # 用户ID
            col = coomat.col[i]  # 测试正样本物品ID

            if tstLocs[row] is None:
                tstLocs[row] = list()
            tstLocs[row].append(col)  # 添加正样本
            tstUsrs.add(row)  # 记录该用户

        # 将用户集合转为 numpy 数组，便于索引和迭代
        tstUsrs = np.array(list(tstUsrs))
        self.tstUsrs = tstUsrs         # 所有在测试集中出现过的用户ID
        self.tstLocs = tstLocs         # 每个用户对应的测试集正样本商品ID列表

    def __len__(self):
        """
        返回测试用户数量（即在测试集中出现过交互的用户数）
        """
        return len(self.tstUsrs)

    def __getitem__(self, idx):
        """
        返回第 idx 个测试用户及其训练交互向量。

        返回值：
        - 用户ID
        - 用户在训练集中的交互记录，形状为 [num_items] 的二值向量
        """
        return self.tstUsrs[idx], np.reshape(self.csrmat[self.tstUsrs[idx]].toarray(), [-1])
