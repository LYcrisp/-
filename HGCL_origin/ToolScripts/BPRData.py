import numpy as np 
import scipy.sparse as sp
import torch.utils.data as data
import pickle

class BPRData(data.Dataset):
    def __init__(self, data,
                 num_item, train_mat=None, num_ng=0, is_training=None):
        super(BPRData, self).__init__()
        """
        构造函数：用于初始化数据集对象（可用于训练或测试）

        参数说明：
        - data: 原始正样本数据，格式为 [user, item] 对组成的列表或数组
        - num_item: 总的物品数量（用于负采样）
        - train_mat: 训练集交互矩阵（用于判断用户是否已交互某物品）
        - num_ng: 每个正样本要采样的负样本数量（此处虽然设置了，但没被用）
        - is_training: 是否用于训练阶段（True：训练，False：测试）
        """
        self.data = np.array(data)
        self.num_item = num_item
        self.train_mat = train_mat
        self.num_ng = num_ng
        self.is_training = is_training

    def ng_sample(self):
        """
        负采样函数，仅在训练阶段使用。
        对每个正样本随机生成一个负样本（即用户未交互过的物品），用于 BPR loss。
        """
        assert self.is_training, 'no need to sampling when testing'

        tmp_trainMat = self.train_mat.todok()  # 转换为 DOK 格式便于查询 (u, i) 是否为正样本
        length = self.data.shape[0]  # 正样本数量
        self.neg_data = np.random.randint(low=0, high=self.num_item, size=length)  # 初始化负样本物品ID数组

        for i in range(length):
            uid = self.data[i][0]       # 用户ID
            iid = self.neg_data[i]      # 当前负样本候选物品ID
            if (uid, iid) in tmp_trainMat:
                # 如果采样到的物品在正样本中已存在，则重新采样直到找到负样本
                while (uid, iid) in tmp_trainMat:
                    iid = np.random.randint(low=0, high=self.num_item)
                self.neg_data[i] = iid  # 更新为新的负样本

    def __len__(self):
        """
        返回数据集中样本的数量（正样本数量）
        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        支持索引访问：返回第 idx 个样本的数据。
        如果是训练模式，则返回三元组 (user, 正样本, 负样本)
        如果是测试模式，则仅返回 (user, 正样本)
        """
        user = self.data[idx][0]
        item_i = self.data[idx][1]
        if self.is_training:
            neg_data = self.neg_data
            item_j   = neg_data[idx]  # 对应的负样本物品ID
            return user, item_i, item_j
        else:
            return user, item_i
