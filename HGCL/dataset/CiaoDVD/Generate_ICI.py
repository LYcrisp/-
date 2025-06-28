import numpy as np
import scipy.sparse as sp
import pickle
import datetime
from tqdm import tqdm
np.random.seed(30)  #random seed

"""
根据物品的类别信息，构建一个稀疏的物品-物品“距离矩阵”用于后续推荐任务或图结构建模。
"""

# 加载数据：包含用户-物品评分矩阵、信任矩阵、类别矩阵和类别字典
with open("data.pkl", 'rb') as fs:
    data = pickle.load(fs)

(trainMat, _, trustMat, categoryMat, categoryDict) = data  # 解包数据，其中 trainMat 是用户-物品评分矩阵
trainMat = trainMat.tocsr()  # 转换为压缩稀疏行格式，便于高效访问
userNum, itemNum = trainMat.shape  # 获取用户数量和物品数量


print(datetime.datetime.now())

########################item distance matrix###############################
ItemDistance_mat = sp.dok_matrix((itemNum, itemNum))  # 初始化一个稀疏字典格式的矩阵，用于表示物品之间的距离（稀疏矩阵）

# 对于每一个物品 i，找到与其属于同一类别的物品，并构造一定的“距离连接”
for i in tqdm(range(itemNum)):  # 遍历所有物品
    itemType = categoryMat[i, 0]  # 获取物品 i 所属的类别（假设第一列表示类别）
    itemList = categoryDict[itemType]  # 获取所有属于该类别的物品索引列表
    itemList = np.array(itemList)  # 转换为 NumPy 数组
    itemList2 = np.random.choice(itemList, size=int(itemList.size * 0.01), replace=False)  # 随机选择 1% 的同类物品作为连接
    itemList2 = itemList2.tolist()  # 转换为 Python 列表
    tmp = [i] * len(itemList2)  # 构建与 item i 匹配的列表
    ItemDistance_mat[tmp, itemList2] = 2.0  # 设置 i 与这些同类物品之间的“距离”为 2.0（可理解为相似度）
    ItemDistance_mat[itemList2, tmp] = 2.0  # 对称设置（无向图）

##final result
ItemDistance_mat = (ItemDistance_mat + sp.eye(itemNum)).tocsr()  # 给每个物品自身加上单位值（self-loop），并转为压缩稀疏行格式
with open('./ICI.pkl', 'wb') as fs:
    pickle.dump(ItemDistance_mat, fs)  # 保存为二进制文件，后续模型中可能使用
print(datetime.datetime.now())  # 打印结束时间
print("Done")  # 任务完成提示



