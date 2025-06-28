import numpy as np
import scipy.sparse as sp
import pickle
import datetime
from tqdm import tqdm
np.random.seed(30)  #random seed

# 加载预处理好的数据
with open("data.pkl", 'rb') as fs:
    data = pickle.load(fs)

# 解包数据，trainMat：用户-物品评分矩阵；trustMat：用户之间的信任关系；
# categoryMat：物品类别矩阵；categoryDict：类别到物品ID的映射
(trainMat, _, trustMat, categoryMat, categoryDict) = data

# 将评分矩阵转为压缩稀疏行格式，便于高效读取
trainMat = trainMat.tocsr()

# 获取用户数和物品数
userNum, itemNum = trainMat.shape


print(datetime.datetime.now())
########################user-item distance matrix###############################
# 初始化一个用户数 + 物品数 维度的稀疏矩阵，表示用户-物品交互的双向连接
UiDistance_mat = (sp.dok_matrix((userNum + itemNum, userNum + itemNum))).tocsr()

# 将用户-物品评分矩阵复制到异构图的相应位置（用户→物品方向）
UiDistance_mat[:userNum, userNum:] = trainMat

# 将评分矩阵的转置复制到异构图中（物品→用户方向）
UiDistance_mat[userNum:, :userNum] = trainMat.T


########################user distance matrix###############################
# 初始化用户-用户之间的稀疏距离矩阵
UserDistance_mat = sp.dok_matrix((userNum, userNum))

# 将信任矩阵转换为COO格式以获取坐标信息
tmp_trustMat = trustMat.tocoo()
uidList1, uidList2 = tmp_trustMat.row, tmp_trustMat.col  # 信任关系中的用户对

# 在用户之间添加边（彼此信任关系为 1）
UserDistance_mat[uidList1, uidList2] = 1.0
UserDistance_mat[uidList2, uidList1] = 1.0  # 双向信任

# 加上单位矩阵（表示自连接），转换为 CSR 格式
UserDistance_mat = (UserDistance_mat + sp.eye(userNum)).tocsr()

# 保存用户距离矩阵到文件
with open('./UserDistance_mat.pkl', 'wb') as fs:
    pickle.dump(UserDistance_mat, fs)


########################item distance matrix###############################
# 初始化物品-物品距离矩阵
ItemDistance_mat = sp.dok_matrix((itemNum, itemNum))

# 遍历每个物品，为其随机添加同类别的“近邻物品”
for i in tqdm(range(itemNum)):
    itemType = categoryMat[i, 0]  # 获取物品 i 的类别 ID
    itemList = categoryDict[itemType]  # 获取与 i 同类别的所有物品 ID 列表
    itemList = np.array(itemList)

    # 从同类中随机选取 1% 的物品作为“相似物品”
    itemList2 = np.random.choice(itemList, size=int(itemList.size * 0.01), replace=False)
    itemList2 = itemList2.tolist()

    tmp = [i] * len(itemList2)  # 构造 item i 与选中的 itemList2 成对的索引
    ItemDistance_mat[tmp, itemList2] = 1.0  # i → 同类物品 设置为 1.0
    ItemDistance_mat[itemList2, tmp] = 1.0  # 对称连接

##final result
ItemDistance_mat = (ItemDistance_mat + sp.eye(itemNum)).tocsr()
with open('./ItemDistance_mat.pkl', 'wb') as fs:
    pickle.dump(ItemDistance_mat, fs)
    
print(datetime.datetime.now())
#############save data################
distanceMat = (UserDistance_mat, ItemDistance_mat, UiDistance_mat)

with open('distanceMat_addIUUI.pkl', 'wb') as fs:
    pickle.dump(distanceMat, fs)
print("Done")


