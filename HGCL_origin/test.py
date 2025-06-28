import pickle
import numpy as np
from scipy.sparse import csr_matrix

# 加载 data.pkl
with open('dataset/CiaoDVD/data.pkl', 'rb') as f:
    trainMat, test_Data, trustMat, categoryMat, categoryDict = pickle.load(f)

# 将类别矩阵转换为 numpy 数组
category_array = categoryMat.toarray().reshape(-1)

# 打印基本信息
num_users, num_items = trainMat.shape
print(f'用户数: {num_users}, 物品数: {num_items}')
print(f'类别矩阵 shape: {categoryMat.shape}')
print(f'前10个物品的类别ID: {category_array[:10]}')

# 查看某个物品的类别
item_id = 42  # 举例
print(f'物品 {item_id} 的类别是: {category_array[item_id]}')

# 查看某个类别下有哪些物品
cat_id = category_array[item_id]
print(f'类别 {cat_id} 下的物品有: {categoryDict[cat_id]}')

# 查看物品的交互用户（从 trainMat 中）
item_col = trainMat.tocsc()[:, item_id]  # 转换为按列访问
user_ids = item_col.nonzero()[0]
print(f'物品 {item_id} 被以下用户交互过: {user_ids.tolist()}')

# 如果你想可视化各类别的分布
import collections
cat_counts = collections.Counter(category_array.tolist())
print(f"共有 {len(cat_counts)} 个不同类别，每类物品数量如下：")
for cid, count in cat_counts.items():
    print(f"类别 {cid}：{count} 个物品")
