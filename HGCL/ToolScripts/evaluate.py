import numpy as np
import torch

def hit(gt_item, pred_items):
	"""
	Hit Ratio（HR）指标计算：
	判断目标物品是否出现在模型推荐的 top-K 列表中。
	如果命中，返回1；否则返回0。
	"""
	if gt_item in pred_items:
		return 1
	return 0

def ndcg(gt_item, pred_items):
	"""
	Normalized Discounted Cumulative Gain（NDCG）指标计算：
	衡量目标物品在推荐列表中的排名位置，
	排名越靠前得分越高，位置从1开始，采用 log2 折扣。
	若不在列表中，得分为0。
	"""
	if gt_item in pred_items:
		index = pred_items.index(gt_item)  # 获取目标物品在推荐列表中的位置
		return np.reciprocal(np.log2(index + 2))  # 计算 NDCG 分数（+2 是为了从位置1开始）
	return 0

def metrics(model, test_loader, top_k):
	"""
	模型评估函数，计算 HR 和 NDCG 两个评估指标的平均值。
	适用于测试集中每个样本包含 1 个正样本 + 若干负样本（通常为99负样本）。

	参数：
	- model: 推荐模型
	- test_loader: DataLoader，按批次提供 (user, item, label)
	- top_k: 评估时选取的推荐列表长度（top-K）

	返回：
	- HR@K 的均值
	- NDCG@K 的均值
	"""
	HR, NDCG = [], []  # 用于存储每个样本的 HR 和 NDCG 分数

	for user, item, label in test_loader:
		user = user.cuda()  # 将用户和物品张量移至 GPU
		item = item.cuda()

		predictions = model(user, item)  # 模型对候选物品打分（正样本 + 负样本）
		_, indices = torch.topk(predictions, top_k)  # 取 top-K 的物品索引
		recommends = torch.take(item, indices).cpu().numpy().tolist()  # 转为推荐列表（物品ID）

		gt_item = item[0].item()  # 正样本物品ID（第一个为正，其余为负）
		HR.append(hit(gt_item, recommends))  # 计算当前样本的 HR
		NDCG.append(ndcg(gt_item, recommends))  # 计算当前样本的 NDCG

	return np.mean(HR), np.mean(NDCG)  # 返回两个指标的平均值
