import pickle
import numpy as np
from scipy.sparse import coo_matrix
from Params import args
import scipy.sparse as sp
import torch
import torch.utils.data as data
import torch.utils.data as dataloader
from collections import defaultdict
from tqdm import tqdm
import random
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class DataHandler:
	def __init__(self):
		# 根据参数选择数据集路径（mind, alibaba, lastfm）
		if args.data == 'mind':
			predir = './Datasets/mind/'
		elif args.data == 'alibaba':
			predir = './Datasets/alibaba/'
		elif args.data == 'lastfm':
			predir = './Datasets/lastfm/'
		self.predir = predir
		self.trnfile = predir + 'trnMat.pkl'  # 训练集交互稀疏矩阵路径
		self.tstfile = predir + 'tstMat.pkl'  # 测试集交互稀疏矩阵路径
		self.kgfile = predir + 'kg.txt'  # 知识图谱三元组路径

	def loadOneFile(self, filename):
		# 从 pickle 文件中加载稀疏交互矩阵（用户-物品交互）
		with open(filename, 'rb') as fs:
			ret = (pickle.load(fs) != 0).astype(np.float32)  # 二值化处理
		if type(ret) != coo_matrix:
			ret = sp.coo_matrix(ret)  # 保证是 COO 格式稀疏矩阵
		return ret

	def readTriplets(self, file_name):
		# 从 kg.txt 读取三元组数据（实体-关系-实体）
		can_triplets_np = np.loadtxt(file_name, dtype=np.int32)
		can_triplets_np = np.unique(can_triplets_np, axis=0)  # 去重

		# 构造逆关系三元组：t -> h，关系ID偏移
		inv_triplets_np = can_triplets_np.copy()
		inv_triplets_np[:, 0] = can_triplets_np[:, 2]
		inv_triplets_np[:, 2] = can_triplets_np[:, 0]
		inv_triplets_np[:, 1] = can_triplets_np[:, 1] + max(can_triplets_np[:, 1]) + 1

		# 合并正向和反向三元组
		triplets = np.concatenate((can_triplets_np, inv_triplets_np), axis=0)

		# 统计关系和实体总数
		n_relations = max(triplets[:, 1]) + 1
		args.relation_num = n_relations
		args.entity_n = max(max(triplets[:, 0]), max(triplets[:, 2])) + 1

		return triplets

	def buildGraphs(self, triplets):
		# 构建图字典（邻接表）和边列表
		kg_dict = defaultdict(list)  # 用于存储邻接信息
		kg_edges = list()  # 存储(h, t, r)三元组

		print("Begin to load knowledge graph triples ...")

		kg_counter_dict = {}  # 记录每个头实体已连接的尾实体，避免重复边

		for h_id, r_id, t_id in tqdm(triplets, ascii=True):
			if h_id not in kg_counter_dict:
				kg_counter_dict[h_id] = set()
			if t_id not in kg_counter_dict[h_id]:
				kg_counter_dict[h_id].add(t_id)
			else:
				continue  # 跳过重复边
			kg_edges.append([h_id, t_id, r_id])
			kg_dict[h_id].append((r_id, t_id))  # 构建邻接表

		return kg_edges, kg_dict

	def buildKGMatrix(self, kg_edges):
		# 构建实体-实体稀疏邻接矩阵（忽略关系，只保留连边）
		edge_list = []
		for h_id, t_id, r_id in kg_edges:
			edge_list.append((h_id, t_id))
		edge_list = np.array(edge_list)

		# 构建 CSR 格式稀疏邻接矩阵
		kgMatrix = sp.csr_matrix(
			(np.ones_like(edge_list[:, 0]), (edge_list[:, 0], edge_list[:, 1])),
			dtype='float64',
			shape=(args.entity_n, args.entity_n)
		)
		return kgMatrix

	def normalizeAdj(self, mat):
		# 对稀疏邻接矩阵进行对称归一化处理（D^{-0.5} A D^{-0.5}）
		degree = np.array(mat.sum(axis=-1))
		dInvSqrt = np.reshape(np.power(degree, -0.5), [-1])
		dInvSqrt[np.isinf(dInvSqrt)] = 0.0
		dInvSqrtMat = sp.diags(dInvSqrt)
		return mat.dot(dInvSqrtMat).transpose().dot(dInvSqrtMat).tocoo()

	def makeTorchAdj(self, mat):
		# 构建用户-物品二部图的稀疏归一化邻接矩阵（拼接 user/item 自环）
		a = sp.csr_matrix((args.user, args.user))  # 用户自环（空）
		b = sp.csr_matrix((args.item, args.item))  # 物品自环（空）
		mat = sp.vstack([sp.hstack([a, mat]), sp.hstack([mat.transpose(), b])])
		mat = (mat != 0) * 1.0  # 二值化
		mat = (mat + sp.eye(mat.shape[0])) * 1.0  # 加单位阵（自环）
		mat = self.normalizeAdj(mat)  # 对称归一化

		# 转换为 PyTorch 稀疏张量并转到 GPU
		idxs = torch.from_numpy(np.vstack([mat.row, mat.col]).astype(np.int64))
		vals = torch.from_numpy(mat.data.astype(np.float32))
		shape = torch.Size(mat.shape)
		device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		return torch.sparse.FloatTensor(idxs, vals, shape).to(device)

	def RelationDictBuild(self):
		# 构建 relation_dict: head -> tail -> relation 的字典结构
		relation_dict = {}
		for head in self.kg_dict:
			relation_dict[head] = {}
			for (relation, tail) in self.kg_dict[head]:
				relation_dict[head][tail] = relation
		return relation_dict

	def buildUIMatrix(self, mat):
		# 构建用户-物品稀疏交互矩阵的 PyTorch 表示
		idxs = torch.from_numpy(np.vstack([mat.row, mat.col]).astype(np.int64))
		vals = torch.from_numpy(mat.data.astype(np.float32))
		shape = torch.Size(mat.shape)
		return torch.sparse.FloatTensor(idxs, vals, shape).to(device)

	def LoadData(self):
		# === 1. 加载训练/测试交互矩阵 ===
		trnMat = self.loadOneFile(self.trnfile)
		tstMat = self.loadOneFile(self.tstfile)
		self.trnMat = trnMat

		# 获取用户数和物品数
		args.user, args.item = trnMat.shape

		# === 2. 构建二部图邻接矩阵 ===
		self.torchBiAdj = self.makeTorchAdj(trnMat)  # 用户-物品图结构（稀疏图）
		self.ui_matrix = self.buildUIMatrix(trnMat)  # 原始交互矩阵稀疏张量

		# === 3. 构建训练/测试数据加载器 ===
		trnData = TrnData(trnMat)
		self.trnLoader = dataloader.DataLoader(trnData, batch_size=args.batch, shuffle=True, num_workers=0)
		tstData = TstData(tstMat, trnMat)
		self.tstLoader = dataloader.DataLoader(tstData, batch_size=args.tstBat, shuffle=False, num_workers=0)

		# === 4. 加载并构建知识图谱 ===
		kg_triplets = self.readTriplets(self.kgfile)
		self.kg_edges, self.kg_dict = self.buildGraphs(kg_triplets)
		self.kg_matrix = self.buildKGMatrix(self.kg_edges)

		print("kg shape: ", self.kg_matrix.shape)
		print("number of edges in KG: ", len(self.kg_edges))

		# === 5. 构建扩散模型的输入数据 ===
		self.diffusionData = DiffusionData(self.kg_matrix.toarray())  # numpy 矩阵传入
		self.diffusionLoader = dataloader.DataLoader(self.diffusionData, batch_size=args.batch, shuffle=True,
													 num_workers=0)

		# === 6. 构建关系字典（用于可解释性） ===
		self.relation_dict = self.RelationDictBuild()

		# new--------统计每个 item 被多少个 user 交互过
		item_freq = np.array(self.trnMat.sum(axis=0)).squeeze()  # shape: (item,)
		item_popularity = item_freq / item_freq.max()  # 归一化到 [0, 1]
		self.item_popularity = torch.tensor(item_popularity, dtype=torch.float32).cuda()


# === 训练数据 Dataset 类 ===
class TrnData(data.Dataset):
	def __init__(self, coomat):
		self.rows = coomat.row                     # 用户索引（行）
		self.cols = coomat.col                     # 物品索引（列）
		self.dokmat = coomat.todok()               # 转换为字典形式稀疏矩阵，方便快速查找
		self.negs = np.zeros(len(self.rows)).astype(np.int32)  # 存放负采样的物品索引

	def negSampling(self):
		# 为每个正样本随机采样一个负样本（用户未交互的物品）
		for i in range(len(self.rows)):
			u = self.rows[i]                       # 当前用户
			while True:
				iNeg = np.random.randint(args.item)  # 随机采样物品
				if (u, iNeg) not in self.dokmat:     # 如果用户未与该物品交互
					break
			self.negs[i] = iNeg                    # 存入负样本列表

	def __len__(self):
		# 返回总样本数（即交互记录数）
		return len(self.rows)

	def __getitem__(self, idx):
		# 返回三元组：(用户, 正样本物品, 负样本物品)
		return self.rows[idx], self.cols[idx], self.negs[idx]


# === 测试数据 Dataset 类 ===
class TstData(data.Dataset):
	def __init__(self, coomat, trnMat):
		# 将训练集稀疏矩阵转换为 CSR 格式并二值化，用于快速判断用户历史物品
		self.csrmat = (trnMat.tocsr() != 0) * 1.0

		tstLocs = [None] * coomat.shape[0]  # 每个用户对应的测试物品集合
		tstUsrs = set()                     # 存在测试数据的用户集合

		# 遍历测试集中的交互记录
		for i in range(len(coomat.data)):
			row = coomat.row[i]            # 用户索引
			col = coomat.col[i]            # 物品索引
			if tstLocs[row] is None:
				tstLocs[row] = list()
			tstLocs[row].append(col)       # 记录该用户在测试集中的点击物品
			tstUsrs.add(row)               # 记录该用户

		tstUsrs = np.array(list(tstUsrs))  # 转换为数组形式
		self.tstUsrs = tstUsrs             # 有测试交互的用户列表
		self.tstLocs = tstLocs             # 每个用户对应的测试物品列表（索引）

	def __len__(self):
		# 返回有测试行为的用户数量
		return len(self.tstUsrs)

	def __getitem__(self, idx):
		# 返回：用户索引、该用户在训练集中的交互向量（形状为 [item_num]）
		return self.tstUsrs[idx], np.reshape(self.csrmat[self.tstUsrs[idx]].toarray(), [-1])

	
# === 知识图谱扩散矩阵的数据封装 Dataset 类 ===
class DiffusionData(data.Dataset):
	def __init__(self, data):
		self.data = data  # 通常为 numpy 格式的邻接矩阵或扩散结果矩阵

	def __getitem__(self, index):
		item = self.data[index]
		return torch.FloatTensor(item), index  # 返回当前行向量（实体或物品表示）+ 索引

	def __len__(self):
		return len(self.data)  # 数据总行数（通常为实体数或物品数）
