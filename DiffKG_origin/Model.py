import torch
from torch import nn
import torch.nn.functional as F
from Params import args
import numpy as np
import random
from torch_scatter import scatter_sum, scatter_softmax
import math

init = nn.init.xavier_uniform_
uniformInit = nn.init.uniform


class Model(nn.Module):
	def __init__(self, handler):
		super(Model, self).__init__()

		# 初始化用户、实体、关系的嵌入向量参数
		self.uEmbeds = nn.Parameter(init(torch.empty(args.user, args.latdim)))  # 用户嵌入矩阵
		self.eEmbeds = nn.Parameter(init(torch.empty(args.entity_n, args.latdim)))  # 实体嵌入矩阵（物品、演员、类别等）
		self.rEmbeds = nn.Parameter(init(torch.empty(args.relation_num, args.latdim)))  # 关系嵌入矩阵（has_genre 等）

		# 多层 GCN 层（堆叠 GCNLayer 结构）
		self.gcnLayers = nn.Sequential(*[GCNLayer() for i in range(args.gnn_layer)])

		# 初始化 RGAT 模块：关系感知图注意力网络（用于知识图谱建模）
		self.rgat = RGAT(args.latdim, args.layer_num_kg, args.mess_dropout_rate)

		# 读取知识图谱字典，供采样使用
		self.kg_dict = handler.kg_dict

		# 从知识图谱字典中采样三元组边，并编码为图输入格式
		self.edge_index, self.edge_type = self.sampleEdgeFromDict(self.kg_dict, triplet_num=args.triplet_num)

	def getEntityEmbeds(self):
		# 返回实体嵌入向量（可供外部使用）
		return self.eEmbeds

	def getUserEmbeds(self):
		# 返回用户嵌入向量
		return self.uEmbeds

	def forward(self, adj, mess_dropout=True, kg=None):
		# 前向传播函数
		if kg == None:
			# 如果没有传入 KG 图结构，使用初始化采样的 edge_index 和 edge_type
			hids_KG = self.rgat.forward(self.eEmbeds, self.rEmbeds, [self.edge_index, self.edge_type], mess_dropout)
		else:
			# 否则使用外部传入的 KG 图结构
			hids_KG = self.rgat.forward(self.eEmbeds, self.rEmbeds, kg, mess_dropout)

		# 拼接用户嵌入与实体嵌入（其中前 args.item 个实体是物品）
		embeds = torch.concat([self.uEmbeds, hids_KG[:args.item, :]], axis=0)

		# GCN 多层信息传播
		embedsLst = [embeds]
		for gcn in self.gcnLayers:
			embeds = gcn(adj, embedsLst[-1])
			embedsLst.append(embeds)
		embeds = sum(embedsLst)  # 残差聚合：逐层嵌入求和

		# 返回：用户嵌入（前 N 行），物品嵌入（后 M 行）
		return embeds[:args.user], embeds[args.user:]

	def sampleEdgeFromDict(self, kg_dict, triplet_num=None):
		# 从知识图谱中按头实体采样部分三元组边
		sampleEdges = []
		for h in kg_dict:
			t_list = kg_dict[h]  # 获取当前 head 所连接的 (relation, tail) 列表
			if triplet_num != -1 and len(t_list) > triplet_num:
				sample_edges_i = random.sample(t_list, triplet_num)  # 若超出采样限制则随机抽样
			else:
				sample_edges_i = t_list
			for r, t in sample_edges_i:
				sampleEdges.append([h, t, r])  # 存储三元组为边
		return self.getEdges(sampleEdges)  # 转换为 PyTorch 图格式

	def getEdges(self, kg_edges):
		# 将三元组边列表转换为 PyTorch 图表示（edge_index, edge_type）
		graph_tensor = torch.tensor(kg_edges)  # [N, 3]，每行是 (h, t, r)
		index = graph_tensor[:, :-1]  # 取前两列为 (h, t)
		type = graph_tensor[:, -1]  # 最后一列为 r
		return index.t().long().cuda(), type.long().cuda()  # 转换为 CUDA tensor，返回转置后的 edge_index 和 edge_type


# ======= GCN 层：用于用户-物品图上信息传播 =======
class GCNLayer(nn.Module):
	def __init__(self):
		super(GCNLayer, self).__init__()  # 初始化 GCN 层（无参数）

	def forward(self, adj, embeds):
		# 输入：adj 稀疏邻接矩阵（torch.sparse）、embeds 节点嵌入
		# 输出：聚合后的嵌入（邻居传播）
		return torch.spmm(adj, embeds)  # 稀疏矩阵乘法：A × X

		
# ======= RGAT 模块：关系感知图注意力网络，用于知识图谱传播 =======
class RGAT(nn.Module):
	def __init__(self, latdim, n_hops, mess_dropout_rate=0.4):
		super(RGAT, self).__init__()
		self.mess_dropout_rate = mess_dropout_rate  # 消息传播时的 dropout 概率
		self.W = nn.Parameter(init(torch.empty(size=(2*latdim, latdim)), gain=nn.init.calculate_gain('relu')))
		# 注意力权重计算的投影矩阵，用于将 [head, tail] 拼接后变换

		self.leakyrelu = nn.LeakyReLU(0.2)  # 激活函数，用于 attention 打分
		self.n_hops = n_hops                # 图消息传播层数（跳数）
		self.dropout = nn.Dropout(p=mess_dropout_rate)  # 用于每层之间的 dropout

	def agg(self, entity_emb, relation_emb, kg):
		# 单跳的关系感知消息聚合过程
		edge_index, edge_type = kg        # 边索引和对应的关系类型
		head, tail = edge_index           # 分别获取头尾实体索引（1D Tensor）

		# 拼接 head 和 tail 的嵌入作为注意力输入（[N, 2D]）
		a_input = torch.cat([entity_emb[head], entity_emb[tail]], dim=-1)

		# 执行线性变换（通过 self.W）并乘以关系嵌入，得到注意力分数
		e_input = torch.multiply(torch.mm(a_input, self.W), relation_emb[edge_type]).sum(-1)
		e = self.leakyrelu(e_input)  # 激活

		# 对 attention 权重进行 softmax（针对每个 head 节点归一化）
		e = scatter_softmax(e, head, dim=0, dim_size=entity_emb.shape[0])

		# 用 attention 权重加权 tail 实体嵌入
		agg_emb = entity_emb[tail] * e.view(-1, 1)

		# 按 head 节点聚合（加权和）
		agg_emb = scatter_sum(agg_emb, head, dim=0, dim_size=entity_emb.shape[0])

		# 残差连接：加上原始的 entity_emb，保留自身特征
		agg_emb = agg_emb + entity_emb

		return agg_emb

	def forward(self, entity_emb, relation_emb, kg, mess_dropout=True):
		entity_res_emb = entity_emb  # 残差初始值

		for _ in range(self.n_hops):  # 多跳传播
			entity_emb = self.agg(entity_emb, relation_emb, kg)  # 聚合邻居

			if mess_dropout:
				entity_emb = self.dropout(entity_emb)  # 消息 dropout，防止过拟合

			entity_emb = F.normalize(entity_emb)  # L2 归一化

			# 多层残差融合（跳数越多保留比例越高）
			entity_res_emb = args.res_lambda * entity_res_emb + entity_emb

		return entity_res_emb  # 输出传播后的实体表示（用于物品等）


class Denoise(nn.Module):
	def __init__(self, in_dims, out_dims, emb_size, norm=False, dropout=0.5):
		super(Denoise, self).__init__()
		self.in_dims = in_dims  # 输入每层维度列表，例如 [d0, d1, d2, ...]
		self.out_dims = out_dims  # 输出每层维度列表
		self.time_emb_dim = emb_size  # 时间步嵌入维度（即时间编码维度）
		self.norm = norm  # 是否对输入特征做归一化

		# 时间步嵌入层，线性变换，输入和输出都是 time_emb_dim
		self.emb_layer = nn.Linear(self.time_emb_dim, self.time_emb_dim)

		# 第一层输入维度调整：将输入第0层维度加上时间嵌入维度
		in_dims_temp = [self.in_dims[0] + self.time_emb_dim] + self.in_dims[1:]

		out_dims_temp = self.out_dims  # 输出维度保持不变

		# 构造输入部分的多层线性层，逐层对应 in_dims_temp 中的相邻维度
		self.in_layers = nn.ModuleList(
			[nn.Linear(d_in, d_out) for d_in, d_out in zip(in_dims_temp[:-1], in_dims_temp[1:])])

		# 构造输出部分的多层线性层，逐层对应 out_dims_temp 中的相邻维度
		self.out_layers = nn.ModuleList(
			[nn.Linear(d_in, d_out) for d_in, d_out in zip(out_dims_temp[:-1], out_dims_temp[1:])])

		# dropout 层，防止过拟合
		self.drop = nn.Dropout(dropout)

		# 初始化所有层的权重和偏置
		self.init_weights()

	def init_weights(self):
		# 初始化输入层权重：使用 He/Xavier 正态分布初始化
		for layer in self.in_layers:
			size = layer.weight.size()
			std = np.sqrt(2.0 / (size[0] + size[1]))
			layer.weight.data.normal_(0.0, std)
			layer.bias.data.normal_(0.0, 0.001)

		# 初始化输出层权重，同上
		for layer in self.out_layers:
			size = layer.weight.size()
			std = np.sqrt(2.0 / (size[0] + size[1]))
			layer.weight.data.normal_(0.0, std)
			layer.bias.data.normal_(0.0, 0.001)

		# 初始化时间嵌入层权重
		size = self.emb_layer.weight.size()
		std = np.sqrt(2.0 / (size[0] + size[1]))
		self.emb_layer.weight.data.normal_(0.0, std)
		self.emb_layer.bias.data.normal_(0.0, 0.001)

	def forward(self, x, timesteps, mess_dropout=True):
		# 计算时间步的频率编码，构造类似 Transformer 中的位置编码
		freqs = torch.exp(-math.log(10000) * torch.arange(start=0, end=self.time_emb_dim // 2, dtype=torch.float32) / (
					self.time_emb_dim // 2)).cuda()
		# timesteps 是一个形状为 [batch_size] 的张量，将其扩展后与频率相乘
		temp = timesteps[:, None].float() * freqs[None]

		# 将频率编码拆分成 cos 和 sin 拼接，构造时间嵌入 [batch_size, time_emb_dim]
		time_emb = torch.cat([torch.cos(temp), torch.sin(temp)], dim=-1)

		# 如果时间维度为奇数，补零扩维
		if self.time_emb_dim % 2:
			time_emb = torch.cat([time_emb, torch.zeros_like(time_emb[:, :1])], dim=-1)

		# 通过线性层处理时间嵌入，得到可训练的时间特征
		emb = self.emb_layer(time_emb)

		# 是否对输入特征 x 做归一化
		if self.norm:
			x = F.normalize(x)

		# 是否应用 dropout
		if mess_dropout:
			x = self.drop(x)

		# 将输入特征与时间嵌入拼接
		h = torch.cat([x, emb], dim=-1)

		# 通过输入层的多层网络，层间激活为 tanh
		for i, layer in enumerate(self.in_layers):
			h = layer(h)
			h = torch.tanh(h)

		# 通过输出层的多层网络，层间激活为 tanh，最后一层无激活
		for i, layer in enumerate(self.out_layers):
			h = layer(h)
			if i != len(self.out_layers) - 1:
				h = torch.tanh(h)

		return h  # 返回输出特征


class GaussianDiffusion(nn.Module):
	def __init__(self, noise_scale, noise_min, noise_max, steps, beta_fixed=True):
		super(GaussianDiffusion, self).__init__()

		self.noise_scale = noise_scale
		self.noise_min = noise_min
		self.noise_max = noise_max
		self.steps = steps

		if noise_scale != 0:
			self.betas = torch.tensor(self.get_betas(), dtype=torch.float64).cuda()
			if beta_fixed:
				self.betas[0] = 0.0001

			self.calculate_for_diffusion()

	def get_betas(self):
		start = self.noise_scale * self.noise_min
		end = self.noise_scale * self.noise_max
		variance = np.linspace(start, end, self.steps, dtype=np.float64)
		alpha_bar = 1 - variance
		betas = []
		betas.append(1 - alpha_bar[0])
		for i in range(1, self.steps):
			betas.append(min(1 - alpha_bar[i] / alpha_bar[i-1], 0.999))
		return np.array(betas)
	
	def calculate_for_diffusion(self):
		alphas = 1.0 - self.betas
		self.alphas_cumprod = torch.cumprod(alphas, axis=0).cuda()
		self.alphas_cumprod_prev = torch.cat([torch.tensor([1.0]).cuda(), self.alphas_cumprod[:-1]]).cuda()
		self.alphas_cumprod_next = torch.cat([self.alphas_cumprod[1:], torch.tensor([0.0]).cuda()]).cuda()

		self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
		self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
		self.log_one_minus_alphas_cumprod = torch.log(1.0 - self.alphas_cumprod)
		self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod)
		self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1)

		self.posterior_variance = (
			self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
		)
		self.posterior_log_variance_clipped = torch.log(torch.cat([self.posterior_variance[1].unsqueeze(0), self.posterior_variance[1:]]))
		self.posterior_mean_coef1 = (self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod))
		self.posterior_mean_coef2 = ((1.0 - self.alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - self.alphas_cumprod))

	def p_sample(self, model, x_start, steps):
		if steps == 0:
			x_t = x_start
		else:
			t = torch.tensor([steps-1] * x_start.shape[0]).cuda()
			x_t = self.q_sample(x_start, t)
		
		indices = list(range(self.steps))[::-1]

		for i in indices:
			t = torch.tensor([i] * x_t.shape[0]).cuda()
			model_mean, model_log_variance = self.p_mean_variance(model, x_t, t)
			x_t = model_mean
		return x_t
			
	def q_sample(self, x_start, t, noise=None):
		if noise is None:
			noise = torch.randn_like(x_start)
		return self._extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start + self._extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
	
	def _extract_into_tensor(self, arr, timesteps, broadcast_shape):
		arr = arr.cuda()
		res = arr[timesteps].float()
		while len(res.shape) < len(broadcast_shape):
			res = res[..., None]
		return res.expand(broadcast_shape)
	
	def p_mean_variance(self, model, x, t):
		model_output = model(x, t, False)

		model_variance = self.posterior_variance
		model_log_variance = self.posterior_log_variance_clipped

		model_variance = self._extract_into_tensor(model_variance, t, x.shape)
		model_log_variance = self._extract_into_tensor(model_log_variance, t, x.shape)

		model_mean = (self._extract_into_tensor(self.posterior_mean_coef1, t, x.shape) * model_output + self._extract_into_tensor(self.posterior_mean_coef2, t, x.shape) * x)
		
		return model_mean, model_log_variance

	def training_losses(self, model, x_start, ui_matrix, userEmbeds, itmEmbeds, batch_index):
		batch_size = x_start.size(0)
		ts = torch.randint(0, self.steps, (batch_size,)).long().cuda()
		noise = torch.randn_like(x_start)
		if self.noise_scale != 0:
			x_t = self.q_sample(x_start, ts, noise)
		else:
			x_t = x_start

		model_output = model(x_t, ts)
		mse = self.mean_flat((x_start - model_output) ** 2)

		weight = self.SNR(ts - 1) - self.SNR(ts)
		weight = torch.where((ts == 0), 1.0, weight)

		diff_loss = weight * mse

		item_user_matrix = torch.spmm(ui_matrix, model_output[:, :args.item].t()).t()
		itmEmbeds_kg = torch.mm(item_user_matrix, userEmbeds)
		ukgc_loss = self.mean_flat((itmEmbeds_kg - itmEmbeds[batch_index]) ** 2)

		return diff_loss, ukgc_loss
		
	def mean_flat(self, tensor):
		return tensor.mean(dim=list(range(1, len(tensor.shape))))
	
	def SNR(self, t):
		self.alphas_cumprod = self.alphas_cumprod.cuda()
		return self.alphas_cumprod[t] / (1 - self.alphas_cumprod[t])