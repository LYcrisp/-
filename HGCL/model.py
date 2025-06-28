import torch as t
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.sparse as sp
import numpy as np


### HGCL
class MODEL(nn.Module):
    def __init__(self, args, userNum, itemNum, userMat, itemMat, uiMat, hide_dim, Layers, item_features):

        super(MODEL, self).__init__()

        # 保存初始化参数
        self.args = args
        self.userNum = userNum
        self.itemNum = itemNum
        self.uuMat = userMat  # 用户-用户邻接矩阵
        self.iiMat = itemMat  # 物品-物品邻接矩阵
        self.uiMat = uiMat  # 用户-物品交互邻接矩阵
        self.hide_dim = hide_dim  # 隐藏层维度
        self.LayerNums = Layers  # GCN层数

        # 构建稀疏邻接矩阵（用户-物品）
        uimat = self.uiMat[: self.userNum, self.userNum:]
        values = torch.FloatTensor(uimat.tocoo().data)
        indices = np.vstack((uimat.tocoo().row, uimat.tocoo().col))
        i = torch.LongTensor(indices)
        v = torch.FloatTensor(values)
        shape = uimat.tocoo().shape
        uimat1 = torch.sparse.FloatTensor(i, v, torch.Size(shape))
        self.uiadj = uimat1  # 用户-物品稀疏图
        self.iuadj = uimat1.transpose(0, 1)  # 物品-用户稀疏图（转置）

        # 门控机制参数初始化（用户和物品）
        self.gating_weightub = nn.Parameter(torch.FloatTensor(1, hide_dim))
        nn.init.xavier_normal_(self.gating_weightub.data)
        self.gating_weightu = nn.Parameter(torch.FloatTensor(hide_dim, hide_dim))
        nn.init.xavier_normal_(self.gating_weightu.data)
        self.gating_weightib = nn.Parameter(torch.FloatTensor(1, hide_dim))
        nn.init.xavier_normal_(self.gating_weightib.data)
        self.gating_weighti = nn.Parameter(torch.FloatTensor(hide_dim, hide_dim))
        nn.init.xavier_normal_(self.gating_weighti.data)

        # 新增物品内容嵌入（类别 one-hot 编码）
        self.item_content_embedding = nn.Embedding(item_features, hide_dim)
        nn.init.xavier_normal_(self.item_content_embedding.weight)

        # 多层GCN编码器堆叠
        self.encoder = nn.ModuleList()
        for _ in range(self.LayerNums):
            self.encoder.append(GCN_layer())

        self.k = args.rank  # meta变换中的rank维度

        # MLP 融合 item_embed 与 content_embed
        self.item_fusion_mlp = nn.Sequential(
            nn.Linear(2 * hide_dim, hide_dim),
            nn.ReLU(),
            nn.Linear(hide_dim, hide_dim)
        )

        # MLP用于meta特征提取及转换矩阵计算（共四个网络）
        self.mlp = MLP(hide_dim, hide_dim * self.k, hide_dim // 2, hide_dim * self.k)
        self.mlp1 = MLP(hide_dim, hide_dim * self.k, hide_dim // 2, hide_dim * self.k)
        self.mlp2 = MLP(hide_dim, hide_dim * self.k, hide_dim // 2, hide_dim * self.k)
        self.mlp3 = MLP(hide_dim, hide_dim * self.k, hide_dim // 2, hide_dim * self.k)

        # Meta特征线性映射层（用户、物品）
        self.meta_netu = nn.Linear(hide_dim * 3, hide_dim)
        self.meta_neti = nn.Linear(hide_dim * 3, hide_dim)

        # 嵌入字典，分别初始化用户/物品及其图注意力嵌入
        self.embedding_dict = nn.ModuleDict({
            'uu_emb': torch.nn.Embedding(userNum, hide_dim).cuda(),
            'ii_emb': torch.nn.Embedding(itemNum, hide_dim).cuda(),
            'user_emb': torch.nn.Embedding(userNum, hide_dim).cuda(),
            'item_emb': torch.nn.Embedding(itemNum, hide_dim).cuda(),
        })

    # 初始化嵌入向量
    def init_weight(self, userNum, itemNum, hide_dim):
        initializer = nn.init.xavier_uniform_
        embedding_dict = nn.ParameterDict({
            'user_emb': nn.Parameter(initializer(torch.empty(userNum, hide_dim))),
            'item_emb': nn.Parameter(initializer(torch.empty(itemNum, hide_dim))),
        })
        return embedding_dict

    # 稀疏矩阵转换为 PyTorch 稀疏张量
    def sparse_mx_to_torch_sparse_tensor(self, sparse_mx):
        if type(sparse_mx) != sp.coo_matrix:
            sparse_mx = sparse_mx.tocoo().astype(np.float32)
        indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
        values = torch.from_numpy(sparse_mx.data).float()
        shape = torch.Size(sparse_mx.shape)
        return torch.sparse.FloatTensor(indices, values, shape)

    # 内容对比损失函数（用于对比学习）
    def content_contrastive_loss(self, item_embed, content_embed):
        """
        item_embed: 融合后的 item 表示 (N × D)
        content_embed: 原始内容嵌入 (N × D)
        """
        item_embed = F.normalize(item_embed, p=2, dim=1)
        content_embed = F.normalize(content_embed, p=2, dim=1)

        pos_score = torch.sum(item_embed * content_embed, dim=1)
        all_score = torch.matmul(item_embed, content_embed.T)

        pos_score = torch.exp(pos_score / self.args.ssl_temp)
        all_score = torch.sum(torch.exp(all_score / self.args.ssl_temp), dim=1)

        loss = -torch.log(pos_score / (all_score + 1e-8)).mean()
        return loss

    # Meta正则化损失函数（对比学习思想）
    def metaregular(self, em0, em, adj):
        def row_column_shuffle(embedding):
            corrupted = embedding[:, torch.randperm(embedding.shape[1])]
            return corrupted[torch.randperm(embedding.shape[0])]

        def score(x1, x2):
            x1 = F.normalize(x1, p=2, dim=-1)
            x2 = F.normalize(x2, p=2, dim=-1)
            return torch.sum(x1 * x2, dim=1)

        Adj_Norm = torch.from_numpy(np.sum(adj, axis=1)).float().cuda()
        adj = self.sparse_mx_to_torch_sparse_tensor(adj)
        edge_emb = torch.spmm(adj.cuda(), em) / Adj_Norm  # 图卷积聚合邻居
        graph = torch.mean(edge_emb, dim=0)  # 图均值向量
        pos = score(em0, graph)
        neg = score(row_column_shuffle(em0), graph)
        return torch.mean(-torch.log(torch.sigmoid(pos - neg)))

    # 门控机制（用户）
    def self_gatingu(self, em):
        return em * torch.sigmoid(torch.matmul(em, self.gating_weightu) + self.gating_weightub)

    # 门控机制（物品）
    def self_gatingi(self, em):
        return em * torch.sigmoid(torch.matmul(em, self.gating_weighti) + self.gating_weightib)

    # meta-transform模块：低秩变换用户/物品辅助特征
    def metafortansform(self, auxiembedu, targetembedu, auxiembedi, targetembedi):
        uneighbor = torch.matmul(self.uiadj.cuda(), self.ui_itemEmbedding)
        ineighbor = torch.matmul(self.iuadj.cuda(), self.ui_userEmbedding)

        tembedu = self.meta_netu(torch.cat((auxiembedu, targetembedu, uneighbor), dim=1).detach())
        tembedi = self.meta_neti(torch.cat((auxiembedi, targetembedi, ineighbor), dim=1).detach())

        metau1 = self.mlp(tembedu).reshape(-1, self.hide_dim, self.k)
        metau2 = self.mlp1(tembedu).reshape(-1, self.k, self.hide_dim)
        metai1 = self.mlp2(tembedi).reshape(-1, self.hide_dim, self.k)
        metai2 = self.mlp3(tembedi).reshape(-1, self.k, self.hide_dim)

        # 计算变换矩阵的平均偏置
        meta_biasu = torch.mean(metau1, dim=0)
        meta_biasu1 = torch.mean(metau2, dim=0)
        meta_biasi = torch.mean(metai1, dim=0)
        meta_biasi1 = torch.mean(metai2, dim=0)

        # 低秩权重
        low_weightu1 = F.softmax(metau1 + meta_biasu, dim=1)
        low_weightu2 = F.softmax(metau2 + meta_biasu1, dim=1)
        low_weighti1 = F.softmax(metai1 + meta_biasi, dim=1)
        low_weighti2 = F.softmax(metai2 + meta_biasi1, dim=1)

        # 变换
        tembedus = torch.sum(auxiembedu.unsqueeze(-1) * low_weightu1, dim=1)
        tembedus = torch.sum(tembedus.unsqueeze(-1) * low_weightu2, dim=1)
        tembedis = torch.sum(auxiembedi.unsqueeze(-1) * low_weighti1, dim=1)
        tembedis = torch.sum(tembedis.unsqueeze(-1) * low_weighti2, dim=1)
        return tembedus, tembedis

    # 前向传播
    def forward(self, iftraining, uid, iid, item_content=None, norm=1):
        item_index = np.arange(self.itemNum)
        user_index = np.arange(self.userNum)
        ui_index = np.array(user_index.tolist() + [i + self.userNum for i in item_index])

        userembed0 = self.embedding_dict['user_emb'].weight
        itemembed0 = self.embedding_dict['item_emb'].weight.clone()  # 克隆防止 in-place 修改
        uu_embed0 = self.self_gatingu(userembed0)
        ii_embed0 = self.self_gatingi(itemembed0)

        if item_content is not None:
            item_content_embed = self.item_content_embedding(item_content)
            concat = torch.cat([itemembed0[iid], item_content_embed], dim=1)
            fused = self.item_fusion_mlp(concat)

            itemembed0 = itemembed0.clone()  # 再次克隆确保安全
            itemembed0.index_copy_(0, iid if isinstance(iid, torch.Tensor) else torch.tensor(iid, device=fused.device), fused)

        # 拼接 UI 嵌入
        self.ui_embeddings = torch.cat([userembed0, ii_embed0], 0)
        self.all_user_embeddings = [uu_embed0]
        self.all_item_embeddings = [ii_embed0]
        self.all_ui_embeddings = [self.ui_embeddings]

        for i, layer in enumerate(self.encoder):
            if i == 0:
                userEmbeddings0 = layer(uu_embed0, self.uuMat, user_index)
                itemEmbeddings0 = layer(ii_embed0, self.iiMat, item_index)
                uiEmbeddings0 = layer(self.ui_embeddings, self.uiMat, ui_index)
            else:
                userEmbeddings0 = layer(userEmbeddings, self.uuMat, user_index)
                itemEmbeddings0 = layer(itemEmbeddings, self.iiMat, item_index)
                uiEmbeddings0 = layer(uiEmbeddings, self.uiMat, ui_index)

            self.ui_userEmbedding0, self.ui_itemEmbedding0 = torch.split(uiEmbeddings0, [self.userNum, self.itemNum])
            userEd = (userEmbeddings0 + self.ui_userEmbedding0) / 2.0
            itemEd = (itemEmbeddings0 + self.ui_itemEmbedding0) / 2.0
            userEmbeddings = userEd
            itemEmbeddings = itemEd
            uiEmbeddings = torch.cat([userEd, itemEd], 0)

            if norm == 1:
                self.all_user_embeddings.append(F.normalize(userEmbeddings0, p=2, dim=1))
                self.all_item_embeddings.append(F.normalize(itemEmbeddings0, p=2, dim=1))
                self.all_ui_embeddings.append(F.normalize(uiEmbeddings0, p=2, dim=1))
            else:
                self.all_user_embeddings.append(userEmbeddings)
                self.all_item_embeddings.append(itemEmbeddings)
                self.all_ui_embeddings.append(uiEmbeddings)

        self.userEmbedding = torch.mean(torch.stack(self.all_user_embeddings, dim=1), dim=1)
        self.itemEmbedding = torch.mean(torch.stack(self.all_item_embeddings, dim=1), dim=1)
        self.uiEmbedding = torch.mean(torch.stack(self.all_ui_embeddings, dim=1), dim=1)
        self.ui_userEmbedding, self.ui_itemEmbedding = torch.split(self.uiEmbedding, [self.userNum, self.itemNum])

        metatsuembed, metatsiembed = self.metafortansform(
            self.userEmbedding, self.ui_userEmbedding,
            self.itemEmbedding, self.ui_itemEmbedding
        )
        self.userEmbedding = self.userEmbedding + metatsuembed
        self.itemEmbedding = self.itemEmbedding + metatsiembed

        metaregloss = 0
        if iftraining:
            self.reg_lossu = self.metaregular(
                self.ui_userEmbedding[uid.cpu().numpy()],
                self.userEmbedding,
                self.uuMat[uid.cpu().numpy()]
            )
            self.reg_lossi = self.metaregular(
                self.ui_itemEmbedding[iid.cpu().numpy()],
                self.itemEmbedding,
                self.iiMat[iid.cpu().numpy()]
            )
            metaregloss = (self.reg_lossu + self.reg_lossi) / 2.0

        return self.userEmbedding, self.itemEmbedding, \
            self.args.wu1 * self.ui_userEmbedding + self.args.wu2 * self.userEmbedding, \
            self.args.wi1 * self.ui_itemEmbedding + self.args.wi2 * self.itemEmbedding, \
            self.ui_userEmbedding, self.ui_itemEmbedding, metaregloss


class GCN_layer(nn.Module):
    def __init__(self):
        super(GCN_layer, self).__init__()

    def sparse_mx_to_torch_sparse_tensor(self, sparse_mx):
        """将 scipy 的稀疏矩阵转换为 PyTorch 稀疏张量"""
        if type(sparse_mx) != sp.coo_matrix:
            sparse_mx = sparse_mx.tocoo().astype(np.float32)
        indices = torch.from_numpy(
            np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
        values = torch.from_numpy(sparse_mx.data).float()
        shape = torch.Size(sparse_mx.shape)
        return torch.sparse.FloatTensor(indices, values, shape)

    def normalize_adj(self, adj):
        """对邻接矩阵进行对称归一化处理"""
        adj = sp.coo_matrix(adj)
        rowsum = np.array(adj.sum(1))  # 每一行求和（即每个节点的度）
        d_inv_sqrt = np.power(rowsum, -0.5).flatten()  # D^(-1/2)
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.  # 避免除零
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)  # 构造对角矩阵
        return (d_mat_inv_sqrt).dot(adj).dot(d_mat_inv_sqrt).tocoo()  # D^(-1/2) A D^(-1/2)

    def forward(self, features, Mat, index):
        """
        features: 输入节点的特征 (N x D)
        Mat: 对应的邻接矩阵（scipy稀疏矩阵）
        index: 当前batch中要更新的节点索引（List或Tensor）
        """
        subset_Mat = Mat  # 对应子图邻接矩阵（可支持子图采样）
        subset_features = features  # 输入的所有节点特征

        # 步骤1：归一化邻接矩阵
        subset_Mat = self.normalize_adj(subset_Mat)

        # 步骤2：转换为 PyTorch 稀疏张量
        subset_sparse_tensor = self.sparse_mx_to_torch_sparse_tensor(subset_Mat).cuda()

        # 步骤3：图卷积乘法（稀疏邻接矩阵 @ 特征）
        out_features = torch.spmm(subset_sparse_tensor, subset_features)

        # 步骤4：仅更新 index 位置的特征，其余保留原特征
        new_features = torch.empty(features.shape).cuda()
        new_features[index] = out_features

        # 非 index 的位置使用原始特征（保持不变）
        dif_index = np.setdiff1d(torch.arange(features.shape[0]), index)
        new_features[dif_index] = features[dif_index]

        return new_features


class MLP(torch.nn.Module):
    def __init__(self, input_dim, feature_dim, hidden_dim, output_dim,
                 feature_pre=True, layer_num=2, dropout=True, **kwargs):
        super(MLP, self).__init__()

        self.feature_pre = feature_pre  # 是否有前置特征变换层
        self.layer_num = layer_num  # 总层数（至少为2）
        self.dropout = dropout  # 是否启用 dropout

        # 如果启用特征预处理层，则定义 linear_pre
        if feature_pre:
            self.linear_pre = nn.Linear(input_dim, feature_dim, bias=True)
        else:
            self.linear_first = nn.Linear(input_dim, hidden_dim)

        # 中间隐藏层（除首层与末层）
        self.linear_hidden = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for i in range(layer_num - 2)
        ])

        # 输出层
        self.linear_out = nn.Linear(feature_dim, output_dim, bias=True)

    def forward(self, data):
        x = data

        # 可选的特征预处理层
        if self.feature_pre:
            x = self.linear_pre(x)

        # PReLU 激活（可学习参数的ReLU）
        prelu = nn.PReLU().cuda()
        x = prelu(x)

        # 多个隐藏层 + tanh 激活 + dropout
        for i in range(self.layer_num - 2):
            x = self.linear_hidden[i](x)
            x = F.tanh(x)  # 使用 tanh 非线性激活
            if self.dropout:
                x = F.dropout(x, training=self.training)

        # 输出层 + L2归一化
        x = self.linear_out(x)
        x = F.normalize(x, p=2, dim=-1)  # 最后输出向量单位化（便于对比学习或余弦计算）

        return x

















