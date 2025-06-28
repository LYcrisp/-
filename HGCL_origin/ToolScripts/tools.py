import torch as t
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import numpy as np
# import scipy.sparse as sp

def showSparseTensor(tensor):
    # 显示稀疏张量中每一行非零元素的值
    index = t.nonzero(tensor)  # 获取非零元素的位置索引
    countArr = t.sum(tensor != 0, dim=1).cpu().numpy()  # 每一行非零元素的数量
    start = 0
    end = 0
    tmp = tensor[index[:, 0], index[:, 1]].cpu().detach().numpy()  # 获取非零元素值的列表
    for i in countArr:
        start = end
        end += i
        print(tmp[start: end])  # 按行打印非零值

def masked_softmax(vec, mask, dim=1, epsilon=1e-6):
    # 带掩码的 softmax，常用于注意力机制中避免对 padding 位置进行归一化
    exps = t.exp(vec)  # 先对原始输入取指数
    masked_exps = exps * mask.float()  # 将 mask 为 0 的位置置零（屏蔽）
    masked_sums = masked_exps.sum(dim, keepdim=True) + epsilon  # 避免除零，加一个小常数
    ret = (masked_exps / masked_sums)  # 执行归一化，得到 softmax 输出
    return ret

def list2Str(s):
    # 将整数列表转为以下划线分隔的字符串，例如 [1,2,3] → "1_2_3"
    ret = str(s[0])
    for i in range(1, len(s)):
        ret = ret + '_' + str(s[i])
    return ret

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """将 scipy 的稀疏矩阵（如 COO 格式）转换为 PyTorch 的稀疏张量"""
    if type(sparse_mx) != sp.coo_matrix:
        sparse_mx = sparse_mx.tocoo().astype(np.float32)  # 强制转换为 float32 的 COO 格式
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))  # 构建稀疏索引 [2, N]
    values = torch.from_numpy(sparse_mx.data)  # 对应的非零值
    shape = torch.Size(sparse_mx.shape)  # 原始稀疏矩阵的形状
    return torch.sparse.FloatTensor(indices, values, shape)  # 构建稀疏张量对象

def normalize_adj(adj):
    """对邻接矩阵进行对称归一化：D^{-1/2} * A * D^{-1/2}，常用于图神经网络"""
    adj = sp.coo_matrix(adj)  # 转为 COO 稀疏格式
    rowsum = np.array(adj.sum(1))  # 计算每个节点的度数（行求和）
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()  # 计算 D^{-1/2}
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.  # 将无穷值（如度为0）置为0，避免计算错误
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)  # 构造对角矩阵 D^{-1/2}
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()  # 返回对称归一化后的邻接矩阵


class sampleLargeGraph(nn.Module):
    def __init__(self):
        super(sampleLargeGraph, self).__init__()
        self.flag = 0  # 保留字段，可用于标记采样状态或条件，当前未使用

    def makeMask(self, nodes, size):
        """
        构造节点掩码数组，用于标记哪些节点已被采样。

        参数：
        - nodes: 已采样的节点索引列表
        - size: 总节点数

        返回值：
        - mask: 大小为 size 的 1D 数组，已采样节点置为 0，未采样节点为 1
        """
        mask = np.ones(size)
        if not nodes is None:
            mask[nodes] = 0.0
        return mask

    def updateBdgt(self, adj, nodes):
        """
        计算当前采样节点的邻接权重总和（即预算 budget），用于确定下轮采样概率。

        参数：
        - adj: 图的邻接矩阵（稀疏矩阵）
        - nodes: 当前采样的节点列表

        返回值：
        - ret: 节点的累计邻接和（作为下一轮采样预算）
        """
        if nodes is None:
            return 0
        ret = np.sum(adj[nodes], axis=0).A  # 相当于邻接矩阵的行求和，转换为 dense 数组
        return ret

    def sample(self, budget, mask, sampNum):
        """
        基于 budget 和掩码进行节点采样。

        参数：
        - budget: 当前每个节点的邻接预算（采样权重）
        - mask: 节点掩码（哪些节点已经采样过）
        - sampNum: 本轮采样数量

        返回值：
        - pckNodes: 采样得到的节点索引数组
        """
        # 使用 mask * budget 的平方作为采样分数
        score = (mask * np.reshape(np.array(budget), [-1])) ** 2
        norm = np.sum(score)

        # 若分数全为 0，随机采样一个节点
        if norm == 0:
            return np.random.choice(len(score), 1)

        score = list(score / norm)  # 归一化为概率分布
        arrScore = np.array(score)
        posNum = np.sum(np.array(score) != 0)  # 有效候选数量

        if posNum < sampNum:
            # 若非零得分节点数不足，补充一些零得分节点
            pckNodes1 = np.squeeze(np.argwhere(arrScore != 0))
            pckNodes2 = np.random.choice(
                np.squeeze(np.argwhere(arrScore == 0.0)),
                min(len(score) - posNum, sampNum - posNum),
                replace=False
            )
            pckNodes = np.concatenate([pckNodes1, pckNodes2], axis=0)
        else:
            # 正常按概率进行无放回采样
            pckNodes = np.random.choice(len(score), sampNum, p=score, replace=False)
        return pckNodes

    def forward(self, graph_adj, pickNodes, sampDepth=2, sampNum=20000, Flage=False):
        """
        主函数：图节点采样流程。

        参数：
        - graph_adj: 输入图的邻接矩阵（scipy sparse）
        - pickNodes: 初始采样节点列表（可以是训练节点或感兴趣区域）
        - sampDepth: 采样层数（迭代深度）
        - sampNum: 每一层采样节点数
        - Flage: 暂未使用

        返回值：
        - pckNodes: 所有最终被采样到的节点索引组成的一维数组
        """
        nodeMask = self.makeMask(pickNodes, graph_adj.shape[0])  # 初始化掩码
        nodeBdgt = self.updateBdgt(graph_adj, pickNodes)  # 根据初始节点生成第一轮预算

        for i in range(sampDepth):
            newNodes = self.sample(nodeBdgt, nodeMask, sampNum)  # 基于预算采样新节点
            nodeMask = nodeMask * self.makeMask(newNodes, graph_adj.shape[0])  # 更新掩码
            if i == sampDepth - 1:
                break
            nodeBdgt += self.updateBdgt(graph_adj, newNodes)  # 更新预算（累计邻居影响）

        # 最终掩码中为 0 的位置即为被采样到的节点
        pckNodes = np.reshape(np.argwhere(nodeMask == 0), [-1])
        return pckNodes


class sampleHeterLargeGraph(nn.Module):
    def __init__(self):
        super(sampleHeterLargeGraph, self).__init__()
        self.flag = 1  # 用于打印一次采样参数，只在第一次 forward 中使用

    def makeMask(self, nodes, size):
        """
        创建掩码数组，用于标记哪些节点已经被采样。

        参数:
        - nodes: 已采样的节点列表
        - size: 节点总数（user 或 item）

        返回:
        - mask: 数组，已采样节点为 0，其余为 1
        """
        mask = np.ones(size)
        if not nodes is None:
            mask[nodes] = 0.0
        return mask

    def updateBdgt(self, adj, nodes):
        """
        根据当前采样节点计算其邻接矩阵的累加预算值（每个节点的连边强度）。

        参数:
        - adj: 稀疏邻接矩阵（user-item 或 item-user）
        - nodes: 当前采样节点索引

        返回:
        - ret: 每个目标节点的预算值向量（numpy array）
        """
        if nodes is None:
            return 0
        ret = np.sum(adj[nodes], axis=0).A  # 提取邻接矩阵中行的总和
        return ret

    def sample(self, budget, mask, sampNum):
        """
        根据 budget 和掩码进行采样。

        参数:
        - budget: 每个节点的采样权重值（importance）
        - mask: 当前未被采样的节点掩码
        - sampNum: 本轮想要采样的节点数量

        返回:
        - pckNodes: 采样得到的节点索引列表
        """
        score = (mask * np.array(budget).reshape(-1)) ** 2  # 被屏蔽的节点乘以 0，其他平方放大差异
        norm = np.sum(score)
        if norm == 0:
            return np.random.choice(len(score), 1)  # 如果所有得分为 0，随机选一个节点
        score = list(score / norm)  # 归一化为概率分布
        arrScore = np.array(score)
        posNum = np.sum(arrScore != 0)  # 可被选择的节点数

        if posNum < sampNum:
            # 如果得分非零的节点不足采样数量，则补充一些随机节点
            pckNodes1 = np.squeeze(np.argwhere(arrScore != 0))
            pckNodes2 = np.random.choice(
                np.squeeze(np.argwhere(arrScore == 0.0)),
                min(len(score) - posNum, sampNum - posNum),
                replace=False
            )
            pckNodes = np.concatenate([pckNodes1, pckNodes2], axis=0)
        else:
            # 按得分概率进行无放回采样
            pckNodes = np.random.choice(len(score), sampNum, p=score, replace=False)
        return pckNodes

    def forward(self, graph_adj, pickNodes, pickNodes2=None, sampDepth=2, sampNum=10000):
        """
        执行异构图双边采样流程（适用于用户-物品的 bipartite graph）

        参数:
        - graph_adj: 用户-物品的邻接矩阵 (user × item)
        - pickNodes: 初始用户节点集合
        - pickNodes2: 初始物品节点集合（可为空）
        - sampDepth: 采样层数（迭代次数）
        - sampNum: 每轮每侧采样节点数量

        返回:
        - pckNodes1: 采样到的用户节点索引
        - pckNodes2: 采样到的物品节点索引
        """
        if self.flag:
            print('sample depth:', sampDepth, '     sample Num:', sampNum)
            self.flag = 0  # 仅第一次 forward 输出信息

        # 初始化用户、物品掩码
        nodeMask1 = self.makeMask(pickNodes, graph_adj.shape[0])  # 用户掩码
        nodeMask2 = self.makeMask(pickNodes2, graph_adj.shape[1])  # 物品掩码

        # 计算物品端 budget（由初始用户节点到物品）
        nodeBdgt2 = self.updateBdgt(graph_adj, pickNodes)

        # 如果初始物品节点为空，则使用 budget 采样出一个初始物品节点集
        if pickNodes2 is None:
            pickNodes2 = self.sample(nodeBdgt2, nodeMask2, len(pickNodes))
            nodeMask2 = nodeMask2 * self.makeMask(pickNodes2, graph_adj.shape[1])

        # 计算用户端 budget（由物品返回用户）
        nodeBdgt1 = self.updateBdgt(graph_adj.T, pickNodes2)

        # 多轮迭代采样（从物品到用户再到物品...）
        for i in range(sampDepth):
            newNodes1 = self.sample(nodeBdgt1, nodeMask1, sampNum)
            nodeMask1 = nodeMask1 * self.makeMask(newNodes1, graph_adj.shape[0])

            newNodes2 = self.sample(nodeBdgt2, nodeMask2, sampNum)
            nodeMask2 = nodeMask2 * self.makeMask(newNodes2, graph_adj.shape[1])

            if i == sampDepth - 1:
                break  # 最后一层采样后不更新 budget

            # 更新下一轮的预算
            nodeBdgt2 += self.updateBdgt(graph_adj, newNodes1)
            nodeBdgt1 += self.updateBdgt(graph_adj.T, newNodes2)

        # 找到所有最终被采样的用户和物品节点
        pckNodes1 = np.reshape(np.where(nodeMask1 == 0), [-1])  # 用户
        pckNodes2 = np.reshape(np.where(nodeMask2 == 0), [-1])  # 物品
        return pckNodes1, pckNodes2


# import pickle
# if __name__ == "__main__":
#     with open(r'dataset/CiaoDVD/data.pkl', 'rb') as fs:
#         data = pickle.load(fs) 
#     trainMat, _, _, _, _ = data
#     sample_graph_nodes = sampleHeterLargeGraph()
#     seed_nodes = np.random.choice(trainMat.shape[0], 1000)
#     sample_graph_nodes(trainMat, seed_nodes)
