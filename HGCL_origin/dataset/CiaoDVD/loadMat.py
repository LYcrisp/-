#coding:UTF-8

import scipy.io as scio
import scipy.sparse as sp
import numpy as np
import random
import pickle
import datetime

# random.seed(123)
# np.random.seed(123)

def splitData(ratingMat,trustMat,categoryMat):
    """
    将原始的用户-物品评分矩阵 ratingMat 拆分为训练集和测试集，并将结果保存为稀疏矩阵格式，用于推荐系统模型训练与评估。
    拆分策略是：每个用户至少有2条交互记录，保留1条作为测试，其余作为训练
    """
    # #filter user interaction number less than 2
    train_row,train_col,train_data=[],[],[]
    test_row,test_col,test_data=[],[],[]

    ratingMat=ratingMat.tocsr()
    userList = np.where(np.sum(ratingMat!=0,axis=1)>=2)[0] 
    
    for i in userList:
        uid = i
        tmp_data = ratingMat[i].toarray()
        
        _,iidList = np.where(tmp_data!=0)
        random.shuffle(iidList)  #shuffle
        test_num = 1
        train_num = len(iidList)-1
        
        #positive data
        train_row += [i] * train_num
        train_col += list(iidList[:train_num])
        train_data += [1] * train_num

        test_row += [i] * test_num
        test_col += list(iidList[train_num:])
        test_data += [1] * test_num

        # negetive data
        # neg_iidList = np.where(np.sum(tmp_data==0))
        # neg_iidList = random.sample(list(neg_iidList),99)
        # test_row += i * 99
        # test_col += neg_iidList

    # 构建训练集和测试集的稀疏矩阵，形状与原始评分矩阵一致
    train = sp.csc_matrix((train_data,(train_row,train_col)),shape=ratingMat.shape)
    test = sp.csc_matrix((test_data,(test_row,test_col)),shape=ratingMat.shape)

    # print('train_num = %d, train rate = %.2f'%(train.nnz,train.nnz/ratingMat.nnz))
    # print('test_num = %d, test rate = %.2f'%(test.nnz,test.nnz/ratingMat.nnz))
    with open('./train.csv','wb') as fs:
        pickle.dump(train.tocsr(),fs)
    with open('./test.csv','wb') as fs:
        pickle.dump(test.tocsr(),fs)
    
def filterData(ratingMat,trustMat,categoryMat):
    """
    对推荐系统中的训练数据进行清洗，确保用户/物品/信任网络中没有孤立节点或无效数据，以提升模型训练的有效性和稳定性。
    """
    with open('./train.csv', 'rb') as fs:
        train = pickle.load(fs)
    with open('./test.csv', 'rb') as fs:
        test = pickle.load(fs)

    ratingMat=ratingMat.tocsr()
    trustMat=trustMat.tocsr()
    categoryMat=categoryMat.tocsr()

    a=np.sum(np.sum(train!=0,axis=1)==0) 
    b=np.sum(np.sum(train!=0,axis=0)==0) 
    c=np.sum(np.sum(trustMat,axis=1)==0) 
    while a!=0 or b!=0 or c!=0:
        if a!=0:
            idx,_=np.where(np.sum(train!=0,axis=1)!=0)
            train=train[idx]
            test=test[idx]
            trustMat=trustMat[idx][:,idx]
        elif b != 0:
            _, idx = np.where(np.sum(train != 0, axis=0) != 0)
            train = train[:, idx]
            test = test[:, idx]
            categoryMat = categoryMat[idx]
        elif c != 0:
            idx, _ = np.where(np.sum(trustMat, axis=1) != 0)
            train = train[idx]
            test = test[idx]
            trustMat = trustMat[idx][:, idx]
        a=np.sum(np.sum(train!=0,axis=1)==0) 
        b=np.sum(np.sum(train!=0,axis=0)==0) 
        c=np.sum(np.sum(trustMat,axis=1)==0)
    
    with open('./train.csv','wb') as fs:
        pickle.dump(train.tocsr(),fs)
    with open('./test.csv','wb') as fs:
        pickle.dump(test.tocsr(),fs)
    return ratingMat,trustMat,categoryMat

def splitAgain(ratingMat, trustMat, categoryMat):
    """
    用于修复测试集中可能存在的“空交互用户”问题。
    如果某些用户在训练/测试划分后，测试集中没有任何记录（这会导致评估指标无法计算），
    则从这些用户的训练集中随机选择一条交互记录，划归到测试集中，保证每个用户都有测试样本
    """
    # 从本地加载已保存的训练集和测试集稀疏矩阵（经过前面 splitData 和 filterData 处理后）
    with open('./train.csv', 'rb') as fs:
        train = pickle.load(fs)
    with open('./test.csv', 'rb') as fs:
        test = pickle.load(fs)

    # 将稀疏矩阵转换为LIL格式（List of Lists），方便逐元素赋值操作
    train = train.tolil()
    test = test.tolil()

    # 找到测试集中，没有任何交互记录（即整行为0）的用户索引
    # `.A` 是将 matrix 类型转换为 ndarray，便于 numpy 处理
    idx = np.where(np.sum(test != 0, axis=1).A == 0)[0]

    # 对于这些测试集中为空的用户，强制从其训练集中抽一条交互数据划归为测试集
    for i in idx:
        uid = i  # 当前用户 ID
        tmp_data = train[i].toarray()  # 获取该用户的训练交互向量
        _, iidList = np.where(tmp_data != 0)  # 找到该用户训练集中有交互的物品列表
        sample_iid = random.sample(list(iidList), 1)  # 从中随机选择一个物品
        test[uid, sample_iid] = 1  # 设置该物品为用户的测试交互
        train[uid, sample_iid] = 0  # 同时从训练集中移除该交互，避免泄露

    # 保存更新后的训练集和测试集到原路径，格式为 CSR 稀疏矩阵
    with open('./train.csv', 'wb') as fs:
        pickle.dump(train.tocsr(), fs)
    with open('./test.csv', 'wb') as fs:
        pickle.dump(test.tocsr(), fs)

        
def testNegSample(ratingMat, trustMat, categoryMat):
    """
    生成每条测试样本的负样本数据用于评估推荐性能
    """
    # 从本地加载训练集和测试集（前面应已由 splitData / filterData / splitAgain 准备好）
    with open('./train.csv', 'rb') as fs:
        train = pickle.load(fs)
    with open('./test.csv', 'rb') as fs:
        test = pickle.load(fs)

    # 将训练集转换为 DOK 格式，便于快速查找 (u, i) 是否存在交互记录
    train = train.todok()

    # 将测试集转换为 COO 格式以便逐条遍历正样本（坐标格式）
    test_u = test.tocoo().row  # 所有测试集中的用户索引（按非零项）
    test_v = test.tocoo().col  # 所有测试集中的物品索引（对应行用户的正样本）

    test_data = []  # 用于存储测试正样本和其对应的负样本（形式为 [u, i]）

    n = test_u.size  # 测试集中正样本数量

    for i in range(n):
        u = test_u[i]  # 当前测试用户
        v = test_v[i]  # 当前用户的一个正样本物品
        test_data.append([u, v])  # 添加正样本对

        # 构造 99 个负样本（随机采样物品，确保不是训练/测试中的正样本）
        for t in range(99):
            j = np.random.randint(test.shape[1])  # 在物品总数范围内随机采样一个物品 j
            while (u, j) in train or j == v:
                # 若 j 是训练集中用户 u 的正样本 或者 是当前的测试正样本，则重新采样
                j = np.random.randint(test.shape[1])
            test_data.append([u, j])  # 添加负样本对

    # 将测试样本（正+负）数据保存到本地 test_Data.csv 文件中
    with open('./test_Data.csv', 'wb') as fs:
        pickle.dump(test_data, fs)


    
if __name__ == '__main__':
    # with open('./train.csv', 'rb') as fs:
    #     train = pickle.load(fs)
    # with open('./test.csv', 'rb') as fs:
    #     test = pickle.load(fs)
    # with open('./test_Data.csv', 'rb') as fs:
    #     test_Data = pickle.load(fs)

    print(datetime.datetime.now())
    cv=1
    #raw data
    ratingsMat = 'rating.mat'
    trustMat = 'trust.mat'

    # userid, productid, categoryid, rating, helpfulness
    ratings = scio.loadmat(ratingsMat)['rating']
    # column1 trusts column 2.
    trust = scio.loadmat(trustMat)['trustnetwork']
    userNum = ratings[:, 0].max() + 1
    itemNum = ratings[:, 1].max() + 1

    #generate train data and test data
    trustMat = sp.dok_matrix((userNum, userNum))
    categoryMat = sp.dok_matrix((itemNum, 1))
    ratingMat = sp.dok_matrix((userNum, itemNum))

    ##generate ratingMat and categoryMat 
    for i in range(ratings.shape[0]):
        data = ratings[i]
        uid = data[0]
        iid = data[1]
        typeid = data[2] 
        categoryMat[iid, 0] = typeid
        ratingMat[uid, iid]=1
    ##generate trust mat
    for i in range(trust.shape[0]):
        data = trust[i]
        trustid = data[0]
        trusteeid = data[1]
        trustMat[trustid, trusteeid] = 1

    splitData(ratingMat,trustMat,categoryMat)
    ratingMat,trustMat,categoryMat=filterData(ratingMat,trustMat,categoryMat)
    splitAgain(ratingMat,trustMat,categoryMat)
    ratingMat,trustMat,categoryMat=filterData(ratingMat,trustMat,categoryMat)
    testNegSample(ratingMat,trustMat,categoryMat)
    
    ##生成categoryDict
    categoryDict = {}
    categoryData = categoryMat.toarray().reshape(-1)
    for i in range(categoryData.size):
        iid = i
        typeid = categoryData[i]
        if typeid in categoryDict:
            categoryDict[typeid].append(iid)
        else:
            categoryDict[typeid] = [iid]

    print(datetime.datetime.now())

    with open('./train.csv', 'rb') as fs:
        trainMat = pickle.load(fs)
    with open('./test_Data.csv', 'rb') as fs:
        test_Data = pickle.load(fs)

    with open('trust.csv','wb') as fs:
        pickle.dump(trustMat,fs)
  
    data = (trainMat, test_Data, trustMat, categoryMat, categoryDict)
  
    with open("data.pkl", 'wb') as fs:
        pickle.dump(data, fs)

    print('Done')