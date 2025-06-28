# Research task code

## HGCL模型

HGCL文件夹下是我改进后的版本，HGCL_origin是原始版本
作者的代码开源的地址是   https://github.com/HKUDS/HGCL

运行模型的命令是   

```cmd
python main.py --dataset CiaoDVD --ssl_temp 0.6 --ssl_ureg 0.04 --ssl_ireg 0.05 --lr 0.055 --reg 0.065 --ssl_beta 0.3 --rank 3
```

数据集太大了，我只留下了HGCL文件夹下的CiaoDV数据集，如果需要运行原模型直接把数据集复制到HGCL_origin下的 dataset 文件夹中就可以了

具体的创新内容是构造了⼀个 `nn.Embedding(item_features,  hide_dim)`，⽤于将每个物品的内容信息进⾏嵌⼊。

优化的结果是HR值由0.7325涨到了0.7411，NDGC由0.5185涨到了0.5293。

随机数使用作者采取的默认随机数，所以每次运行得到的结果是一样的。

## DiffKG模型

尝试了对热门物品加⼊惩罚项，并尝试加入 Item  共现图，但是优化效果不明显。
