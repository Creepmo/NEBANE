# Node-Edge Bilateral Attributed Network Embedding

NEBANE的实现版本

## 论文地址

https://www.researchgate.net/publication/337773659_Node-Edge_Bilateral_Attributed_Network_Embedding

## ppt地址

链接：https://pan.baidu.com/s/1XKnay-QqKP_Yxq_UnoBX0Q 
提取码：j89m

## 数据集地址

链接：https://pan.baidu.com/s/1HCrrgSX-wZbPk8euqeGeGw 
提取码：xaga

## 使用方法

```python
cd Alg
python main.py -options
```

#### 基本参数选择介绍

- --datapath，输入数据集名称，以.mat形式存储，位置在./Alg/Datasets下，初始值设为./Datasets/Amazon，原有数据集选项：[AMiner, Amazon]；
- --batch_size，每批次训练样本个数，初始值设为200；
- --epoch_num，训练周期数，初始值设为20；
- --num_sampled，负采样个数，初始值设为5；
- --eta，正则化损失函数权重，初始值设为1.0；
- --alpha，节点属性相似性损失函数权重，初始值设为1000；
- --beta，连边属性相似性损失函数权重，初始值设为1000；
- ---lr，学习率，初始值设为0.01；
- --train_size，输入连边的训练比率，用于链路预测实验，如果节点分类实验设置为1.0，初始值为1.0。

#### 输入数据格式

以文件夹形式存储，包含四个文件：node_attr，link_attr，label，init/
- node_attr：节点属性文件，第一行为节点个数，下面每一行依次为从node id0开始的每个节点的属性，每个属性以空格分隔；
- link_attr：边属性文件，每一行格式为:head_id tail_id#link attr(每个属性以空格分隔)；
- label：节点label文件，格式为node_id node_name node_label;
- init/：节点embedding初始化文件，以deepwalk pre-train产生的embedding作为初始化embedding。

#### 输出数据格式

在原有数据集的文件夹下，生成node_emb文件，格式与初始化的embedding文件相同

- 第一行为：节点个数 表示维度

- 下面为每个节点的隐式向量：节点id 隐式向量1  隐式向量2 ...

#### 主要源文件介绍

- main.py：主函数；
- graph.py：图数据预处理模块；
- train.py：训练函数;
- evaluation.py：评估函数。

#### 评估函数

- node_classification()：节点分类实验；
- node_clustering(): 节点聚类实验;
- simlilarity_search()：相似节点搜索实验；
- link_prediction()：链路预测实验；
- node_recommendation()：节点推荐实验。

