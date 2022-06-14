# Paddle-FSL-GNN

## 一、简介
论文：《Few-Shot Learning with Graph Neural Networks》[论文链接](https://arxiv.org/pdf/1711.04043.pdf)

论文提出用推理的 prism来研究few-shot学习的问题，该模型由输入图像的集合构建，其标签可以被观察到或不能被观察到。通过同化通用的消息传递推理算法和神经网络对应的算法，定义了一个图神经网络架构，该架构概括了最近提出的几个few-shot学习模型。除了提供改进的数值性能，框架很容易扩展到few-shot学习的变体，如半监督或主动学习。

参考项目地址[FSL-GNN-Image](https://github.com/vgsatorras/few-shot-gnn)
[FSL-GNN-Text](https://github.com/thunlp/FewRel) 

## 二、复现精度
项目在miniImageNet和FewRel数据集下训练和测试。
 
 miniImagenet精度：

| GNN |5-way-1-shot|5-way-5shot|
| :---: | :---: | :---: |
|官方|50.3% |66.4%|
|复现|50.3% |66.5%|

 FewRel精度： 
 
| GNN |5-way-1-shot|5-way-5-shot|
| :---: | :---: | :---: |
|官方|66.2% |81.3%|
|复现|71.0% |84.4%|

## 三、数据集
Mini-Imagenet： 2016年google DeepMind团队从Imagnet数据集中抽取的一小部分（大小约3GB）制作了Mini-Imagenet数据集，共有100个类别，每个类别都有600张图片，共60000张（都是.jpg结尾的文件）。

Mini-Imagenet数据集中还包含了train.csv、val.csv以及test.csv三个文件。

train.csv包含38400张图片，共64个类别。
val.csv包含9600张图片，共16个类别。
test.csv包含12000张图片，共20个类别。
每个csv文件之间的图像以及类别都是相互独立的，即共60000张图片，100个类。

FewRel： 运行基于FewRel 1.0版本，这是第一个将few-shot学习与关系提取结合在一起的，其中模型需要处理few-shot挑战和从纯文本中提取实体关系。该基准测试提供了一个具有64个关系的训练数据集和一个具有16个关系的验证集


## 四、环境依赖
paddlepaddle-gpu==2.2.2

## 五、快速开始

本项目5-way分类可设1-shot和5-shot。如果用5-shot可设置--K 5，用1-shot可设置--K 1。下面以1-shot为例。

### step1: 加载预训练数据
解压miniImagenet相关的compacted_datasets.zip到./pretrain目录下

解压fewrel相关的fewrel_wiki.zip和glove.zip到./pretrain目录下

以上数据可以在这里[下载](https://aistudio.baidu.com/aistudio/datasetdetail/149267)


### step2: 训练

训练的模型保存在./checkpoint目录下

训练的日志保存在./logs目录下

在minimagenet上训练
```bash
python3  main.py --mode train  --exp_name minimagenet_N5_S1 --N 5 --K 1
```

在fewrel上训练
```bash
python3  main_fewrel.py --encoder cnn --N 5 --K 1
```

### step3: 测试

在minimagenet上测试
```bash
python3 main.py --mode test --exp_name minimagenet_N5_S1 --N 5 --K 1
```

在fewrel上测试
```bash
python3 main_fewrel.py --only_test True --encoder cnn --N 5 --K 1
```

可以加载./checkpoint目录下训练好的模型进行测试，也可以直接加载[最优模型](https://aistudio.baidu.com/aistudio/datasetdetail/140016) 进行测试。

## 六、代码结构与参数说明

### 6.1 代码结构

```
├─checkpoint                     # 训练保存文件
├─data                           # minimagenet加载文件
├─fewshot_re_kit                 # fewrel相关文件
├─models                         # GNN模型
├─logs                           # 训练日志
├─pretrain                       # 预训练数据  
├─utils                          # 工具   
│  main_fewrel.py                # fewrel主文件
│  README.md                     # readme
│  main.py                       # minimagenet主文件
```
### 6.2 参数说明

可以在主文件中查看设置训练与测试相关参数

## 七、模型信息

关于模型的其他信息，可以参考下表：

| 信息 | 说明 |
| --- | --- |
| 发布者 | Lieber |
| 时间 | 2022.06 |
| 框架版本 | Paddle 2.2.2 |
| 应用场景 | 小样本 |
| 支持硬件 | GPU、CPU |
| 下载链接 | [最优模型](https://aistudio.baidu.com/aistudio/datasetdetail/140016)|
| 在线运行 | [notebook](https://aistudio.baidu.com/aistudio/projectdetail/4145121)|