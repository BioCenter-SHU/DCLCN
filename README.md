# 数据集架构

数据集总共有三个文件夹，分别为renji、luodian、huashan。两个分类，high代表高风险（低-混合回声），low代表低风险（高回声）。

renji

|---- high

|---- low

luodian

|---- high

|---- low

huashan

|---- high

|---- low

# 运行环境

- python=3.8.5
- pytorch==1.12.1
- torchvision==0.13.1
- torchaudio==0.12.1
- cudatoolkit=11.3
- numpy==1.22.3
- torchmetrics==1.0.1
- matplotlib==3.5.2
- scikit-learn==1.1.1

详情请见文件 `环境安装指令.txt`


# WUCF

![image.png](Read%20Me%20914d5639974d425388a226b8a631ae29/image%201.png)

分类损失cls：源中心1、2和他们的真实标签的损失计算。

领域损失dm：就是要训练领域判别专家(分类器），会给到两个源中心一个特定的标签（0，1)，来判断送过来的数据是属于哪个源中心。

mmd损失：分成两个来计算（源中心1的数据和目标数据）（源中心2的数据和目标数据）计算mmd损失。

### 文件：

`data_loader.py` 用于数据加载，包含了一些预处理。

[`mmd.py](http://mmd.py)` 计算mmd距离，输入为源数据和目标数据的张量（**`def** guassian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None)`）

[`resnet.py](http://resnet.py)` 网络模型，其中clasMADA为主要class，其中前向方法中有两个if函数。

是否训练？

是：以mark字段进行区分源中心来自哪里。

源中心1进行训练

源中心2进行训练。

否：

则为测试部分，输出领域判别专家的加权预测和两个分类器的预测结果。

[`WUCF.py`](http://WUCF.py) 主要文件

[`WUCFwog.py](http://WUCFwog.py)` 没有领域判别专家的部分。

## 运行

`python wucf.py`

输出：

`record_file+source1_name+'*'+source2_name+'*'+target_name+'_'+'acc.txt'`

### 修改参数：

#wucf.py中：

```python
# Training settings
batch_size = 8
iteration = 4000
lr = [0.001, 0.01]
momentum = 0.9
cuda = True
seed = 8
log_interval = 10
l2_decay = 5e-4
root_path = "/storage1/21721505/data/"
record_file = root_path
source1_name = "renji"
source2_name = 'luodian'
target_name = "huashan_onlyl"
```

### 修改损失函数参数：

#wucf.py

```python
gamma = 2 / (1 + math.exp(-10 * (i) / (iteration) )) - 1
loss = cls_loss + gamma * (mmd_loss + l1_loss)
```

### 消融实验

取消了领域判别专家，运行#WUCFwog.py
