## 一个自用练手的CNN+RNN音频分类

>小白用来练手的，有很多不懂的东西，求看不顺眼的大佬们轻喷，如果能给建议就实在感激不尽！

测试数据集来自<a href="https://www.kaggle.com/datasets/deepcontractor/musical-instrument-chord-classification">kaggle</a>

### 1.目前进度
- 训练脚本可用
- [!]数据预处理能跑，但是用的是最笨的填0法，且指定音频长度
- network.py 可用
- [!]还在补全markdown文件和调试用的ipynb文件
### 2.目前目标
- [!]修改Dataset类使标签值可用
- 修改神经网络类，使其能根据参数自动生成结构

### 3.总体目标
- 修改Dataloader类，使其能根据参数返回对应类型 
- 修改前向传播方法，使其能根据特征种类和对应权重进行计算，大概会用解包实现
- 优化预处理方法，代替填0法和只使用最长长度