## 一个自用练手的CNN+RNN音频分类

>小白用来练手的，有很多不懂的东西，求看不顺眼的大佬们轻喷，如果能给建议就实在感激不尽！

测试数据集来自<a href="https://www.kaggle.com/datasets/deepcontractor/musical-instrument-chord-classification">kaggle</a>

### 虚拟环境相关
#### 基本环境
python 3.10(使用较低的版本会使某些注释非法) \
matplotlib 

#### 深度学习
PyTorch 2.3.0 \
torchaudio 2.3.0
#### 音频处理
pysoundfile(配合torchaudio) \
python3.8下使用librosa（相同步骤在服务器上无效，原因未知）
### 1.目前进度
- 可以进行训练了
- 还在补全markdown文件和调试用的ipynb文件
### 2.目前目标
- 处理过拟合问题和构建更简单的模型
- 更换更复杂的数据集
