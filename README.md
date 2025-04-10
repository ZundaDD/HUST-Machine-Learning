# 衣服图像分类 —— Fashion-MNIST

## 任务流程
<ol>
<li>准备数据集</li>
<li>配置实验参数</li>
<li>训练模型</li>
<li>评估结果</li>
</ol>

## 数据准备
从[此处](https://github.com/zalandoresearch/fashion-mnist)下载数据集的压缩形式，放置在当前目录的dataset文件夹下，利用unzip.py进行解压缩，得到ImageFolder标准的结构

## 进行实验
通过调用train.py进行实验，训练参数通过命令行设置。其中editable的参数表示进行实验一般改变的参数，default参数不需要改变，environment变量根据目录需求而该变，样例：
```
nohup python -u train.py --seed 0 --model resnet18 > exp0.log 2>&1 &
```
<b>注意</b>:为了节省空间，数据集和保存的模型不予存储，按照流程可以获取相应文件

## 分组实验

| code | best_acc | model    | augmentation |
|------|----------|----------|--------------|
| 0    | 91.97%   | ResNet18 | False        |
| 1    | 91.76%   | ResNet18 | True         |
| 2    | 92.13%   | ResNet34 | False        |
| 3    | 93.05%   | VGG11_BN | False        |
| 4    | 92.17%   | ResNet50 | False        |

## 评估结果
利用test.py来对结果进行评测，结果默认保存在result文件夹下，包括每5个epoch记录的模型和测试集正确率最高模型
样例:
```
python test.py --seed 0 --model resnet18
```
在评估结果之后，可以使用drawLine.py绘制曲线图，修改exp字典来选择目标实验，只有执行过test.py的实验才能绘制图线