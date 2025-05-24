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
通过调用train.py进行实验，训练参数通过命令行设置。同时，可以使用封装好的launcher.py来批量进行实验，样例：
```
nohup python -u launcher.py > exp.log 2>&1 &
```
<b>注意</b>:为了节省空间，数据集和保存的模型不予存储，按照流程可以获取相应文件


## 评估结果
在评估结果之后，可以使用gen_table.py自动生成实验表格