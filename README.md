
# 通过pytorch 实现一个矩阵分解
矩阵分解的原理可以参考http://hameddaily.blogspot.com/2016/12/simple-matrix-factorization-with.html

这里通过Pytorch进行一个简单的实现

代码是python2的



```python
import torch
import pandas as pd 
import numpy as np
from torch.autograd import Variable
from numpy.random import randint
from sklearn import metrics
```

读入训练集和测试集，这里用的movielens 100k的数据集，数据我清洗过了，清洗后的数据格式如下所示


```python
trainData = pd.read_csv('ml100k.train.rating',header=None,names=['user','item','rate'],sep='\t')
testData = pd.read_csv('ml100k.test.rating',header=None,names=['user','item','rate'],sep='\t')

userIdx = trainData.user.values
itemIdx = trainData.item.values
rates = trainData.rate.values
```


```python
trainData.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>user</th>
      <th>item</th>
      <th>rate</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>195</td>
      <td>241</td>
      <td>3</td>
    </tr>
    <tr>
      <th>1</th>
      <td>185</td>
      <td>301</td>
      <td>3</td>
    </tr>
    <tr>
      <th>2</th>
      <td>21</td>
      <td>376</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>243</td>
      <td>50</td>
      <td>2</td>
    </tr>
    <tr>
      <th>4</th>
      <td>165</td>
      <td>345</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



设置初始参数，这里K为factor的长度取20，正则项参数lambd 取0.00001 学习率1e-6


```python
K=20
lambd = 0.00001
learning_rate =1e-6
```

初始化 矩阵$U$和$P$


```python
U = Variable(torch.randn([len(set(userIdx)),K]), requires_grad=True)
P = Variable(torch.randn([len(set(itemIdx)),K]), requires_grad=True)
U
```




    Variable containing:
    -6.8339e-02  6.7907e-01 -1.8883e+00  ...  -8.2526e-01  1.4129e+00 -9.9104e-01
    -1.0582e+00  1.1918e+00 -7.3917e-01  ...  -1.8044e+00 -3.6639e-01 -5.5533e-01
     9.3784e-01 -6.5192e-01 -2.7269e-01  ...  -2.6185e-01  1.0857e+00  4.8078e-01
                    ...                   ⋱                   ...                
    -3.5025e-01  9.3231e-01  7.4405e-01  ...  -8.9989e-02 -5.9873e-01 -1.7450e-01
     3.8695e-01 -1.6384e+00 -3.0816e-01  ...   3.3690e-01  2.2259e+00  5.1559e-01
    -1.6188e-02  7.7720e-01  5.5722e-01  ...   1.5586e+00  1.6987e+00  1.0659e+00
    [torch.FloatTensor of size 943x20]



计算矩阵$U$和$P$的乘积，gather函数取得了训练集所对应的数据，然后计算loss


```python
R = torch.mm(U,P.t())
ratesPred = torch.gather(R.view(1,-1)[0],0,Variable(torch.LongTensor (userIdx * len(set(itemIdx)) + itemIdx)))
diff_op = ratesPred - Variable(torch.FloatTensor(rates))
baseLoss = diff_op.pow(2).sum()


```

计算正则项，这里采用的是L2的正则，注释中的为L1正则


```python
#regularizer = lambd* (U.abs().sum()+P.abs().sum())
regularizer = lambd* (U.pow(2).sum()+P.pow(2).sum())
loss = baseLoss + regularizer   
```

选择优化方法，这里使用了随机梯度下降，注释中的为Adam


```python
#optimizer = torch.optim.Adam([U,P], lr = learning_rate)
optimizer = torch.optim.SGD([U,P], lr = learning_rate,momentum = 0.9)
```

算法迭代迭代250次，每隔50次打印一次当前的loss值


```python
for i in range(250):
    loss.backward()
    optimizer.step()
    R = torch.mm(U,P.t())
    if i %50 ==0:
        print (loss.data.numpy()[0])
    ratesPred = torch.gather(R.view(1,-1)[0],0,Variable(torch.LongTensor (userIdx * len(set(itemIdx)) + itemIdx)))
    diff_op = ratesPred - Variable(torch.FloatTensor(rates))
    baseLoss = diff_op.pow(2).mean()#torch.abs()
    #baseLoss = torch.sum(diff_abs)
    #regularizer = lambd* (U.abs().sum()+P.abs().sum())
    regularizer = lambd* (U.pow(2).sum()+P.pow(2).sum())
    loss = baseLoss + regularizer
```

    3.40576e+06
    23.4698
    17.0843
    15.6253
    17.085


测试算法效果，使用MAE


```python
def getMAE():
    userIdx = testData.user.values
    itemIdx = testData.item.values
    rates = testData.rate.values
    R = torch.mm(U,P.t())
    ratesPred = torch.gather(R.view(1,-1)[0],0,Variable(torch.LongTensor (userIdx * len(set(itemIdx)) + itemIdx)))
    diff_op = ratesPred - Variable(torch.FloatTensor(rates))
    MAE = diff_op.abs().mean()
    return MAE.data.numpy()[0]
getMAE()
```




    4.0511918



由于这边数据集是implicit feedback 的，所以结果不是很好，后面我们再来讨论如何解决这一问题
