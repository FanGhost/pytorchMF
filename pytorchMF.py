import torch
import pandas as pd 
import numpy as np
from torch.autograd import Variable
from numpy.random import randint
from sklearn import metrics
trainData = pd.read_csv('ml100k.train.rating',header=None,names=['user','item','rate'],sep='\t')
testData = pd.read_csv('ml100k.test.rating',header=None,names=['user','item','rate'],sep='\t')

userIdx = trainData.user.values
itemIdx = trainData.item.values
rates = trainData.rate.values

K=20
lambd = 0.00001
learning_rate =1e-6
U = Variable(torch.randn([len(set(userIdx)),K]), requires_grad=True)
P = Variable(torch.randn([len(set(itemIdx)),K]), requires_grad=True)
R = torch.mm(U,P.t())
ratesPred = torch.gather(R.view(1,-1)[0],0,Variable(torch.LongTensor (userIdx * len(set(itemIdx)) + itemIdx)))
diff_op = ratesPred - Variable(torch.FloatTensor(rates))
baseLoss = diff_op.pow(2).sum()

#regularizer = lambd* (U.abs().sum()+P.abs().sum())
regularizer = lambd* (U.pow(2).sum()+P.pow(2).sum())
loss = baseLoss + regularizer 

#optimizer = torch.optim.Adam([U,P], lr = learning_rate)
optimizer = torch.optim.SGD([U,P], lr = learning_rate,momentum = 0.9)
print ('Training')
for i in range(250):
    loss.backward()
    optimizer.step()
    R = torch.mm(U,P.t())
    if i %50 ==0:
        print 'loss:',loss.data.numpy()[0]
    ratesPred = torch.gather(R.view(1,-1)[0],0,Variable(torch.LongTensor (userIdx * len(set(itemIdx)) + itemIdx)))
    diff_op = ratesPred - Variable(torch.FloatTensor(rates))
    baseLoss = diff_op.pow(2).mean()#torch.abs()
    #baseLoss = torch.sum(diff_abs)
    #regularizer = lambd* (U.abs().sum()+P.abs().sum())
    regularizer = lambd* (U.pow(2).sum()+P.pow(2).sum())
    loss = baseLoss + regularizer

def getMAE():
    userIdx = testData.user.values
    itemIdx = testData.item.values
    rates = testData.rate.values
    R = torch.mm(U,P.t())
    ratesPred = torch.gather(R.view(1,-1)[0],0,Variable(torch.LongTensor (userIdx * len(set(itemIdx)) + itemIdx)))
    diff_op = ratesPred - Variable(torch.FloatTensor(rates))
    MAE = diff_op.abs().mean()
    return MAE.data.numpy()[0]

print ('testing')
print'MAE:',getMAE()