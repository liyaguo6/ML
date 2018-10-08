# dic = {'k1':1,'k2':2}
# for i,j in dic.items():
#     print(i,j)
import numpy as np
# #
# d=np.random.randint(1,5,(2,3,4,4))
# print(d)
# # x_padded = np.pad(d, ((0,0),(0,0),(2, 2),(2, 2)), mode='constant')
# # print(x_padded)
#
# N=d.shape[0]
# print(N)
# x_row = d.reshape(N,-1)
# print(x_row)

# d1 = np.random.randint(1,4,(2,3))
# # print(d1)
# z=d1/d1.sum(axis=1,keepdims=True)
# print(z)

# print(z[[0,1],[2,1]])
# print(z[[0,1,0],[1,1,0]])
# [d1.shape[0],np.array([[0,1,0],[1,0,0]])]

# X= np.random.randint(1,5,(2,75))
#
# # X = X.reshape(1, 3, 5, 5)
# Y = X.reshape(2, 3, 5, 5).transpose(0,2,3,1)
# print(Y)
# l=[]
# l.append(Y)
# # print(X)
#
# print(l)
# print(np.concatenate(l))

# kwargs={'update_rule':'sgd_momentum','optim_config':{'learning_rate': 5e-4, 'momentum': 0.9}}
# kwargs={'update_rule':'sgd_momentum',}
# args = kwargs.pop('update_rule','seg')
# args1 = kwargs.pop('optim_config', {})
# print(args1)

from CNN_TYD import optim
# update_rule = getattr(optim,'sgd_momentum')
# update_rule = setattr(optim,'sgd_momentum')
# print(update_rule)
# d1={}
# d={'learning_rate': 5e-4, 'momentum': 0.9}
#
# h={k: v for k, v in d.items()}
# print(h)


# mask = np.random.choice(500, 4)
# print(mask)


# t= {'learning_rate': 1e-2,'momentum': 0.9}
# t.setdefault('learning_rate1',0.3)
# print(t)

# t=[[4,2],[3,4]]
# print(t)
# d=[4,3,2,1]
# t2=np.hstack(t)
# print(np.mean(t2==d))

# te = np.random.randint(3,5,(3,2,4))
# print(te)
# print(te.reshape(3,-1))