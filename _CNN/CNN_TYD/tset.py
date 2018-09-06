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

d1 = np.random.randint(1,4,(2,3))
# print(d1)
z=d1/d1.sum(axis=1,keepdims=True)
print(z)

print(z[[0,1],[2,1]])
# print(z[[0,1,0],[1,1,0]])
# [d1.shape[0],np.array([[0,1,0],[1,0,0]])]