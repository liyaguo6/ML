import numpy as np
# arr1=np.array([[1,4],[2,3]])
# arr2=np.random.randint(1,4,(2,2))
# print(arr2)

arr1=np.array(
        [[[-1, 1, 0],
          [0, 1, 0],
          [0, 1, 1]],

         [[-1, -1, 0],
          [0, 0, 0],
          [0, -1, 0]],

         [[0, 0, -1],
          [0, 1, 0],
          [1, -1, -1]]])
# arr3 = map(lambda i: np.rot90(i, 2),arr1)

#三维数组翻转180
c=map(lambda i:np.rot90(i,2),arr1)
# print(list(c))

# n= np.array(list(c))
# print(n)
z=list(c)

# for i in z:
#     print(i)


n= np.array(z)
print(n)
print(n[1])
# print(arr2*arr1)