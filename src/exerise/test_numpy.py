import numpy as np

a_b = np.array([[0, 1, 2, 3],
                [5, 6, 7, 8],
                [10, 11, 12, 13]])


a = np.array([[0, 1, 2, 1],
              [5, 6, 7, 1],
              [10, 11, 12, 1]])

b = np.array([[0],
              [5],
              [10],
              [12]])

c = np.array([1, 2, 3])

d = np.array([1,
              2,
              3])

e = np.array([[1],
              [2],
              [3]])

f = np.array([[1, 2, 3]])

# 基本操作
print(a * a_b)
print(a @ b)  # 25 100 175
print(c @ d)  # 14
print(e @ f)  # e (3,1) f(1, 3)
print(f @ e)

# 浅复制
g = a.reshape(2, 6)
g[1, 5] = 100
print(a)

# 深复制
j = np.arange(int(1e8))
i = j[50: 100].copy()
h = j[50: 100]  # j will persist in memory even if del j is executed.
del j
print(i)

#
a = np.arange(12)**2
