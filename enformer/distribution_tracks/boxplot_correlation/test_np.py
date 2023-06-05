import numpy as np

x = np.empty(shape=(5,10))
# x.fill(1)
print(x.shape)
x[0].fill(1)
print(x)

np.savetxt('data.csv', x, delimiter=',')
data = np.loadtxt('data.csv', delimiter=',')
print(data.shape)
print(data)

print(x == data)