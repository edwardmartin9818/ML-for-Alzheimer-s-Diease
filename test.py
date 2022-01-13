import numpy as np

t1 = np.asarray([1,2,3,4,5])
t2 = np.asarray([2,3,4,5,6])

print(t1)
print(t2)

t3 = np.asarray([t1,t2])
print(t3)

print(np.mean(t3))
print(np.mean(t3, axis=0))
print(np.mean(t3, axis=1))
