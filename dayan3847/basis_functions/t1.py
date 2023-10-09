import numpy as np

d = np.column_stack([np.array([1, 2])])
d = np.row_stack([np.array([1, 2])])
print(d)
print(d.shape)

np.column_stack(np.zeros(2))