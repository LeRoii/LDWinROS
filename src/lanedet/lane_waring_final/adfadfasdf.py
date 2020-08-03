import numpy as np

data = np.load('/home/iairiv/PycharmProjects/LaneNet/output/456_result.npy', allow_pickle=True)
data = list(data)
print(data)