import numpy as np
import torch
filename = performance_recordbaseline_net.npy
M = np.load(filename).item()

for item in M['train_acc']:
	item = float(item.to(torch.device("cpu")))
for item in M['test_acc']:
	item = float(item.to(torch.device("cpu")))
np.save(M, filename)
