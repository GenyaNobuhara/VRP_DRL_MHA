import torch
import numpy as np
import os
from tqdm import tqdm
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader

CAPACITIES = {10: 20.,12:20, 20: 30.,21:30., 50: 30., 100: 50.,30:33,40:30}
def generate_data(device, n_samples = 10, n_customer = 20, seed = None):
	"""
		x[0] -- depot_xy: (batch, 2)
		x[1] -- customer_xy: (batch, n_nodes-1, 2)
		x[2] -- demand: (batch, n_nodes-1)
		x[3] -- customer_readyTime: (batch, n_node-1)
		x[4] -- customer_dueTime: (batch, n_node-1)
		x[5] -- depot_readyTime (batch,1)
		x[6] -- depot_dueTime(batch,1)
	"""
	if seed is not None:
		torch.manual_seed(seed)

	customer_readyTime = torch.rand((n_samples, n_customer), device = device)*(2/3)
	customer_dueTime = customer_readyTime+(1/3)
	depot_readyTime = torch.zeros((n_samples,1),dtype=torch.float,device = device)
	depot_dueTime = torch.ones((n_samples,1),device = device)
	return (torch.rand((n_samples, 2), device = device),
			torch.rand((n_samples, n_customer, 2), device = device),
			(torch.randint(size = (n_samples, n_customer), low = 1, high = 10, device = device) / CAPACITIES[n_customer]),
			customer_readyTime,
			customer_dueTime,
			depot_readyTime,
			depot_dueTime
			)

class Generator(Dataset):
	def __init__(self, device, n_samples = 5120, n_customer = 20, seed = None):
		self.tuple = generate_data(device, n_samples, n_customer)

	def __getitem__(self, idx):
		return (self.tuple[0][idx], self.tuple[1][idx], self.tuple[2][idx],self.tuple[3][idx],self.tuple[4][idx],self.tuple[5][idx],self.tuple[6][idx])

	def __len__(self):
		return self.tuple[0].size(0)


if __name__ == '__main__':
	device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
	print('device-->', device)
	
	data = generate_data(device, n_samples = 128, n_customer = 20, seed = 123)
	for i in range(len(data)):
		print(data[i].dtype)
		print(data[i].size())
	
	
	batch, batch_steps, n_customer = 128, 100000, 20
	dataset = Generator(device, n_samples = batch*batch_steps, n_customer = n_customer)
	data = next(iter(dataset))	
	
	dataloader = DataLoader(dataset, batch_size = 128, shuffle = True)
	print('use datalodaer ...')
	for i, data in enumerate(dataloader):
		for j in range(len(data)):
			print(data[j].dtype)
			print(data[j].size())	
		if i == 0:
			break
