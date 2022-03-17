import torch
import numpy as np
import os
from tqdm import tqdm
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import torch.nn.functional as F

CAPACITIES = {10: 20.,12:20, 20: 30.,21:30., 50: 30., 100: 50.,30:33,40:30}
def generate_data(device, n_samples = 10, n_customer = 20, seed = None):
	"""
		x[0] -- depot_xy: (batch,8,2)
		x[1] -- customer_xy: (batch, n_nodes-1, 2)
		x[2] -- demand: (batch, n_nodes-1)
		x[3] -- customer_readyTime: (batch, n_node-1)
		x[4] -- customer_dueTime: (batch, n_node-1)
		x[5] -- depot_readyTime (batch,2)
		x[6] -- depot_dueTime(batch,2)
		x[7] -- depot_gender(batch,8,2)
		x[8] -- customer_gender(batch,n_nodes -1 ,2)
		x[9] -- depot_department(batch, 8 ,4)
		x[10] -- customer_department(batch, n_nodes -1 ,4)
	"""
	if seed is not None:
		torch.manual_seed(seed)


	depot_xy = torch.ones((n_samples,8,2), device = device)*(1/2)
	customer_readyTime = torch.rand((n_samples, n_customer), device = device)*(2/3)
	customer_dueTime = customer_readyTime+(1/3)
	depot_readyTime = torch.zeros((n_samples,8),dtype=torch.float,device = device)
	depot_dueTime = torch.ones((n_samples,8),device = device)
	customer_man = torch.randint(size = (n_samples,n_customer,1), low = 0, high = 2,device = device)
	customer_women =1- customer_man
	customer_gender = torch.cat([customer_man,customer_women],axis=2)
	tmp_gender = torch.tensor([[1,0],[0,1],[1,0],[0,1],[1,0],[0,1],[1,0],[0,1]],device = device)
	depot_gender = torch.zeros(n_samples,8,2,dtype=torch.float,device = device)
	depot_gender += tmp_gender
	tmp_department = torch.tensor([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],device = device)
	depot_department = torch.zeros(n_samples,8,4,dtype=torch.float,device = device)
	depot_department += tmp_department
	customer_department = F.one_hot(torch.randint(size = (1,n_samples*n_customer), low = 0, high = 4,device = device).view(n_samples,n_customer) % 4)
	return (depot_xy,
			torch.rand((n_samples, n_customer, 2), device = device),
			(torch.randint(size = (n_samples, n_customer), low = 1, high = 10, device = device) / CAPACITIES[n_customer]),
			customer_readyTime,
			customer_dueTime,
			depot_readyTime,
			depot_dueTime,
			depot_gender,
			customer_gender,
			depot_department,
			customer_department
			)

class Generator(Dataset):
	def __init__(self, device, n_samples = 5120, n_customer = 20, seed = None):
		self.tuple = generate_data(device, n_samples, n_customer)

	def __getitem__(self, idx):
		return (self.tuple[0][idx], self.tuple[1][idx], self.tuple[2][idx],self.tuple[3][idx],self.tuple[4][idx],self.tuple[5][idx],self.tuple[6][idx],self.tuple[7][idx],self.tuple[8][idx],self.tuple[9][idx],self.tuple[10][idx])

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
			print(data[j].dtype)# torch.float32
			print(data[j].size())	
		if i == 0:
			break

