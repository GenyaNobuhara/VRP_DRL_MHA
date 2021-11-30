from time import time
import torch
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import random

from model import AttentionModel
from data import generate_data, data_from_txt
from baseline import load_model
from config import test_parser


def get_clean_path(arr):
	"""Returns extra zeros from path.
	   Dynamical model generates duplicated zeros for several graphs when obtaining partial solutions.
	"""
	p1, p2 = 0, 1
	output = []
	while p2 < len(arr):
		if arr[p1] != arr[p2]:
			output.append(arr[p1])
			if p2 == len(arr) - 1:
				output.append(arr[p2])
		p1 += 1
		p2 += 1

	if output[0] != 0:
		output.insert(0, 0)# insert 0 in 0th of the array
	if output[-1] != 0:
		output.append(0)# insert 0 at the end of the array
	return output

def plot_route(data, pi, costs, title, idx_in_batch = 0):
	"""Plots journey of agent
	Args:
		data: dataset of graphs
		pi: (batch, decode_step) # tour
		idx_in_batch: index of graph in data to be plotted
	"""
	cost = costs[idx_in_batch].cpu().numpy()
	# Remove extra zeros
	pi_ = get_clean_path(pi[idx_in_batch].cpu().numpy())

	depot_xy = data[0][idx_in_batch].cpu().numpy()
	customer_xy = data[1][idx_in_batch].cpu().numpy()
	demands = data[2][idx_in_batch].cpu().numpy()
	readyTime = data[3][idx_in_batch].cpu().numpy()
	dueTime = data[4][idx_in_batch].cpu().numpy()
	# customer_labels = ['(' + str(i) + ', ' + str(demand) + ')' for i, demand in enumerate(demands.round(2), 1)]
	customer_labels = ['(' + str(np.round(readyTime[i],2))+ ')' for i in range(len(dueTime))]
	
	xy = np.concatenate([depot_xy.reshape(1, 2), customer_xy], axis = 0)

	# Get list with agent loops in path
	list_of_paths, cur_path = [], []
	for idx, node in enumerate(pi_):

		cur_path.append(node)

		if idx != 0 and node == 0:
			if cur_path[0] != 0:
				cur_path.insert(0, 0)
			list_of_paths.append(cur_path)
			cur_path = []

	path_traces = []
	for i, path in enumerate(list_of_paths, 1):
		coords = xy[[int(x) for x in path]]

		# Calculate length of each agent loop
		lengths = np.sqrt(np.sum(np.diff(coords, axis = 0) ** 2, axis = 1))
		total_length = np.sum(lengths)

		path_traces.append(go.Scatter(x = coords[:, 0],
									y = coords[:, 1],
									mode = 'markers+lines',
									name = f'tour{i} Length = {total_length:.3f}',
									opacity = 1.0))

	trace_points = go.Scatter(x = customer_xy[:, 0],
							  y = customer_xy[:, 1],
							  mode = 'markers+text', 
							  name = 'Customer (demand)',
							  text = customer_labels,
							  textposition = 'top center',
							  marker = dict(size = 7),
							  opacity = 1.0
							  )

	trace_depo = go.Scatter(x = [depot_xy[0]],
							y = [depot_xy[1]],
							mode = 'markers+text',
							name = 'Depot (capacity = 1.0)',
							#text = ['depot'],
							textposition = 'bottom center',
							marker = dict(size = 23),
							marker_symbol = 'triangle-up'
							)
	
	layout = go.Layout(
		#title = dict(text = f'<b>VRP{customer_xy.shape[0]} {title}, Total Length = {cost:.3f}</b>', x = 0.5, y = 1, yanchor = 'bottom', yref = 'paper', pad = dict(b = 10)),#https://community.plotly.com/t/specify-title-position/13439/3
						# xaxis = dict(title = 'X', ticks='outside'),
						# yaxis = dict(title = 'Y', ticks='outside'),#https://kamino.hatenablog.com/entry/plotly_for_report
						xaxis = dict(title = 'X', range = [-0.1, 1.1], showgrid=False, ticks='outside', linewidth=1, mirror=True),
						yaxis = dict(title = 'Y', range = [-0.1, 1.1], showgrid=False, ticks='outside', linewidth=1, mirror=True),
						showlegend = False,
						width = 750,
						height = 700,
						autosize = True,
						template = "plotly_white",
						legend = dict(x = 1, xanchor = 'right', y =0, yanchor = 'bottom', bordercolor = '#444', borderwidth = 0)
						# legend = dict(x = 0, xanchor = 'left', y =0, yanchor = 'bottom', bordercolor = '#444', borderwidth = 0)
						)

	data = [trace_points, trace_depo] + path_traces
	fig = go.Figure(data = data, layout = layout)
	fig.show()

if __name__ == '__main__':
	args = test_parser()
	t1 = time()
	device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
	pretrained = load_model(args.path, embed_dim = 128, n_customer = args.n_customer, n_encode_layers = 3)
	print(f'model loading time:{time()-t1}s')
	if(args.seed == -1):
		rand = [123,128,88,78,69,24,53,111,122,126]
		all_cos = []
		all_time_cos = []
		all_vehicle_list = []
		best_vehicle_list = []
		for j in range(100):
			data = []
			vehicle_list = []
			seed = j*10+8
			for i in range(7):
				elem = [generate_data(device, 1, args.n_customer,seed)[i].squeeze(0) for j in range(args.batch)]
				data.append(torch.stack(elem, 0))
			pretrained = pretrained.to(device)
			pretrained.eval()
			with torch.no_grad():
				costs, _, pi,time_cost = pretrained(data, return_pi = True, decode_type = args.decode_type)
			#print('costs:', costs)
			#print('time_costs:',time_cost)
			idx_in_batch = torch.argmin(costs, dim = 0)
			#print(f'decode type:{args.decode_type}\nminimum cost: {costs[idx_in_batch]:.3f} and idx: {idx_in_batch} out of {args.batch} solutions')
			#print(f'{pi[idx_in_batch]}\ninference time: {time()-t1}s')
			#plot_route(data, pi, costs, 'Pretrained', idx_in_batch)
			#print(costs[idx_in_batch])
			#print(time_cost[idx_in_batch])
			for i in range(128):
				vehicle_list.append(pi[i].tolist().count(0))
			all_vehicle_list.append(vehicle_list)
			best_vehicle_list.append(pi[idx_in_batch].tolist().count(0))
			all_cos.append(costs[idx_in_batch].item())
			all_time_cos.append(time_cost[idx_in_batch][0].item())
		print(best_vehicle_list)
		#print((np.array(all_cos)-np.array(all_time_cos)).tolist())
		#print(all_vehicle_list)

		

	else:
		if args.txt is not None:
			datatxt = data_from_txt(args.txt)
			data = []
			for i in range(7):
				elem = [datatxt[i].squeeze(0) for j in range(args.batch)]
				data.append(torch.stack(elem, 0))
				data = list(map(lambda x: x.to(device), data))
		else:
			data = []
			for i in range(7):
				elem = [generate_data(device, 1, args.n_customer, args.seed)[i].squeeze(0) for j in range(args.batch)]
				data.append(torch.stack(elem, 0))
		print(f'data generate time:{time()-t1}s')
		data[0] = torch.tensor([[0.5,0.5],
			[0.5,0.5],
			[0.5,0.5],
			[0.5,0.5],
			[0.5,0.5],
			[0.5,0.5],
			[0.5,0.5],
			[0.5,0.5],
			[0.5,0.5],
			[0.5,0.5],
			[0.5,0.5],
			[0.5,0.5],
			[0.5,0.5],
			[0.5,0.5],
			[0.5,0.5],
			[0.5,0.5],
			[0.5,0.5],
			[0.5,0.5],
			[0.5,0.5],
			[0.5,0.5],
			[0.5,0.5],
			[0.5,0.5],
			[0.5,0.5],
			[0.5,0.5],
			[0.5,0.5],
			[0.5,0.5],
			[0.5,0.5],
			[0.5,0.5],
			[0.5,0.5],
			[0.5,0.5],
			[0.5,0.5],
			[0.5,0.5],
			[0.5,0.5],
			[0.5,0.5],
			[0.5,0.5],
			[0.5,0.5],
			[0.5,0.5],
			[0.5,0.5],
			[0.5,0.5],
			[0.5,0.5],
			[0.5,0.5],
			[0.5,0.5],
			[0.5,0.5],
			[0.5,0.5],
			[0.5,0.5],
			[0.5,0.5],
			[0.5,0.5],
			[0.5,0.5],
			[0.5,0.5],
			[0.5,0.5],
			[0.5,0.5],
			[0.5,0.5],
			[0.5,0.5],
			[0.5,0.5],
			[0.5,0.5],
			[0.5,0.5],
			[0.5,0.5],
			[0.5,0.5],
			[0.5,0.5],
			[0.5,0.5],
			[0.5,0.5],
			[0.5,0.5],
			[0.5,0.5],
			[0.5,0.5],
			[0.5,0.5],
			[0.5,0.5],
			[0.5,0.5],
			[0.5,0.5],
			[0.5,0.5],
			[0.5,0.5],
			[0.5,0.5],
			[0.5,0.5],
			[0.5,0.5],
			[0.5,0.5],
			[0.5,0.5],
			[0.5,0.5],
			[0.5,0.5],
			[0.5,0.5],
			[0.5,0.5],
			[0.5,0.5],
			[0.5,0.5],
			[0.5,0.5],
			[0.5,0.5],
			[0.5,0.5],
			[0.5,0.5],
			[0.5,0.5],
			[0.5,0.5],
			[0.5,0.5],
			[0.5,0.5],
			[0.5,0.5],
			[0.5,0.5],
			[0.5,0.5],
			[0.5,0.5],
			[0.5,0.5],
			[0.5,0.5],
			[0.5,0.5],
			[0.5,0.5],
			[0.5,0.5],
			[0.5,0.5],
			[0.5,0.5],
			[0.5,0.5],
			[0.5,0.5],
			[0.5,0.5],
			[0.5,0.5],
			[0.5,0.5],
			[0.5,0.5],
			[0.5,0.5],
			[0.5,0.5],
			[0.5,0.5],
			[0.5,0.5],
			[0.5,0.5],
			[0.5,0.5],
			[0.5,0.5],
			[0.5,0.5],
			[0.5,0.5],
			[0.5,0.5],
			[0.5,0.5],
			[0.5,0.5],
			[0.5,0.5],
			[0.5,0.5],
			[0.5,0.5],
			[0.5,0.5],
			[0.5,0.5],
			[0.5,0.5],
			[0.5,0.5],
			[0.5,0.5],
			[0.5,0.5],
			[0.5,0.5]])
		'''
		b = [[0.2517, 0.6886],
			[0.0740, 0.8665],
			[0.1366, 0.1025],
			[0.1841, 0.7264],
			[0.3153, 0.6871],
			[0.0756, 0.1966],
			[0.3164, 0.4017],
			[0.1186, 0.8274],
			[0.3821, 0.6605],
			[0.8536, 0.5932],
			[0.6367, 0.9826],
			[0.2745, 0.6584],
			[0.2775, 0.8573],
			[0.8993, 0.0390],
			[0.9268, 0.7388],
			[0.7179, 0.7058],
			[0.9156, 0.4340],
			[0.0772, 0.3565],
			[0.1479, 0.5331],
			[0.4066, 0.2318],]
		'''
		b = [[0,0.5],[0.1,0.4],[0.1,0.6],[0.4,0.9],[0.6,0.9],[0.5,1],[0.9,0.6],[0.9,0.4],[1,0.5],[0.4,0.1],[0.6,0.1],[0.5,0]]
		king = []
		king2 = []
		d = [0.2667, 0.2667, 0.3000, 0.2667, 0.0667, 0.1000, 0.0333, 0.0333, 0.2667,
			0.1000, 0.2333, 0.2667, 0.0333, 0.1667, 0.1000, 0.0667, 0.1333, 0.0667,
			0.2667, 0.2000]
		#d = [0.1]*12
		#ready = [i/30 for i in range(20)]
		ready = [0,1/3,2/3,2/3,1/3,0,1/3,2/3,0,0,2/3,1/3]
		due = [ready[i]+(1/3) for i in range(12)]
		#ready.append(np.random.rand()*(2/3))
		#due.append(1)
		king3 = []
		king4 = []
		for i in range(128):
			king.append(b)
			king2.append(d)
			king3.append(ready)
			king4.append(due)
		data[1] = torch.tensor(king)
		data[2] = torch.tensor(king2)
		data[3] = torch.tensor(king3)
		data[4] = torch.tensor(king4)
		pretrained = pretrained.to(device)
		pretrained.eval()
		with torch.no_grad():
			costs, _, pi,time_cost = pretrained(data, return_pi = True, decode_type = args.decode_type)
		print('costs:', costs)
		print('time_costs:',time_cost)
		idx_in_batch = torch.argmin(costs, dim = 0)
		print(f'decode type:{args.decode_type}\nminimum cost: {costs[idx_in_batch]:.3f} and idx: {idx_in_batch} out of {args.batch} solutions')
		print(f'{pi[idx_in_batch]}\ninference time: {time()-t1}s')
		plot_route(data, pi, costs, 'Pretrained', idx_in_batch)
		print(costs[idx_in_batch])
		print(time_cost[idx_in_batch])
		
