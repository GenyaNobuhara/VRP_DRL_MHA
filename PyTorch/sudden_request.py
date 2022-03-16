from time import time
import torch
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import random

from model import AttentionModel
from data import generate_data
from baseline import load_model
from config import test_parser
from plot import get_clean_path
from utils import get_doctor_position,set_route_greedy,get_costs,set_route_queue,set_route_go_back

def plot_route(data, pi, costs, title):
	"""Plots journey of agent
	Args:
		data: dataset of graphs
		pi: (batch, decode_step) # tour
		idx_in_batch: index of graph in data to be plotted
	"""
	cost = costs[idx_in_batch].cpu().numpy()
	pi_ = pi

	depot_xy = data[0][idx_in_batch].cpu().numpy()
	customer_xy = data[1][idx_in_batch].cpu().numpy()
	demands = data[2][idx_in_batch].cpu().numpy()
	readyTime = data[3][idx_in_batch].cpu().numpy()
	dueTime = data[4][idx_in_batch].cpu().numpy()
	# customer_labels = ['(' + str(i) + ', ' + str(demand) + ')' for i, demand in enumerate(demands.round(2), 1)]
	customer_labels = ['(' + str(np.round(i+1,2))+ ')' for i in range(len(dueTime))]
	
	xy = np.concatenate([depot_xy.reshape(1, 2), customer_xy], axis = 0)

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
		title = dict(text = f'<b>{title}, Total Length = {cost:.3f}</b>', x = 0.5, y = 1, yanchor = 'bottom', yref = 'paper', pad = dict(b = 10)),
						xaxis = dict(title = 'X', range = [-0.1, 1.1], showgrid=False, ticks='outside', linewidth=1, mirror=True),
						yaxis = dict(title = 'Y', range = [-0.1, 1.1], showgrid=False, ticks='outside', linewidth=1, mirror=True),
						showlegend = False,
						width = 750,
						height = 700,
						autosize = True,
						template = "plotly_white",
						legend = dict(x = 1, xanchor = 'right', y =0, yanchor = 'bottom', bordercolor = '#444', borderwidth = 0)
						)

	data = [trace_points, trace_depo] + path_traces
	fig = go.Figure(data = data, layout = layout)
	fig.show()

if __name__ == '__main__':
    args = test_parser()
    t1 = time()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    pretrained = load_model(args.path, embed_dim = 128, n_customer = args.n_customer, n_encode_layers = 3)
    if(args.seed == -1):
        rand = [123,128,88,78,69,24,53,111,122,126]
        num_greedy = 0
        num_queue = 0
        num_backed_doctor = 0
        FT_greedy = []
        FT_queue = []
        FT_backed_doctor = []
        BT_backed_doctor = []
        for j in range(100):
            data = []
            vehicle_list = []
            seed = j*10+8
            data = []
            for i in range(7):
                elem = [generate_data(device, 1, args.n_customer, seed)[i].squeeze(0) for j in range(args.batch)]
                data.append(torch.stack(elem, 0))
            #print(f'data generate time:{time()-t1}s')
            centre = [0.5,0.5]
            b = data[1][0].cpu().tolist()
            king0 = []
            king = []
            king2 = []
            d = [0.125 for i in range(20)]
            ready = [i/30 for i in range(20)]
            due = [(i/30)+1/3 for i in range(20)]
            king3 = []
            king4 = []
            for i in range(128):
                king0.append(centre)
                king.append(b)
                king2.append(d)
                king3.append(ready)
                king4.append(due)
            data[0] = torch.tensor(king0)
            data[1] = torch.tensor(king)
            data[2] = torch.tensor(king2)
            data[3] = torch.tensor(king3)
            data[4] = torch.tensor(king4)
            pretrained = pretrained.to(device)
            pretrained.eval()
            with torch.no_grad():
                costs, _, pi,time_costs = pretrained(data, return_pi = True, decode_type = args.decode_type)
                
            idx_in_batch = torch.argmin(costs, dim = 0)
            #ここから追加
            pi_ = get_clean_path(pi[idx_in_batch].cpu().numpy())

            list_of_paths, cur_path = [], []
            for idx, node in enumerate(pi_):
                cur_path.append(node)
                if idx != 0 and node == 0:
                    if cur_path[0] != 0:
                        cur_path.insert(0, 0)
                    list_of_paths.append(cur_path)
                    cur_path = []



            #往診の患者の位置と発生時刻を取得
            fast_xy = [0.49,0.49]
            fast_time = random.random()*0.5+0.5
            depot_xy = data[0][idx_in_batch].cpu().numpy()
            customer_xy = data[1][idx_in_batch].cpu().numpy()
            demands = data[2][idx_in_batch].cpu().numpy()
            readyTime = data[3][idx_in_batch].cpu().numpy()
            readyTime = np.append(0,readyTime)
            dueTime = data[4][idx_in_batch].cpu().numpy()
            dueTime = np.append(1,dueTime)
            xy = np.concatenate([depot_xy.reshape(1, 2), customer_xy], axis = 0)

            #コストの算出
            dis_cost,time_cost,_ = get_costs(list_of_paths,xy,readyTime,dueTime,fast_time)

            doctor_position,doctor_position_idx,doctor_time = get_doctor_position(list_of_paths,xy,fast_time)
            pi_ = set_route_greedy(pi_,doctor_position,doctor_position_idx,fast_xy,21)
            b.append(fast_xy)
            king = []
            king2 = []
            d.append(0)
            ready.append(fast_time)
            due.append(1)
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
            depot_xy = data[0][idx_in_batch].cpu().numpy()
            customer_xy = data[1][idx_in_batch].cpu().numpy()
            demands = data[2][idx_in_batch].cpu().numpy()
            readyTime = data[3][idx_in_batch].cpu().numpy()
            readyTime = np.append(0,readyTime)
            dueTime = data[4][idx_in_batch].cpu().numpy()
            dueTime = np.append(1,dueTime)
            xy = np.concatenate([depot_xy.reshape(1, 2), customer_xy], axis = 0)
            list_of_paths, cur_path = [], []
            for idx, node in enumerate(pi_):
                cur_path.append(node)
                if idx != 0 and node == 0:
                    if cur_path[0] != 0:
                        cur_path.insert(0, 0)
                    list_of_paths.append(cur_path)
                    cur_path = []
            
            dis_cost_kai,time_cost_kai,FT = get_costs(list_of_paths,xy,readyTime,dueTime,fast_time)

            if(time_cost_kai > time_cost):
                pi_ = get_clean_path(pi[idx_in_batch].cpu().numpy())
                list_of_paths, cur_path = [], []
                for idx, node in enumerate(pi_):
                    cur_path.append(node)
                    if idx != 0 and node == 0:
                        if cur_path[0] != 0:
                            cur_path.insert(0, 0)
                        list_of_paths.append(cur_path)
                        cur_path = []
                pi_,backed_doctor,BT = set_route_go_back(doctor_time,pi_,21,fast_time)
                if not(backed_doctor):
                    pi_ = get_clean_path(pi[idx_in_batch].cpu().numpy())
                    pi_ = set_route_queue(list_of_paths,xy,21,fast_xy,pi_)
                    num_queue += 1
                    list_of_paths, cur_path = [], []
                    for idx, node in enumerate(pi_):
                        cur_path.append(node)
                        if idx != 0 and node == 0:
                            if cur_path[0] != 0:
                                cur_path.insert(0, 0)
                            list_of_paths.append(cur_path)
                            cur_path = []
                    dis_cost_kai,time_cost_kai,FT = get_costs(list_of_paths,xy,readyTime,dueTime,fast_time)
                    FT_queue.append(FT)
                    if(num_queue < 3):
                        #plot_route(data, pi_, costs, '後回し')
                        print("a")
                else:
                    num_backed_doctor += 1
                    list_of_paths, cur_path = [], []
                    for idx, node in enumerate(pi_):
                        cur_path.append(node)
                        if idx != 0 and node == 0:
                            if cur_path[0] != 0:
                                cur_path.insert(0, 0)
                            list_of_paths.append(cur_path)
                            cur_path = []
                    dis_cost_kai,time_cost_kai,FT = get_costs(list_of_paths,xy,readyTime,dueTime,fast_time)
                    FT_backed_doctor.append(FT)
                    BT_backed_doctor.append(FT+fast_time)
                    if(num_backed_doctor < 3):
                        #plot_route(data, pi_, costs, '再出発')
                        print("b")
            else:
                num_greedy += 1
                FT_greedy.append(FT)
                if(num_greedy < 3):
                    print("c")
                    #plot_route(data, pi_, costs, '優先')
            #plot_route(data, pi_, costs, 'Pretrained')
            #print(pi_)
        print(num_greedy)
        print(num_queue)
        print(num_backed_doctor)
        print(FT_greedy)
        print(FT_queue)
        print(FT_backed_doctor)
        print(BT_backed_doctor)