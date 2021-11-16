from time import time
import torch
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go

from model import AttentionModel
from data import generate_data, data_from_txt
from baseline import load_model
from config import test_parser

if __name__ == '__main__':
    args = test_parser()
    t1 = time()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    pretrained = load_model(args.path, embed_dim = 128, n_customer = args.n_customer, n_encode_layers = 3)
    print(f'model loading time:{time()-t1}s')
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
    data[0] = torch.tensor([[0.2961, 0.5166],
        [0.2961, 0.5166],
        [0.2961, 0.5166],
        [0.2961, 0.5166],
        [0.2961, 0.5166],
        [0.2961, 0.5166],
        [0.2961, 0.5166],
        [0.2961, 0.5166],
        [0.2961, 0.5166],
        [0.2961, 0.5166],
        [0.2961, 0.5166],
        [0.2961, 0.5166],
        [0.2961, 0.5166],
        [0.2961, 0.5166],
        [0.2961, 0.5166],
        [0.2961, 0.5166],
        [0.2961, 0.5166],
        [0.2961, 0.5166],
        [0.2961, 0.5166],
        [0.2961, 0.5166],
        [0.2961, 0.5166],
        [0.2961, 0.5166],
        [0.2961, 0.5166],
        [0.2961, 0.5166],
        [0.2961, 0.5166],
        [0.2961, 0.5166],
        [0.2961, 0.5166],
        [0.2961, 0.5166],
        [0.2961, 0.5166],
        [0.2961, 0.5166],
        [0.2961, 0.5166],
        [0.2961, 0.5166],
        [0.2961, 0.5166],
        [0.2961, 0.5166],
        [0.2961, 0.5166],
        [0.2961, 0.5166],
        [0.2961, 0.5166],
        [0.2961, 0.5166],
        [0.2961, 0.5166],
        [0.2961, 0.5166],
        [0.2961, 0.5166],
        [0.2961, 0.5166],
        [0.2961, 0.5166],
        [0.2961, 0.5166],
        [0.2961, 0.5166],
        [0.2961, 0.5166],
        [0.2961, 0.5166],
        [0.2961, 0.5166],
        [0.2961, 0.5166],
        [0.2961, 0.5166],
        [0.2961, 0.5166],
        [0.2961, 0.5166],
        [0.2961, 0.5166],
        [0.2961, 0.5166],
        [0.2961, 0.5166],
        [0.2961, 0.5166],
        [0.2961, 0.5166],
        [0.2961, 0.5166],
        [0.2961, 0.5166],
        [0.2961, 0.5166],
        [0.2961, 0.5166],
        [0.2961, 0.5166],
        [0.2961, 0.5166],
        [0.2961, 0.5166],
        [0.2961, 0.5166],
        [0.2961, 0.5166],
        [0.2961, 0.5166],
        [0.2961, 0.5166],
        [0.2961, 0.5166],
        [0.2961, 0.5166],
        [0.2961, 0.5166],
        [0.2961, 0.5166],
        [0.2961, 0.5166],
        [0.2961, 0.5166],
        [0.2961, 0.5166],
        [0.2961, 0.5166],
        [0.2961, 0.5166],
        [0.2961, 0.5166],
        [0.2961, 0.5166],
        [0.2961, 0.5166],
        [0.2961, 0.5166],
        [0.2961, 0.5166],
        [0.2961, 0.5166],
        [0.2961, 0.5166],
        [0.2961, 0.5166],
        [0.2961, 0.5166],
        [0.2961, 0.5166],
        [0.2961, 0.5166],
        [0.2961, 0.5166],
        [0.2961, 0.5166],
        [0.2961, 0.5166],
        [0.2961, 0.5166],
        [0.2961, 0.5166],
        [0.2961, 0.5166],
        [0.2961, 0.5166],
        [0.2961, 0.5166],
        [0.2961, 0.5166],
        [0.2961, 0.5166],
        [0.2961, 0.5166],
        [0.2961, 0.5166],
        [0.2961, 0.5166],
        [0.2961, 0.5166],
        [0.2961, 0.5166],
        [0.2961, 0.5166],
        [0.2961, 0.5166],
        [0.2961, 0.5166],
        [0.2961, 0.5166],
        [0.2961, 0.5166],
        [0.2961, 0.5166],
        [0.2961, 0.5166],
        [0.2961, 0.5166],
        [0.2961, 0.5166],
        [0.2961, 0.5166],
        [0.2961, 0.5166],
        [0.2961, 0.5166],
        [0.2961, 0.5166],
        [0.2961, 0.5166],
        [0.2961, 0.5166],
        [0.2961, 0.5166],
        [0.2961, 0.5166],
        [0.2961, 0.5166],
        [0.2961, 0.5166],
        [0.2961, 0.5166],
        [0.2961, 0.5166],
        [0.2961, 0.5166],
        [0.2961, 0.5166],
        [0.2961, 0.5166],
        [0.2961, 0.5166]])
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
        [0.4066, 0.2318]]
    king = []
    king2 = []
    d = [0.2667, 0.2667, 0.3000, 0.2667, 0.0667, 0.1000, 0.0333, 0.0333, 0.2667,
        0.1000, 0.2333, 0.2667, 0.0333, 0.1667, 0.1000, 0.0667, 0.1333, 0.0667,
        0.2667, 0.2000]
    ready = [i/30 for i in range(20)]
    due = [(i/30)+1/3 for i in range(20)]
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
        
    #print('costs:', costs)
    #print('time_costs')
    idx_in_batch = torch.argmin(costs, dim = 0)
    print(f'decode type:{args.decode_type}\nminimum cost: {costs[idx_in_batch]:.3f} and idx: {idx_in_batch} out of {args.batch} solutions')
    print(f'{pi[idx_in_batch]}\ninference time: {time()-t1}s')
    #ここから追加

    sudden_time = args.sudden_time
    print(pi[idx_in_batch])
    print(costs[idx_in_batch])
    print(time_cost[idx_in_batch])