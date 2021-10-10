import torch
import torch.nn as nn

class Env():
	def __init__(self, x, node_embeddings):
		super().__init__()
		""" depot_xy: (batch, 2)
			customer_xy: (batch, n_nodes-1, 2)
			--> self.xy: (batch, n_nodes, 2), Coordinates of depot + customer nodes
			demand: (batch, n_nodes-1)
			node_embeddings: (batch, n_nodes, embed_dim)

			is_next_depot: (batch, 1), e.g., [[True], [True], ...]
			Nodes that have been visited will be marked with True.
		"""
		self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
		self.depot_xy, customer_xy, self.demand, self.customer_readyTime, self.customer_dueTime,self.depot_readyTime, self.depot_dueTime = x
		self.depot_xy, customer_xy, self.demand,  self.customer_readyTime, self.customer_dueTime, self.depot_readyTime, self.depot_dueTime= self.depot_xy.to(self.device), customer_xy.to(self.device), self.demand.to(self.device), self.customer_readyTime.to(self.device),self.customer_dueTime.to(self.device), self.depot_readyTime.to(self.device),self.depot_dueTime.to(self.device)
		self.xy = torch.cat([self.depot_xy[:,None,:], customer_xy], 1).to(self.device)
		self.readyTime = torch.cat([self.depot_readyTime, self.customer_readyTime], 1).to(self.device)
		self.dueTime = torch.cat([self.depot_dueTime, self.customer_dueTime], 1).to(self.device)
		self.node_embeddings = node_embeddings
		self.batch, self.n_nodes, self.embed_dim = node_embeddings.size()

		#次にdepotに行く必要があるかどうか（0：行く必要あり,1:必要なし）
		self.is_next_depot = torch.ones([self.batch, 1], dtype = torch.bool).to(self.device)
		#すでにおとづれた顧客
		self.visited_customer = torch.zeros((self.batch, self.n_nodes-1, 1), dtype = torch.bool).to(self.device)

	#マスクを返す、制約条件を満たさない訪問先の可能性を消す
	def get_mask_D_T(self,next_node, visited_mask, D, T):
		""" next_node: ([[0],[0],[not 0], ...], (batch, 1), dtype = torch.int32), [0] denotes going to depot
			visited_mask **includes depot**: (batch, n_nodes, 1)
			visited_mask[:,1:,:] **excludes depot**: (batch, n_nodes-1, 1)
			customer_idx **excludes depot**: (batch, 1), range[0, n_nodes-1] e.g. [[3],[0],[5],[11], ...], [0] denotes 0th customer, not depot
			self.demand **excludes depot**: (batch, n_nodes-1)
			selected_demand: (batch, 1)
			if next node is depot, do not select demand
			D: (batch, 1), D denotes "remaining vehicle capacity"
			self.capacity_over_customer **excludes depot**: (batch, n_nodes-1)
			visited_customer **excludes depot**: (batch, n_nodes-1, 1)
		 	is_next_depot: (batch, 1), e.g. [[True], [True], ...]
		 	return mask: (batch, n_nodes, 1)		
		"""
		#次のノードがdepot:true else:false
		self.is_next_depot = next_node == 0
		#次に行くノードがdepotなら、１に回復する
		D = D.masked_fill(self.is_next_depot == True, 1.0)
		T = T.masked_fill(self.is_next_depot == True, 0.0)
		#顧客がすでに訪問されているかどうかを調べる
		self.visited_customer = self.visited_customer | visited_mask[:,1:,:]
		#次におとづれるノードの顧客id
		customer_idx = torch.argmax(visited_mask[:,1:,:].type(torch.long), dim = 1)
		#その顧客の需要量
		selected_demand = torch.gather(input = self.demand, dim = 1, index = customer_idx)
		#積載量から、需要量をひく
		D = D - selected_demand * (1.0 - self.is_next_depot.float())

		#将来的に時間超過をここで規制する

		#Dを超過しているdemandを持つ顧客のid
		capacity_over_customer = self.demand > D
		#キャパオーバーの顧客をマスク
		mask_customer = capacity_over_customer[:,:,None] | self.visited_customer
		mask_depot = self.is_next_depot & (torch.sum((mask_customer == False).type(torch.long), dim = 1) > 0)

		""" mask_depot = True
			==> We cannot choose depot in the next step if 1) next destination is depot or 2) there is a node which has not been visited yet
		"""
		return torch.cat([mask_depot[:,None,:], mask_customer], dim = 1), D, T
	
	def _get_step(self,next_node, D, T):
		""" next_node **includes depot** : (batch, 1) int, range[0, n_nodes-1]
			--> one_hot: (batch, 1, n_nodes)
			node_embeddings: (batch, n_nodes, embed_dim)
			demand: (batch, n_nodes-1)
			--> if the customer node is visited, demand goes to 0 
			prev_node_embedding: (batch, 1, embed_dim)
			context: (batch, 1, embed_dim+1)
		"""
		#次におとづれるノードをone-hotに変換
		one_hot = torch.eye(self.n_nodes)[next_node]		
		visited_mask = one_hot.type(torch.bool).permute(0,2,1).to(self.device)

		mask, D,T = self.get_mask_D_T(next_node, visited_mask, D,T)
		self.demand = self.demand.masked_fill(self.visited_customer[:,:,0] == True, 0.0)
		
		prev_node_embedding = torch.gather(input = self.node_embeddings, dim = 1, index = next_node[:,:,None].repeat(1,1,self.embed_dim))
		# prev_node_embedding = torch.gather(input = self.node_embeddings, dim = 1, index = next_node[:,:,None].expand(self.batch,1,self.embed_dim))

		step_context = torch.cat([prev_node_embedding, D[:,:,None],T[:,:,None]], dim = -1)
		return mask, step_context, D, T

	def _create_t1(self):
		#t1時のマスク
		mask_t1 = self.create_mask_t1()
		#t1時のデポのコンテキスト、積載量、経過時間
		step_context_t1, D_t1, T_t1 = self.create_context_D_t1()
		return mask_t1, step_context_t1, D_t1,T_t1 #t1時に行けないところ,コンテキスト, 積載量、経過時間

	def create_mask_t1(self):
		#顧客のマスク
		mask_customer = self.visited_customer.to(self.device)
		#デポのマスク
		mask_depot = torch.ones([self.batch, 1, 1], dtype = torch.bool).to(self.device)
		return torch.cat([mask_depot, mask_customer], dim = 1)

	#t1時のコンテキスト、D（積載量）、T（経過時間）を取得
	def create_context_D_t1(self):
		#積載量
		D_t1 = torch.ones([self.batch, 1], dtype=torch.float).to(self.device)
		#経過時間
		T_t1 = torch.zeros([self.batch, 1], dtype=torch.float).to(self.device)
		#出発地点のid
		depot_idx = torch.zeros([self.batch, 1], dtype = torch.long).to(self.device)# long == int64
		#node_embeddingからdepotの部分を抽出
		depot_embedding = torch.gather(input = self.node_embeddings, dim = 1, index = depot_idx[:,:,None].repeat(1,1,self.embed_dim))# [10, 1, 128]
		# depot_embedding = torch.gather(input = self.node_embeddings, dim = 1, index = depot_idx[:,:,None].expand(self.batch,1,self.embed_dim))
		# https://medium.com/analytics-vidhya/understanding-indexing-with-pytorch-gather-33717a84ebc4

		return torch.cat([depot_embedding, D_t1[:,:,None],T_t1[:,:,None]], dim = -1), D_t1, T_t1 # [10, 1, 129]depot_embeddingに積載量を追加

	def get_log_likelihood(self, _log_p, pi):
		""" _log_p: (batch, decode_step, n_nodes)
			pi: (batch, decode_step), predicted tour
		"""
		log_p = torch.gather(input = _log_p, dim = 2, index = pi[:,:,None])
		return torch.sum(log_p.squeeze(-1), 1)

	def get_costs(self, pi):
		""" self.xy: (batch, n_nodes, 2), Coordinates of depot + customer nodes
			pi: (batch, decode_step), predicted tour
			d: (batch, decode_step, 2)
			Note: first element of pi is not depot, the first selected node in the path
		"""
		d = torch.gather(input = self.xy, dim = 1, index = pi[:,:,None].repeat(1,1,2))
		# d = torch.gather(input = self.xy, dim = 1, index = pi[:,:,None].expand(self.batch,pi.size(1),2))
		return (torch.sum((d[:, 1:] - d[:, :-1]).norm(p = 2, dim = 2), dim = 1)
				+ (d[:, 0] - self.depot_xy).norm(p = 2, dim = 1)# distance from depot to first selected node
				+ (d[:, -1] - self.depot_xy).norm(p = 2, dim = 1)# distance from last selected node (!=0 for graph with longest path) to depot
				)

	def get_cost_path(self,now_node,next_node):
		now_d = torch.gather(input = self.xy, dim = 1, index = now_node[:,:,None].repeat(1,1,2))
		next_d = torch.gather(input = self.xy, dim = 1, index = next_node[:,:,None].repeat(1,1,2))
		return (now_d[:, 0] -next_d[:, 0]).norm(p = 2, dim = 1)[:,None]/3

	def get_time_cost(self,next_node,T):
		RT = torch.gather(input = self.readyTime, dim= 1, index = next_node)
		DT = torch.gather(input = self.dueTime, dim= 1, index = next_node)
		ReadyCost = RT - T
		ReadyCost = ReadyCost.masked_fill( ReadyCost < 0, 0.0)
		DueCost = T - DT
		DueCost = DueCost.masked_fill( DueCost < 0, 0.0)
		time_cost = ReadyCost+DueCost
		return time_cost

class Sampler(nn.Module):
	""" args; logits: (batch, n_nodes)
		return; next_node: (batch, 1)
		TopKSampler <=> greedy; sample one with biggest probability
		CategoricalSampler <=> sampling; randomly sample one from possible distribution based on probability
	"""
	def __init__(self, n_samples = 1, **kwargs):
		super().__init__(**kwargs)
		self.n_samples = n_samples
		
class TopKSampler(Sampler):
	def forward(self, logits):
		return torch.topk(logits, self.n_samples, dim = 1)[1]# == torch.argmax(log_p, dim = 1).unsqueeze(-1)

class CategoricalSampler(Sampler):
	def forward(self, logits):
		return torch.multinomial(logits.exp(), self.n_samples)
