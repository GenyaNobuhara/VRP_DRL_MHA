import torch
import torch.nn as nn

from layers import MultiHeadAttention, DotProductAttention
from data import generate_data
from decoder_utils import TopKSampler, CategoricalSampler, Env

class DecoderCell(nn.Module):
	def __init__(self, embed_dim = 128, n_heads = 8, clip = 10., **kwargs):
		super().__init__(**kwargs)
		
		self.Wk1 = nn.Linear(embed_dim, embed_dim, bias = False)
		self.Wv = nn.Linear(embed_dim, embed_dim, bias = False)
		self.Wk2 = nn.Linear(embed_dim, embed_dim, bias = False)
		self.Wq_fixed = nn.Linear(embed_dim, embed_dim, bias = False)
		self.Wout = nn.Linear(embed_dim, embed_dim, bias = False)
		self.Wq_step = nn.Linear(embed_dim+2, embed_dim, bias = False)
		
		self.MHA = MultiHeadAttention(n_heads = n_heads, embed_dim = embed_dim, need_W = False)
		self.SHA = DotProductAttention(clip = clip, return_logits = True, head_depth = embed_dim)
		# SHA ==> Single Head Attention, because this layer n_heads = 1 which means no need to spilt heads
		self.env = Env

	def compute_static(self, node_embeddings, graph_embedding):
		self.Q_fixed = self.Wq_fixed(graph_embedding[:,None,:])
		self.K1 = self.Wk1(node_embeddings)
		self.V = self.Wv(node_embeddings)
		self.K2 = self.Wk2(node_embeddings)
		
	def compute_dynamic(self, mask, step_context):
		Q_step = self.Wq_step(step_context)
		Q1 = self.Q_fixed + Q_step
		Q2 = self.MHA([Q1, self.K1, self.V], mask = mask)
		Q2 = self.Wout(Q2)
		logits = self.SHA([Q2, self.K2, None], mask = mask)
		return logits.squeeze(dim = 1)

	def forward(self, x, encoder_output, return_pi = False, decode_type = 'sampling'):
		node_embeddings, graph_embedding = encoder_output
		self.compute_static(node_embeddings, graph_embedding)
		env = Env(x, node_embeddings)
		mask, step_context, D , T= env._create_t1()

		selecter = {'greedy': TopKSampler(), 'sampling': CategoricalSampler()}.get(decode_type, None)
		log_ps, tours = [], []
		first_node = [[0]]*step_context.size()[0]
		now_node = torch.tensor(first_node).to(self.device)
		#時間コストの計算
		time_cost = torch.tensor(first_node,dtype=torch.float).to(self.device)
		#累積時間
		#CTime = torch.tensor(first_node,dtype = torch.float)
		for i in range(env.n_nodes*2):
			logits = self.compute_dynamic(mask, step_context)
			log_p = torch.log_softmax(logits, dim = -1)
			next_node = selecter(log_p)
			#距離を計算
			required_time = env.get_cost_path(now_node,next_node) #[batchsize]
			T = T + required_time
			time_cost += env.get_time_cost(next_node,T)
			T += 1/12
			mask, step_context, D, T = env._get_step(next_node, D, T)
			tours.append(next_node.squeeze(1))
			log_ps.append(log_p)
			now_node = next_node
			if env.visited_customer.all():
				break

		pi = torch.stack(tours, 1)
		cost = env.get_costs(pi)
		cost += time_cost.view(-1)
		ll = env.get_log_likelihood(torch.stack(log_ps, 1), pi)
		
		if return_pi:
			return cost, ll, pi, time_cost
		return cost, ll, time_cost

if __name__ == '__main__':
	batch, n_nodes, embed_dim = 1, 11, 128
	data = generate_data('cpu',n_samples = batch, n_customer = n_nodes-1)
	decoder = DecoderCell(embed_dim, n_heads = 8, clip = 10.)
	node_embeddings = torch.rand((batch, n_nodes, embed_dim), dtype = torch.float)
	graph_embedding = torch.rand((batch, embed_dim), dtype = torch.float)
	encoder_output = (node_embeddings, graph_embedding)
	# a = graph_embedding[:,None,:].expand(batch, 7, embed_dim)
	# a = graph_embedding[:,None,:].repeat(1, 7, 1)
	# print(a.size())

	decoder.train()
	cost, ll, pi, time_cost = decoder(data, encoder_output, return_pi = True, decode_type = 'sampling')
	print('\ndata: ',data)
	print('\ncost: ', cost.size(), cost)
	print('\nll: ', ll.size(), ll)
	print('\npi: ', pi.size(), pi)
	print('\ntimecost:',time_cost.size(),time_cost)

	# ll.mean().backward()
	# print(decoder.Wk1.weight.grad)
	# https://discuss.pytorch.org/t/model-param-grad-is-none-how-to-debug/52634	