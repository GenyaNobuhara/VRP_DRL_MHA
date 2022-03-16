import torch
import torch.nn as nn

from layers import MultiHeadAttention
from data import generate_data
import math

class Normalization(nn.Module):

	def __init__(self, embed_dim, normalization = 'batch'):
		super().__init__()

		normalizer_class = {
			'batch': nn.BatchNorm1d,
			'instance': nn.InstanceNorm1d}.get(normalization, None)
		self.normalizer = normalizer_class(embed_dim, affine=True)

	def forward(self, x):

		if isinstance(self.normalizer, nn.BatchNorm1d):
			return self.normalizer(x.view(-1, x.size(-1))).view(*x.size())
		
		elif isinstance(self.normalizer, nn.InstanceNorm1d):
			return self.normalizer(x.permute(0, 2, 1)).permute(0, 2, 1)
		else:
			assert self.normalizer is None, "Unknown normalizer type"
			return x


class ResidualBlock_BN(nn.Module):
	def __init__(self, MHA, BN, **kwargs):
		super().__init__(**kwargs)
		self.MHA = MHA
		self.BN = BN

	def forward(self, x, mask = None):
		if mask is None:
			return self.BN(x + self.MHA(x))
		return self.BN(x + self.MHA(x, mask))

class SelfAttention(nn.Module):
	def __init__(self, MHA, **kwargs):
		super().__init__(**kwargs)
		self.MHA = MHA

	def forward(self, x, mask = None):
		return self.MHA([x, x, x], mask = mask)

class EncoderLayer(nn.Module):
	# nn.Sequential):
	def __init__(self, n_heads = 8, FF_hidden = 512, embed_dim = 128, **kwargs):
		super().__init__(**kwargs)
		self.n_heads = n_heads
		self.FF_hidden = FF_hidden
		self.BN1 = Normalization(embed_dim, normalization = 'batch')
		self.BN2 = Normalization(embed_dim, normalization = 'batch')

		self.MHA_sublayer = ResidualBlock_BN(
				SelfAttention(
					MultiHeadAttention(n_heads = self.n_heads, embed_dim = embed_dim, need_W = True)
				),
			self.BN1
			)

		self.FF_sublayer = ResidualBlock_BN(
			nn.Sequential(
					nn.Linear(embed_dim, FF_hidden, bias = True),
					nn.ReLU(),
					nn.Linear(FF_hidden, embed_dim, bias = True)
			),
			self.BN2
		)
		
	def forward(self, x, mask=None):
		"""	arg x: (batch, n_nodes, embed_dim)
			return: (batch, n_nodes, embed_dim)
		"""
		return self.FF_sublayer(self.MHA_sublayer(x, mask = mask))
		
class GraphAttentionEncoder(nn.Module):
	def __init__(self, embed_dim = 128, n_heads = 8, n_layers = 3, FF_hidden = 512):
		super().__init__()
		self.init_W_depot = torch.nn.Linear(2, embed_dim, bias = True)
		self.init_W = torch.nn.Linear(7, embed_dim, bias = True)
		self.encoder_layers = nn.ModuleList([EncoderLayer(n_heads, FF_hidden, embed_dim) for _ in range(n_layers)])
	
	def forward(self, x, mask = None):
		""" x[0] -- depot_xy: (batch, 2,2)
		x[1] -- customer_xy: (batch, n_nodes-1, 2)
		x[2] -- demand: (batch, n_nodes-1)
		x[3] -- customer_ReadyTime: (batch, n_node-1)
		x[4] -- customer_DueTime: (batch, n_node)
			--> concated_customer_feature: (batch, n_nodes-1, 7) --> embed_customer_feature: (batch, n_nodes-1, embed_dim)
			embed_x(batch, n_nodes, embed_dim)

			return: (node embeddings(= embedding for all nodes), graph embedding(= mean of node embeddings for graph))
				=((batch, n_nodes, embed_dim), (batch, embed_dim))
		"""
		x = torch.cat([self.init_W_depot(x[0]),self.init_W(torch.cat([x[1], x[2][:, :, None],x[3][:,:,None],x[4][:,:,None],x[8]], dim = -1))],dim=1)
	
		for layer in self.encoder_layers:
			x = layer(x, mask)

		return (x, torch.mean(x, dim = 1))

if __name__ == '__main__':
	batch = 5
	n_nodes = 22
	encoder = GraphAttentionEncoder(n_layers = 1)
	data = generate_data('cuda:0' if torch.cuda.is_available() else 'cpu',n_samples = batch, n_customer = 20)
	output = encoder(data, mask = None)
	print('output[0].shape:', output[0].size())
	print('output[1].shape', output[1].size())
	
	cnt = 0
	for i, k in encoder.state_dict().items():
		print(i, k.size(), torch.numel(k))
		cnt += torch.numel(k)
	print(cnt)

