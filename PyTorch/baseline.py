import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from scipy.stats import ttest_rel
from tqdm import tqdm
import copy

from data import generate_data, Generator
from model import AttentionModel
def load_model(path, embed_dim = 128, n_customer = 20, n_encode_layers = 3):
	model_loaded = AttentionModel(embed_dim = embed_dim, n_encode_layers = n_encode_layers, n_heads = 8, tanh_clipping = 10., FF_hidden = 512)
	if torch.cuda.is_available():
		model_loaded.load_state_dict(torch.load(path))
	else:
		model_loaded.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
	return model_loaded


class RolloutBaseline:
	def __init__(self, model, task, weight_dir, n_rollout_samples = 10000, 
				embed_dim = 128, n_customer = 20, warmup_beta = 0.8, wp_epochs = 1, device = 'cpu',
				from_checkpoint = False, path_to_checkpoint = None, epoch = 0,
				):
		"""
		Args:
			model: current model
			task: suffix for baseline checkpoint task
			from_checkpoint: start from checkpoint flag
			path_to_checkpoint: path to baseline model weights
			wp_epochs: until when epoch reaches wp_n_epocohs do we warm-up
			epoch: current epoch number
			n_rollout_samples: number of samples to be generated for baseline dataset
			warmup_beta: warmup mixing parameter (exp. exponential moving average parameter)
		"""

		self.n_rollout_samples = n_rollout_samples
		self.cur_epoch = epoch
		self.wp_epochs = wp_epochs
		self.beta = warmup_beta

		self.alpha = 0.0

		self.M = None

		# パラメータ
		self.task = task
		self.from_checkpoint = from_checkpoint
		self.path_to_checkpoint = path_to_checkpoint

		# 問題のパラメータ
		self.embed_dim = embed_dim
		self.n_customer = n_customer
		self.weight_dir = weight_dir

		self.device = device

		self._update_baseline(model, epoch)
		

	def _update_baseline(self, model, epoch):
		if self.from_checkpoint and self.alpha == 0:
			print('Baseline model loaded')
			self.model = self.load_model(self.path_to_checkpoint, embed_dim = self.embed_dim, n_customer = self.n_customer)
		else:
			print('Baseline model copied')
			self.model = self.copy_model(model)
			torch.save(self.model.state_dict(), '%s%s_epoch%s.pt'%(self.weight_dir, self.task, epoch))
		
		self.model = self.model.to(self.device)
		self.dataset = Generator(self.device, n_samples = self.n_rollout_samples, n_customer = self.n_customer)

		print(f'Evaluating baseline model on baseline dataset (epoch = {epoch})')
		self.bl_vals = self.rollout(self.model, self.dataset).cpu().numpy()
		self.mean = self.bl_vals.mean()
		self.cur_epoch = epoch

	def ema_eval(self, cost):
		if self.M is None:
			self.M = cost.mean()
		else:
			self.M = self.beta * self.M + (1. - self.beta) * cost.mean()
		return self.M.detach()

	def eval(self, batch, cost):
		if self.alpha == 0:
			return self.ema_eval(cost)

		if self.alpha < 1:
			v_ema = self.ema_eval(cost)
		else:
			v_ema = 0.0

		with torch.no_grad():
			v_b, _ = self.model(batch, decode_type = 'greedy')

		return self.alpha * v_b + (1 - self.alpha) * v_ema

	def eval_all(self, dataset):
		if self.alpha < 1:
			return None

		val_costs = self.rollout(self.model, dataset, batch = 2048)

		return val_costs

	def epoch_callback(self, model, epoch):
		self.cur_epoch = epoch

		print(f'Evaluating candidate model on baseline dataset (callback epoch = {self.cur_epoch})')
		candidate_vals = self.rollout(model, self.dataset).cpu().numpy()# costs for training model on baseline dataset
		candidate_mean = candidate_vals.mean()

		print(f'Epoch {self.cur_epoch} candidate mean {candidate_mean}, baseline mean {self.mean}')

		if candidate_mean < self.mean:
			t, p = ttest_rel(candidate_vals, self.bl_vals)# scipy.stats.ttest_rel

			p_val = p / 2
			print(f'p-value: {p_val}')

			if p_val < 0.05:
				print('Update baseline')
				self._update_baseline(model, self.cur_epoch)

		if self.alpha < 1.0:
			self.alpha = (self.cur_epoch + 1) / float(self.wp_epochs)
			print(f'alpha was updated to {self.alpha}')

	def copy_model(self, model):
		new_model = copy.deepcopy(model)
		return new_model

	def rollout(self, model, dataset, batch = 1000, disable_tqdm = False):
		costs_list = []
		dataloader = DataLoader(dataset, batch_size = batch)
		for inputs in tqdm(dataloader, disable = disable_tqdm, desc = 'Rollout greedy execution'):
			with torch.no_grad():
				cost, _ ,_,_= model(inputs, decode_type = 'greedy')
				costs_list.append(cost)
		return torch.cat(costs_list, 0)