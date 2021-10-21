import os
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from stable_sgd.models import MLP, ResNet18, ResNet18MiniImagenet
from stable_sgd.data_utils import get_permuted_mnist_tasks, get_rotated_mnist_tasks, get_split_cifar100_tasks, get_mini_imagenet_tasks, get_train_loader, get_test_loader, coreset_selection, iterate_data
from stable_sgd.utils import parse_arguments, DEVICE, init_experiment, end_experiment, log_metrics, log_hessian, save_checkpoint, sample, rand_features, dotp_kernel, kl_div, cosine_dist, AverageMeter, rbf_kernel, linear, poly
import math
from torchvision.utils import make_grid
import time
import pdb

def train_single_epoch(net, optimizer, loader, coreset_loader, criterion, bias, lmd, gma, bta, args, task_id=None):
	"""
	Train the model for a single epoch
	
	:param net:
	:param optimizer:
	:param loader:
	:param criterion:
	:param task_id:
	:return:
	"""
	# import pdb; pdb.set_trace()
	net = net.to(DEVICE)
	net.train()

	coreset_input, coreset_target = iterate_data(coreset_loader)
	coreset_input  = coreset_input.to(DEVICE)
	coreset_target = torch.nn.functional.one_hot(coreset_target, num_classes=10 if "mnist" in args.dataset else 100)
	coreset_target = coreset_target.to(DEVICE).float()
	
	task_acc = AverageMeter()
	task_loss = AverageMeter()
	
	for _, (data, target) in enumerate(loader):
		data = data.to(DEVICE)
		target = target.to(DEVICE)
		optimizer.zero_grad()
		
		if task_id:
			features_train   = net(data, task_id, post = False, kernel=args.kernel)
			features_coreset = net(coreset_input, task_id, post = True, kernel=args.kernel)
		else:
			features_train   = net(data,          post = False, kernel=args.kernel)
			features_coreset = net(coreset_input, post = True, kernel=args.kernel)

		# import pdb; pdb.set_trace()
		if args.kernel == "rbf":
			support_kernel = rbf_kernel(features_coreset, features_coreset)  # (number of coreset samples , number of coreset samples)
		elif args.kernel == "linear":
			support_kernel = linear(features_coreset, features_coreset)
		elif args.kernel == "poly":
			support_kernel = poly(features_coreset, features_coreset)
		else:
			raise Exception(f"Unknown kenrel.\n"+
						"Only support RBF, RFF, POLY, LIN.")

		alpha = torch.matmul(torch.inverse(support_kernel + (torch.abs(lmd[task_id-1]) + 0.01) * torch.eye(support_kernel.shape[0]).to(DEVICE)), coreset_target) 
		
		if args.kernel == "rbf":
			cross_kernel = rbf_kernel(features_coreset, features_train)  # (number of coreset samples , number of coreset samples)
		elif args.kernel == "linear":
			cross_kernel = linear(features_coreset, features_train)
		elif args.kernel == "poly":
			cross_kernel = poly(features_coreset, features_train)
		else:
			raise Exception(f"Unknown kenrel.\n"+
						"Only support RBF, RFF, POLY, LIN.")

		pred = gma[task_id-1] * torch.matmul(cross_kernel.T, alpha) + bta[task_id-1]
		
		
		if "cifar" in args.dataset:
			t = task_id
			offset1 = int((t-1) * 5)
			offset2 = int(t * 5)
			if offset1 > 0:
				pred[:, :offset1].data.fill_(-10e10)
			if offset2 < 100:
				pred[:, offset2:100].data.fill_(-10e10)

		
		loss = criterion(pred, target)
		task_loss.update(loss.item())

		acc = (pred.argmax(dim=-1) == target).float().mean()
		task_acc.update(acc.item())


		loss.backward()
		optimizer.step()
	print(task_acc.avg, task_loss.avg)
	return net


def loss_fn_kd(outputs, labels, teacher_outputs, alpha=0.95, T=6):
    KD_loss = nn.KLDivLoss()(F.log_softmax(outputs/T, dim=1),
                             F.softmax(teacher_outputs/T, dim=1)) * (alpha * T * T) + \
              F.cross_entropy(outputs, labels) * (1. - alpha)

    return KD_loss

def loss_dot_product(x):
	bs = x.shape[0]
	A = x.view(bs, -1)
	return torch.bmm(A.view(bs, 1, -1), A.view(bs, -1, 1)).mean()

def train_embedding(net, 
				optimizer, 
				loader, 
				coreset_loader, 
				task_embedding, 
				task_embedding_label, 
				criterion, 
				bias, 
				lmd, 
				gma, 
				bta, 
				args, 
				task_id=None, 
				writer=None):
	pdb.set_trace()
	"""
	Train the model for a single epoch
	
	:param net:
	:param optimizer:
	:param loader:
	:param criterion:
	:param task_id:
	:return:
	"""
	net = net.to(DEVICE)
	task_acc = AverageMeter()
	task_acc_embed = AverageMeter()
	task_loss = AverageMeter()

	coreset_input, coreset_target = iterate_data(coreset_loader)
	coreset_input  = coreset_input.to(DEVICE)
	coreset_target = torch.nn.functional.one_hot(coreset_target, num_classes=10 if "mnist" in args.dataset else 100)
	coreset_target = coreset_target.to(DEVICE).float()
	
	task_embedding_optimizer = torch.optim.Adam([task_embedding], lr=1e-1)
	task_embedding_label     = torch.nn.functional.one_hot(task_embedding_label[task_id-1], num_classes=10 if "mnist" in args.dataset else 100).float()

	# pdb.set_trace()
	vis_idx = 0

	for i, (data, target) in enumerate(loader):
		data = data.to(DEVICE)
		target = target.to(DEVICE)
		task_embedding_optimizer.zero_grad()
		optimizer.zero_grad()
		
		
		if task_id:
			features_train     = net(data, task_id, post = False, kernel=args.kernel)
			# features_coreset   = net(coreset_input, task_id, post = True, kernel=args.kernel)
			task_embd_features = net(task_embedding[task_id-1], task_id, post = True, kernel=args.kernel)
		else:
			features_train     = net(data,          post = False, kernel=args.kernel)
			# features_coreset   = net(coreset_input, post = True, kernel=args.kernel)
			task_embd_features = net(task_embedding[task_id-1], post = True, kernel=args.kernel)

		# import pdb; pdb.set_trace()
		if args.kernel == "rbf":
			# support_kernel = rbf_kernel(features_coreset, features_coreset)  # (number of coreset samples , number of coreset samples)
			task_embd_support_kernel = rbf_kernel(task_embd_features, task_embd_features)  # (number of coreset samples , number of coreset samples)
		elif args.kernel == "linear":
			# support_kernel = linear(features_coreset, features_coreset)
			task_embd_support_kernel = linear(task_embd_features, task_embd_features)  # (number of coreset samples , number of coreset samples)
		elif args.kernel == "poly":
			# support_kernel = poly(features_coreset, features_coreset)
			task_embd_support_kernel = poly(task_embd_features, task_embd_features)  # (number of coreset samples , number of coreset samples)

		else:
			raise Exception(f"Unknown kenrel.\n"+
						"Only support RBF, RFF, POLY, LIN.")

		# alpha           = torch.matmul(torch.inverse(support_kernel + (torch.abs(lmd[task_id-1]) + 0.01) * torch.eye(support_kernel.shape[0]).to(DEVICE)), coreset_target) 
		task_embd_alpha = torch.matmul(torch.inverse(task_embd_support_kernel + (torch.abs(lmd[task_id-1]) + 0.01) * torch.eye(task_embd_support_kernel.shape[0]).to(DEVICE)), task_embedding_label) 
		
		if args.kernel == "rbf":
			# cross_kernel           = rbf_kernel(features_coreset, features_train)  # (number of coreset samples , number of coreset samples)
			task_embd_cross_kernel = rbf_kernel(task_embd_features, features_train)  # (number of coreset samples , number of coreset samples)
		elif args.kernel == "linear":
			# cross_kernel           = linear(features_coreset, features_train)
			task_embd_cross_kernel = linear(task_embd_features, features_train)
		elif args.kernel == "poly":
			# cross_kernel           = poly(features_coreset, features_train)
			task_embd_cross_kernel = poly(task_embd_features, features_train)
		else:
			raise Exception(f"Unknown kenrel.\n"+
						"Only support RBF, RFF, POLY, LIN.")

		# pred           = gma[task_id-1] * torch.matmul(cross_kernel.T, alpha) + bta[task_id-1]
		task_embd_pred = gma[task_id-1] * torch.matmul(task_embd_cross_kernel.T, task_embd_alpha) + bta[task_id-1]

		
		# import pdb; pdb.set_trace()
		if "cifar" in args.dataset or "mini" in args.dataset:
			t = task_id
			offset1 = int((t-1) * 5)
			offset2 = int(t * 5)
			
			if offset1 > 0:
				# pred[:, :offset1].data.fill_(-10e10)
				task_embd_pred[:, :offset1].data.fill_(-10e10)
			if offset2 < 100:
				# pred[:, offset2:100].data.fill_(-10e10)
				task_embd_pred[:, offset2:100].data.fill_(-10e10)

		# pdb.set_trace()
		# task_embd_loss = torch.nn.functional.mse_loss(task_embd_pred, pred)
		# kl_loss        = kl_div(task_embd_mu, task_embd_logvar, r_mu, r_log_var)
		# loss           = task_embd_loss
		# loss           = args.tau * kl_loss
		# loss           = args.tau * kl_loss + task_embd_loss
		# inner_product_loss = loss_dot_product(task_embedding[task_id-1])
		# loss = loss_fn_kd(task_embd_pred, target, pred) #+ 1e-4*inner_product_loss
		loss = criterion(task_embd_pred, target)
		task_loss.update(loss.item())

		# acc            = (pred.argmax(dim=-1) == target).float().mean()
		acc_task_embed = (task_embd_pred.argmax(dim=-1) == target).float().mean()
		# task_acc.update(acc.item())
		task_acc_embed.update(acc_task_embed.item())


		loss.backward()
		optimizer.step()
		task_embedding_optimizer.step()
		
		# renormalize data between 0 and 1
		task_embedding.data.clamp_(0, 1)
		if (i+1) % 10 == 0:
			grid = make_grid(task_embedding[task_id-1], nrow=5)
			writer.add_image('images', grid, vis_idx)
			# pdb.set_trace()
			for idx in range(len(task_embedding[task_id-1])):
				# if idx % 2 == 0:
				writer.add_histogram(f"TASK-{task_id-1}-IMG-GRAD-{idx}", task_embedding.grad[task_id-1][idx].cpu().numpy(), vis_idx)
				writer.add_histogram(f"TASK-{task_id-1}-IMG-DATA-{idx}", task_embedding.data[task_id-1][idx].cpu().numpy(), vis_idx)
			vis_idx = vis_idx + 1
	# print(task_acc.avg, task_acc_embed.avg, task_loss.avg)
	print(task_acc_embed.avg, task_loss.avg)
	# pdb.set_trace()
	return net


def eval_single_epoch_embedding(net, 
					loader, 
					coreset_loader, 
					task_embedding, 
					task_embedding_label, 
					criterion, 
					bias, 
					lmd, 
					gma, 
					bta, 
					args, 
					task_id=None):
	"""
	Evaluate the model for single epoch
	
	:param net:
	:param loader:
	:param criterion:
	:param task_id:
	:return:
	"""

	net = net.to(DEVICE)
	net.eval()

	coreset_input, coreset_target = iterate_data(coreset_loader)
	coreset_input  = coreset_input.to(DEVICE)
	coreset_target = torch.nn.functional.one_hot(coreset_target, num_classes=10 if "mnist" in args.dataset else 100)
	coreset_target = coreset_target.to(DEVICE).float()
	task_embedding_label     = torch.nn.functional.one_hot(task_embedding_label, num_classes=10 if "mnist" in args.dataset else 100).float()

	test_loss = AverageMeter()
	correct   = AverageMeter()

	# pdb.set_trace()
	with torch.no_grad():
		for data, target in loader:
			data    = data.to(DEVICE)
			target  = target.to(DEVICE)
			
			
			if task_id:
				features_test      = net(data,           task_id, post = False, kernel=args.kernel)
				task_embd_features = net(task_embedding, task_id, post = True, kernel=args.kernel)
			else:
				features_test      = net(data,           post = False, kernel=args.kernel)
				task_embd_features = net(task_embedding, post = True, kernel=args.kernel)

			if args.kernel == "rbf":
				support_kernel_task_embed = rbf_kernel(task_embd_features, task_embd_features)  # (number of coreset samples , number of coreset samples)
			elif args.kernel == "linear":
				support_kernel_task_embed = linear(task_embd_features, task_embd_features)
			elif args.kernel == "poly":
				support_kernel_task_embed = poly(task_embd_features, task_embd_features)	
			else:
				raise Exception(f"Unknown kenrel.\n"+
							"Only support RBF, RFF, POLY, LIN.")
			
			# closed-form solution with trainable lmd
			alpha = torch.matmul(torch.inverse(support_kernel_task_embed + (torch.abs(lmd[task_id-1]) + 0.01) * torch.eye(support_kernel_task_embed.shape[0]).to(DEVICE)), task_embedding_label)
			
			if args.kernel == "rbf":
				cross_kernel = rbf_kernel(task_embd_features, features_test)  # (number of coreset samples , number of coreset samples)
			elif args.kernel == "linear":
				cross_kernel = linear(task_embd_features, features_test)
			elif args.kernel == "poly":
				cross_kernel = poly(task_embd_features, features_test)
			else:
				raise Exception(f"Unknown kenrel.\n"+
							"Only support RBF, RFF, POLY, LIN.")

			pred = gma[task_id-1] * torch.matmul(cross_kernel.T, alpha) + bta[task_id-1]
			
			loss = criterion(pred, target)
			
			test_loss.update(loss.item())

			acc = (pred.argmax(dim=-1) == target).float().mean()
			correct.update(acc.item())
	
	return {'accuracy': correct.avg, 'loss': test_loss.avg}

def eval_single_epoch_coreset(net, loader, coreset_loader, criterion, bias, lmd, gma, bta, args, task_id=None):
	"""
	Evaluate the model for single epoch
	
	:param net:
	:param loader:
	:param criterion:
	:param task_id:
	:return:
	"""

	net = net.to(DEVICE)
	net.eval()

	coreset_input, coreset_target = iterate_data(coreset_loader)
	coreset_input  = coreset_input.to(DEVICE)
	coreset_target = torch.nn.functional.one_hot(coreset_target, num_classes=10 if "mnist" in args.dataset else 100)
	coreset_target = coreset_target.to(DEVICE).float()

	test_loss = AverageMeter()
	correct   = AverageMeter()

	with torch.no_grad():
		for data, target in loader:
			data = data.to(DEVICE)
			target = target.to(DEVICE)
			start = time.process_time()
			if task_id:
				features_test   = net(data, task_id, post = False, kernel=args.kernel)
				features_coreset = net(coreset_input, task_id, post = True, kernel=args.kernel)
			else:
				features_test   = net(data,          post = False, kernel=args.kernel)
				features_coreset = net(coreset_input, post = True, kernel=args.kernel)

			# compute_bases
			# compute_bases
			# rs = sample(r_mu, r_log_var, args.d_rn_f, DEVICE).squeeze()


			# Here we map the coreset data into random fourier space. 
			# x_supp_phi_t = rand_features(rs, torch.transpose(features_coreset, 1, 0), bias)  # (d_rn_f , number of coreset samples)
			# in the random fourier space we compute the the kernel matrix, the dot product between mapped samples
			if args.kernel == "rbf":
				support_kernel = rbf_kernel(features_coreset, features_coreset)  # (number of coreset samples , number of coreset samples)
			elif args.kernel == "linear":
				support_kernel = linear(features_coreset, features_coreset)
			elif args.kernel == "poly":
				support_kernel = poly(features_coreset, features_coreset)	
			else:
				raise Exception(f"Unknown kenrel.\n"+
							"Only support RBF, RFF, POLY, LIN.")
			# import pdb; pdb.set_trace()
			# closed-form solution with trainable lmd
			# alpha = torch.matmul(torch.inverse(support_kernel + (lmd_abs + 0.01) * torch.eye(support_kernel[0])), target)
			# (number of coreset samples, number of classes)
			alpha = torch.matmul(torch.inverse(support_kernel + (torch.abs(lmd[task_id-1]) + 0.01) * torch.eye(support_kernel.shape[0]).to(DEVICE)), coreset_target) 
			
			# Here we map the train data into random fourier space. 
			# x_que_phi_t = rand_features(rs, torch.transpose(features_train, 1, 0), bias)

			# Compute cross kernel
			# cross_kernel = dotp_kernel(torch.transpose(x_supp_phi_t, 1, 0), x_que_phi_t) # (number of coreset samples, number of train samples)
			if args.kernel == "rbf":
				cross_kernel = rbf_kernel(features_coreset, features_test)  # (number of coreset samples , number of coreset samples)
			elif args.kernel == "linear":
				cross_kernel = linear(features_coreset, features_test)
			elif args.kernel == "poly":
				cross_kernel = poly(features_coreset, features_test)
			else:
				raise Exception(f"Unknown kenrel.\n"+
							"Only support RBF, RFF, POLY, LIN.")


			pred = gma[task_id-1] * torch.matmul(cross_kernel.T, alpha) + bta[task_id-1]
			end = time.process_time()
		
			loss = criterion(pred, target)
			
			test_loss.update(loss.item())

			acc = (pred.argmax(dim=-1) == target).float().mean()
			correct.update(acc.item())
	print(f"estimated time: {end - start}")
	return {'accuracy': correct.avg, 'loss': test_loss.avg}

# def eval_single_epoch(net, loader, coreset_loader, criterion, bias, lmd, gma, bta, args, task_id=None):
# 	"""
# 	Evaluate the model for single epoch
	
# 	:param net:
# 	:param loader:
# 	:param criterion:
# 	:param task_id:
# 	:return:
# 	"""

# 	net = net.to(DEVICE)
# 	net.eval()

# 	coreset_input, coreset_target = iterate_data(coreset_loader)
# 	coreset_input  = coreset_input.to(DEVICE)
# 	coreset_target = torch.nn.functional.one_hot(coreset_target, num_classes=10 if "mnist" in args.dataset else 100)
# 	coreset_target = coreset_target.to(DEVICE).float()

# 	test_loss = AverageMeter()
# 	correct   = AverageMeter()

# 	with torch.no_grad():
# 		for data, target in loader:
# 			data = data.to(DEVICE)
# 			target = target.to(DEVICE)
# 			if task_id:
# 				features_test,   p_mu, p_log_var = net(data, task_id, post = False)
# 				features_coreset, r_mu, r_log_var  = net(coreset_input, task_id, post = True)
# 			else:
# 				features_test,   p_mu, p_log_var = net(data,          post = False)
# 				features_coreset, r_mu, r_log_var  = net(coreset_input, post = True)

# 			# compute_bases
# 			rs = sample(r_mu, r_log_var, args.d_rn_f, DEVICE).squeeze()

# 			x_supp_phi_t = rand_features(rs, torch.transpose(features_coreset, 1, 0), bias)  
			
			
# 			support_kernel = dotp_kernel(torch.transpose(x_supp_phi_t, 1, 0), x_supp_phi_t)  

# 			# closed-form solution with trainable lmd
# 			alpha = torch.matmul(torch.inverse(support_kernel + (torch.abs(lmd[task_id-1]) + 0.01) * torch.eye(support_kernel.shape[0]).to(DEVICE)), coreset_target)
# 			x_que_phi_t = rand_features(rs, torch.transpose(features_test, 1, 0), bias)

# 			# Compute cross kernel
# 			cross_kernel = dotp_kernel(torch.transpose(x_supp_phi_t, 1, 0), x_que_phi_t)

# 			pred = gma[task_id-1] * torch.matmul(cross_kernel.T, alpha) + bta[task_id-1]
# 			# align loss
# 			target_kernel = dotp_kernel(coreset_target, coreset_target.T)
# 			target_kernel = 0.99 * (target_kernel + 0.01)
			
# 			kernel_align_loss = cosine_dist(target_kernel, support_kernel)

# 			kl_loss = kl_div(r_mu, r_log_var, p_mu, p_log_var)
			
# 			loss = criterion(pred, target) + args.zeta * kernel_align_loss + args.tau * kl_loss
			
# 			test_loss.update(loss.item())

# 			acc = (pred.argmax(dim=-1) == target).float().mean()
# 			correct.update(acc.item())
	
# 	return {'accuracy': correct.avg, 'loss': test_loss.avg}




def save_stats_avg(net, coreset_all, bias, args):
	mu = []
	logvar = []
	rff = []

	net = net.to(DEVICE)
	net.eval()
	for task_id, coreset in enumerate(coreset_all):
		coreset_loader  = get_test_loader(coreset)
		coreset_input, _ = iterate_data(coreset_loader)
		coreset_input  = coreset_input.to(DEVICE)
		features_coreset, r_mu, r_log_var  = net(coreset_input, task_id, post = True)
		mu.append(r_mu)
		logvar.append(r_log_var)
		rs = sample(r_mu, r_log_var, args.d_rn_f, DEVICE).squeeze()
		x_supp_phi_t = rand_features(rs, torch.transpose(features_coreset, 1, 0), bias).detach().cpu()
		rff.append(x_supp_phi_t)
	
	mu     = torch.cat(mu, dim=0).detach().cpu()
	logvar = torch.cat(logvar, dim=0).detach().cpu()

	stats = {"mu": mu, "logvar": logvar, "rff": rff}
	torch.save(stats, f'./stats_avg/stat_{args.dataset}.pth')


def save_stats_one_task(net, coreset_all, bias, args, task_id, basis = [], rff = [], mu=[], log_var=[]):
	net = net.to(DEVICE)
	net.eval()

	coreset = coreset_all[task_id-1]
	coreset_loader  = get_test_loader(coreset)
	coreset_input, _ = iterate_data(coreset_loader)
	coreset_input  = coreset_input.to(DEVICE)
	features_coreset, r_mu, r_log_var  = net(coreset_input, task_id, post = True)
	rs = sample(r_mu, r_log_var, args.d_rn_f, DEVICE).squeeze()
	x_supp_phi_t = rand_features(rs, torch.transpose(features_coreset, 1, 0), bias).detach().cpu()
	basis = basis + [rs.cpu()]
	rff = rff + [x_supp_phi_t]
	mu = mu + [r_mu]
	log_var = log_var + [r_log_var]
	
	return basis, rff, mu, log_var


def get_benchmark_dataset(args):
	"""
	Returns the benchmark loader which could be either of these:
	get_split_cifar100_tasks, get_permuted_mnist_tasks, or get_rotated_mnist_tasks
	
	:param args:
	:return: a function which when called, returns all tasks
	"""
	if args.dataset == 'perm-mnist' or args.dataset == 'permuted-mnist':
		return get_permuted_mnist_tasks
	elif args.dataset == 'rot-mnist' or args.dataset == 'rotation-mnist':
		return get_rotated_mnist_tasks
	elif args.dataset == 'cifar-100' or args.dataset == 'cifar100':
		return get_split_cifar100_tasks
	elif args.dataset == 'mini-imagenet':
		return get_mini_imagenet_tasks
	else:
		raise Exception("Unknown dataset.\n"+
						"The code supports 'perm-mnist, rot-mnist, and cifar-100.")


def get_benchmark_model(args):
	"""
	Return the corresponding PyTorch model for experiment
	:param args:
	:return:
	"""
	if 'mnist' in args.dataset:
		if args.tasks == 20 and args.hiddens < 256:
			print("Warning! the main paper MLP with 256 neurons for experiment with 20 tasks")
		return MLP(args.hiddens, {'dropout': args.dropout}).to(DEVICE)
	elif 'cifar' in args.dataset:
		return ResNet18(config={'dropout': args.dropout}).to(DEVICE)
	elif 'imagenet' in args.dataset:
		return ResNet18MiniImagenet(config={'dropout': args.dropout}).to(DEVICE)
	else:
		raise Exception("Unknown dataset.\n"+
						"The code supports 'perm-mnist, rot-mnist, and cifar-100.")


def train_task_embedding(model, coreset_loader):
	coreset_input, coreset_target = iterate_data(coreset_loader)
	coreset_input  = coreset_input.to(DEVICE)
	coreset_target = torch.nn.functional.one_hot(coreset_target, num_classes=10 if "mnist" in args.dataset else 100)
	coreset_target = coreset_target.to(DEVICE).float()

def run(args):
	"""
	Run a single run of experiment.
	
	:param args: please see `utils.py` for arguments and options
	"""
	# pdb.set_trace()
	acc_db, loss_db, hessian_eig_db = init_experiment(args)
	print("Loading {} tasks for {}".format(args.tasks, args.dataset))
	tasks = get_benchmark_dataset(args)(args.tasks)
	print("loaded all tasks!")
	model = get_benchmark_model(args)

	# Prepare visulization toolbox
	writer = SummaryWriter()

	# criterion
	criterion = nn.CrossEntropyLoss().to(DEVICE)
	bias = 2 * math.pi * torch.rand(args.d_rn_f, 1).to(DEVICE)
	time = 0

	# Learnable coefficient

	lmd   = torch.tensor([args.lmd for _ in range(args.tasks)], requires_grad=True, device=DEVICE)
	gma   = torch.tensor([1.0      for _ in range(args.tasks)], requires_grad=True, device=DEVICE)
	bta   = torch.tensor([0.       for _ in range(args.tasks)], requires_grad=True, device=DEVICE)

	
	num_sample_per_cls = 1
	task_embedding       = torch.rand(args.tasks, 5 * num_sample_per_cls, 3, 32, 32).float().to(DEVICE).requires_grad_()
	task_embedding_label = torch.tensor([i for i in range(args.tasks*5)]).long().to(DEVICE)
	task_embedding_label = task_embedding_label.view(args.tasks, -1)
	task_embedding_label = torch.cat([task_embedding_label for i in range(num_sample_per_cls)], dim=-1)

	coreset_all = []
	# basis = []
	# rff = []
	# mu = []
	# log_var = []
	for current_task_id in range(1, args.tasks+1):
		print("================== TASK {} / {} =================".format(current_task_id, args.tasks))
		train_dataset = tasks[current_task_id]['train']
		train_dataset, coreset_dataset = coreset_selection(train_dataset, args.core_size, args.dataset)

		# import pdb; pdb.set_trace()
		# save coreset
		coreset_all.append(coreset_dataset)
		
		# uncomment for replay
		# coreset_dataset = torch.utils.data.dataset.ConcatDataset(coreset_all)
		# print(len(coreset_dataset))

		train_loader    = get_train_loader(train_dataset, args.batch_size)
		coreset_loader  = get_train_loader(coreset_dataset, args.batch_size)
		lr = max(args.lr * args.gamma ** (current_task_id), 0.00005) # Gamma Fucky is here. Fucky mocky
		
	
		
		for epoch in range(1, args.epochs_per_task+1):
			# 1. train and save
			params = list(model.parameters()) + [lmd, gma, bta]
			optimizer = torch.optim.SGD(params, lr=lr, momentum=0.8) # buggy fuccky
			train_embedding(model, optimizer, train_loader, coreset_loader, task_embedding, task_embedding_label, criterion, bias, lmd, gma, bta, args, current_task_id, writer)
			time += 1
			# import pdb; pdb.set_trace()
			# 2. evaluate on all tasks up to now, including the current task
			for prev_task_id in range(1, current_task_id+1):
				# 2.0. only evaluate once a task is finished
				if epoch == args.epochs_per_task:
					model = model.to(DEVICE)
					val_dataset = tasks[prev_task_id]['test']
					coreset_dataset = coreset_all[prev_task_id-1]

					val_loader      = get_test_loader(val_dataset)
					coreset_loader  = get_test_loader(coreset_dataset)

					task_embed = task_embedding[prev_task_id-1]
					task_lbl   = task_embedding_label[prev_task_id-1]
					
					# 2.1. compute accuracy and loss
					metrics = eval_single_epoch_embedding(model, val_loader, coreset_loader, task_embed, task_lbl, criterion, bias, lmd, gma, bta, args, current_task_id)
					# metrics = eval_single_epoch_coreset(model, val_loader, coreset_loader, criterion, bias, lmd, gma, bta, args, current_task_id)
					acc_db, loss_db = log_metrics(metrics, time, prev_task_id, current_task_id, acc_db, loss_db)
					
					# 2.2. (optional) compute eigenvalues and eigenvectors of Loss Hessian
					if prev_task_id == current_task_id and args.compute_eigenspectrum:
						hessian_eig_db = log_hessian(model, val_loader, time, prev_task_id, hessian_eig_db)
						
					# 2.3. save model parameters
					save_checkpoint(model, time)
		# basis, rff, mu, log_var = save_stats_one_task(model, coreset_all, bias, args, 1, basis, rff, mu, log_var)
	end_experiment(args, acc_db, loss_db, hessian_eig_db)
	# save_stats_avg(model, coreset_all, bias, args)


	# # save runing stats
	# stats = {"basis": basis, "rff": rff, "mu": mu, "log_var": log_var}
	# torch.save(stats, f'./stats_one_task/stat_{args.dataset.replace("-","_")}_one_task.pth')

if __name__ == "__main__":
	args = parse_arguments()
	run(args)