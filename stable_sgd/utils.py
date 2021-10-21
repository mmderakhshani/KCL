import uuid
import torch
import argparse
import matplotlib
import numpy as np
import pandas as pd
matplotlib.use('Agg')
import seaborn as sns
from pathlib import Path
import matplotlib.pyplot as plt
from external_libs.hessian_eigenthings import compute_hessian_eigenthings
import torch.nn.functional as F
import math
import random
import sys


TRIAL_ID = uuid.uuid4().hex.upper()[0:6]
EXPERIMENT_DIRECTORY = None
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def parse_arguments():
	parser = argparse.ArgumentParser(description='Argument parser')
	parser.add_argument('--tasks', default=5, type=int, help='total number of tasks')
	parser.add_argument('--epochs-per-task', default=1, type=int, help='epochs per task')
	parser.add_argument('--dataset', default='rot-mnist', type=str, help='dataset. options: rot-mnist, perm-mnist, cifar100')
	parser.add_argument('--kernel', default='rff', type=str, help='kernel type')
	parser.add_argument('--batch-size', default=10, type=int, help='batch-size')
	parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
	parser.add_argument('--gamma', default=0.4, type=float, help='learning rate decay. Use 1.0 for no decay')
	parser.add_argument('--lmd', default=0.1, type=float, help='the init of lmb')
	parser.add_argument("--zeta", type=float, default=0., help="hyper param for kernel")
	parser.add_argument("--tau", type=float, default=0.0001, help="hyper param for kl_loss")
	parser.add_argument("--d_rn_f", type=int, default=512, help="Size of the random feature base.")
	parser.add_argument("--core_size", type=int, default=20, help="coreset size")
	parser.add_argument('--dropout', default=0.25, type=float, help='dropout probability. Use 0.0 for no dropout')
	parser.add_argument('--hiddens', default=256, type=int, help='num of hidden neurons in each layer of a 2-layer MLP')
	parser.add_argument('--compute-eigenspectrum', default=False, type=bool, help='compute eigenvalues/eigenvectors?')
	parser.add_argument('--predefined', default=True, type=bool, help='predefined prior')
	parser.add_argument('--seed', default=1234, type=int, help='random seed')

	args = parser.parse_args()
	global EXPERIMENT_DIRECTORY
	EXPERIMENT_DIRECTORY = './outputs/{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}'.format(args.dataset, args.d_rn_f, args.core_size, args.seed, args.gamma, args.lr, args.dropout, args.batch_size, args.kernel, args.tau, args.predefined, args.tasks)
	return args


def init_experiment(args):
	print('------------------- Experiment started -----------------')
	print(f"Parameters:\n  seed={args.seed}\n  benchmark={args.dataset}\n  num_tasks={args.tasks}\n  "+
		  f"epochs_per_task={args.epochs_per_task}\n  batch_size={args.batch_size}\n  "+
		  f"learning_rate={args.lr}\n  learning rate decay(gamma)={args.gamma}\n  dropout prob={args.dropout}\n")
	
	# 1. setup seed for reproducibility
	random.seed(args.seed)
	np.random.seed(args.seed)
	torch.manual_seed(args.seed)
	torch.cuda.manual_seed(args.seed)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False
	# torch.backends.cudnn.enabled = False
	
	# 2. create directory to save results
	Path(EXPERIMENT_DIRECTORY).mkdir(parents=True, exist_ok=True)
	print("The results will be saved in {}\n".format(EXPERIMENT_DIRECTORY))
	
	# 3. create data structures to store metrics
	# loss_db = {t: [0 for i in range(args.tasks*args.epochs_per_task)] for t in range(1, args.tasks+1)}
	# acc_db =  {t: [0 for i in range(args.tasks*args.epochs_per_task)] for t in range(1, args.tasks+1)}
	loss_db = {t: [0 for i in range(args.tasks)] for t in range(1, args.tasks+1)}
	acc_db =  {t: [0 for i in range(args.tasks)] for t in range(1, args.tasks+1)}
	hessian_eig_db = {}
	return acc_db, loss_db, hessian_eig_db


def end_experiment(args, acc_db, loss_db, hessian_eig_db):
	
	# 1. save all metrics into csv file
	acc_df = pd.DataFrame(acc_db)
	acc_df.to_csv(EXPERIMENT_DIRECTORY+'/accs.csv')
	visualize_result(acc_df, "Accuracy", args, EXPERIMENT_DIRECTORY+'/accs.png')
	
	loss_df = pd.DataFrame(loss_db)
	loss_df.to_csv(EXPERIMENT_DIRECTORY+'/loss.csv')
	visualize_result(loss_df, "Loss", args, EXPERIMENT_DIRECTORY+'/loss.png')
	
	hessian_df = pd.DataFrame(hessian_eig_db)
	hessian_df.to_csv(EXPERIMENT_DIRECTORY+'/hessian_eigs.csv')
	
	# 2. calculate average accuracy and forgetting (c.f. ``evaluation`` section in our paper)
	score = np.mean([acc_db[i][-1] for i in acc_db.keys()])
	forget = np.mean([max(acc_db[i])-acc_db[i][-1] for i in range(1, args.tasks)])/100.0
	

	original_stdout = sys.stdout
	with open(EXPERIMENT_DIRECTORY+'/final_result.txt', 'w') as f:
		sys.stdout = f
		print('average accuracy = {}, forget = {}'.format(score, forget))
		sys.stdout = original_stdout
	print('average accuracy = {}, forget = {}'.format(score, forget))
	print()
	print('------------------- Experiment ended -----------------')


def log_metrics(metrics, time, prev_task_id, current_task_id, acc_db, loss_db):
	"""
	Log accuracy and loss at different times of training
	"""
	print('epoch {}, task:{}, metrics: {}'.format(time, prev_task_id, metrics))
	# log to db
	acc = metrics['accuracy']
	loss = metrics['loss']
	loss_db[prev_task_id][current_task_id-1]= loss
	acc_db[prev_task_id][current_task_id-1] = acc
	return acc_db, loss_db


def save_eigenvec(filename, arr):
	"""
	Save eigenvectors to file
	"""
	np.save(filename, arr)


def log_hessian(model, loader, time, task_id, hessian_eig_db):
	"""
	Compute and log Hessian for a specific task
	
	:param model:  The PyTorch Model
	:param loader: Dataloader [to calculate loss and then Hessian]
	:param time: time is a discrete concept regarding epoch. If we have T tasks each with E epoch,
	time will be from 0, to (T x E)-1. E.g., if we have 5 tasks with 5 epochs each, then when we finish
	task 1, time will be 5.
	:param task_id: Task id (to distiniguish between Hessians of different tasks)
	:param hessian_eig_db: (The dictionary to store hessians)
	:return:
	"""
	criterion = torch.nn.CrossEntropyLoss().to(DEVICE)
	use_gpu = True if DEVICE != 'cpu' else False
	est_eigenvals, est_eigenvecs = compute_hessian_eigenthings(
		model,
		loader,
		criterion,
		num_eigenthings=3,
		power_iter_steps=18,
		power_iter_err_threshold=1e-5,
		momentum=0,
		use_gpu=use_gpu,
	)
	key = 'task-{}-epoch-{}'.format(task_id, time-1)
	hessian_eig_db[key] = est_eigenvals
	save_eigenvec(EXPERIMENT_DIRECTORY+"/{}-vec.npy".format(key), est_eigenvecs)
	return hessian_eig_db


def save_checkpoint(model, time):
	"""
	Save checkpoints of model paramters
	:param model: pytorch model
	:param time: int
	"""
	filename = '{directory}/model-{trial}-{time}.pth'.format(directory=EXPERIMENT_DIRECTORY, trial=TRIAL_ID, time=time)
	torch.save(model.cpu().state_dict(), filename)


def visualize_result(df, y_label, args, filename):
	
	metrics = df.to_numpy()
	T = np.arange(1, args.tasks+1, 1)
	
	labels = [f"{i+1}" for i in range(args.tasks)]
	plt.xticks(T, labels)
	
	for idx in range(args.tasks):
		plt.plot(T[idx:], metrics[idx:,idx], marker="o")
	
	plt.legend(labels)
	plt.xlabel('Tasks')
	plt.ylabel(y_label)
	plt.ylim((0, max(metrics.max(), 1)))
	plt.grid(True)
	plt.savefig(filename, dpi=250)
	plt.close()

def sample(mu, logvar, L, device):
    shape = (L, ) + mu.size()
    eps   = torch.randn(shape).to(device)
    w     = mu.unsqueeze(0) + eps * logvar.exp().sqrt().unsqueeze(0)
    return w

def rand_features(bases, features, bias):
    # tf.random_normal()
    return math.sqrt(2/bias.shape[0]) * torch.cos(torch.matmul(bases, features) + bias)

def dotp_kernel(feature1, feature2):
    return torch.matmul(feature1, feature2)

def rbf_kernel(feature1, feature2):
	res = torch.exp(-0.25 * torch.norm(feature1.unsqueeze(1)-feature2, dim=2, p=1))
	return res

def linear(feature1, feature2):
    return feature1 @ feature2.T

def poly(feature1, feature2):
	return (torch.matmul(feature1, feature2.T) + 1).pow(3)

def kl_div(m, log_v, m0, log_v0):
    
    v = log_v.exp()
    v0 = log_v0.exp()

    dout, din = m.shape
    const_term = -0.5 * dout * din
    
    log_std_diff = 0.5 * torch.sum(torch.log(v0) - torch.log(v))
    mu_diff_term = 0.5 * torch.sum((v + (m0-m)**2) / v0)
    kl = const_term + log_std_diff + mu_diff_term
    return kl

def cosine_dist(a, b):
    # sqrt(<a, b>) / (sqrt(<a, a>), sqrt(<b, b>))
    a = a.view(1, -1)
    b = b.view(1, -1)
    normalize_a = F.normalize(a, dim=-1)# %
    normalize_b = F.normalize(b, dim=-1)
    return torch.sqrt(torch.sum(torch.multiply(normalize_a, normalize_b)))


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


