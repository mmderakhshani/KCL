import numpy as np
import torch
import torchvision
from torch.utils.data import TensorDataset, DataLoader
import torchvision.transforms.functional as TorchVisionFunc
import pickle

# class_ids = [(0, 12),
#  (13, 24),
#  (25, 27),
#  (28, 40),
#  (41, 46),
#  (47, 49),
#  (50, 53),
#  (54, 56),
#  (57, 63),
#  (64, 66),
#  (67, 71),
#  (72, 75),
#  (76, 83),
#  (84, 88),
#  (89, 91),
#  (92, 96),
#  (97, 99)]

def get_permuted_mnist(task_id):
	"""
	Get the dataset loaders (train and test) for a `single` task of permuted MNIST.
	This function will be called several times for each task.
	
	:param task_id: id of the task [starts from 1]
	:param batch_size:
	:return: a tuple: (train loader, test loader)
	"""
	
	# convention, the first task will be the original MNIST images, and hence no permutation
	if task_id == 1:
		idx_permute = np.array(range(784))
	else:
		idx_permute = torch.from_numpy(np.random.RandomState().permutation(784))
	transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
				torchvision.transforms.Lambda(lambda x: x.view(-1)[idx_permute] ),
				])
	mnist_train = torchvision.datasets.MNIST('./data/', train=True, download=True, transform=transforms)
	mnist_test  = torchvision.datasets.MNIST('./data/', train=False, download=True, transform=transforms)
	
	# train_loader = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size, num_workers=4, pin_memory=True, shuffle=True)
	# test_loader  = torch.utils.data.DataLoader(mnist_test,  batch_size=256, shuffle=False, num_workers=4, pin_memory=True)
	# return train_loader, test_loader
	return mnist_train, mnist_test



def get_permuted_mnist_tasks(num_tasks):
	"""
	Returns the datasets for sequential tasks of permuted MNIST
	
	:param num_tasks: number of tasks.
	:param batch_size: batch-size for loaders.
	:return: a dictionary where each key is a dictionary itself with train, and test loaders.
	"""
	datasets = {}
	for task_id in range(1, num_tasks+1):
		train_dataset, test_dataset = get_permuted_mnist(task_id)
		datasets[task_id] = {'train': train_dataset, 'test': test_dataset}
	return datasets


class RotationTransform:
	"""
	Rotation transforms for the images in `Rotation MNIST` dataset.
	"""
	def __init__(self, angle):
		self.angle = angle

	def __call__(self, x):
		return TorchVisionFunc.rotate(x, self.angle, fill=(0,))


def get_rotated_mnist(task_id):
	"""
	Returns the dataset for a single task of Rotation MNIST dataset
	:param task_id:
	:param batch_size:
	:return:
	"""
	per_task_rotation = 10
	rotation_degree = (task_id - 1)*per_task_rotation
	rotation_degree -= (np.random.random()*per_task_rotation)

	transforms = torchvision.transforms.Compose([
		RotationTransform(rotation_degree),
		torchvision.transforms.ToTensor(),
		])

	mnist_train = torchvision.datasets.MNIST('./data/', train=True, download=True, transform=transforms)
	mnist_test  = torchvision.datasets.MNIST('./data/', train=False, download=True, transform=transforms)
	
	# train_loader = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size, num_workers=4, pin_memory=True, shuffle=True)
	# test_loader  = torch.utils.data.DataLoader(mnist_test,  batch_size=256, shuffle=False, num_workers=4, pin_memory=True)
	# return train_loader, test_loader
	return mnist_train, mnist_test



def get_rotated_mnist_tasks(num_tasks):
	"""
	Returns data loaders for all tasks of rotation MNIST dataset.
	:param num_tasks: number of tasks in the benchmark.
	:param batch_size:
	:return:
	"""
	datasets = {}
	for task_id in range(1, num_tasks+1):
		train_dataset, test_dataset = get_rotated_mnist(task_id)
		datasets[task_id] = {'train': train_dataset, 'test': test_dataset}
	return datasets


def get_split_cifar100(task_id, cifar_train, cifar_test):
	"""
	Returns a single task of split CIFAR-100 dataset
	:param task_id:
	:param batch_size:
	:return:
	"""
	

	start_class = (task_id-1) * 5
	end_class = task_id * 5

	# start_class = class_ids[task_id-1][0]
	# end_class   = class_ids[task_id-1][1]

	targets_train = torch.tensor(cifar_train.targets)
	target_train_idx = ((targets_train >= start_class) & (targets_train < end_class))
	
	targets_test = torch.tensor(cifar_test.targets)
	target_test_idx = ((targets_test >= start_class) & (targets_test < end_class))

	# train_loader = torch.utils.data.DataLoader(torch.utils.data.dataset.Subset(cifar_train, np.where(target_train_idx==1)[0]), batch_size=batch_size, shuffle=True)
	# test_loader = torch.utils.data.DataLoader(torch.utils.data.dataset.Subset(cifar_test, np.where(target_test_idx==1)[0]), batch_size=batch_size)

	train_dataset = torch.utils.data.dataset.Subset(cifar_train, np.where(target_train_idx==1)[0])
	test_dataset  = torch.utils.data.dataset.Subset(cifar_test, np.where(target_test_idx==1)[0])

	return train_dataset, test_dataset


def get_split_cifar100_tasks(num_tasks):
	"""
	Returns data loaders for all tasks of split CIFAR-100
	:param num_tasks:
	:param batch_size:
	:return:
	"""
	datasets = {}
	
	# convention: tasks starts from 1 not 0 !
	# task_id = 1 (i.e., first task) => start_class = 0, end_class = 4
	cifar_transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),])
	cifar_train = torchvision.datasets.CIFAR100('./data/', train=True, download=True, transform=cifar_transforms)
	cifar_test = torchvision.datasets.CIFAR100('./data/', train=False, download=True, transform=cifar_transforms)
	
	for task_id in range(1, num_tasks+1):
		train_dataset, test_dataset = get_split_cifar100(task_id, cifar_train, cifar_test)
		datasets[task_id] = {'train': train_dataset, 'test': test_dataset}
	return datasets


def get_mini_imagenet_tasks(num_tasks):
	"""
	Returns data loaders for all tasks of split CIFAR-100
	:param num_tasks:
	:param batch_size:
	:return:
	"""
	data_file = './data/MiniImageNet/miniImageNet_full.pickle'
	imagenet_data = _load_miniImagenet(data_file)
	datasets = {}

	imagenet_images = imagenet_data['images']
	imagenet_labels = imagenet_data['labels']
	train_count = 500
	test_count = 100

	
	# import pdb; pdb.set_trace()
	for task_id in range(1, num_tasks+1):
		start_class = (task_id-1) * 5
		end_class = task_id * 5

		# start_class = class_ids[task_id-1][0]
		# end_class   = class_ids[task_id-1][1]	
		for count, cls in enumerate(range(start_class, end_class)):
			# Load all the examples of this class and split the data in train/ test
			class_indices = np.where(imagenet_labels == cls)[0]
			class_indices = np.sort(class_indices, axis=None)

			# Check if required 600 images per class class are present
			assert(class_indices.shape[0] == 600)

			if count == 0:
				x_train = imagenet_images[class_indices[:train_count]]
				y_train = imagenet_labels[class_indices[:train_count]]
				x_test = imagenet_images[class_indices[train_count:train_count+test_count]]
				y_test = imagenet_labels[class_indices[train_count:train_count+test_count]]
			else:
				x_train = np.concatenate((x_train, imagenet_images[class_indices[:train_count]]), axis=0)
				y_train = np.concatenate((y_train, imagenet_labels[class_indices[:train_count]]), axis=0)
				x_test = np.concatenate((x_test, imagenet_images[class_indices[train_count:train_count+test_count]]), axis=0)
				y_test = np.concatenate((y_test, imagenet_labels[class_indices[train_count:train_count+test_count]]), axis=0)
				
		train_dataset = torch.utils.data.TensorDataset(torch.tensor(x_train).float(), torch.tensor(y_train).long())
		test_dataset  = torch.utils.data.TensorDataset(torch.tensor(x_test).float(), torch.tensor(y_test).long())
		datasets[task_id] = {'train': train_dataset, 'test': test_dataset}
	return datasets

def _load_miniImagenet(data_file):
	"""
	Load the ImageNet data
	Args:
		data_file    Pickle file containing the miniImageNet data
	"""
	# Dictionary to store the dataset
	dataset = dict()
	IMG_MEAN = np.array((0.406, 0.456, 0.485), dtype=np.float32)
	IMG_STD = np.array((0.225, 0.224, 0.229), dtype=np.float32)

	# Load the whole miniimageNet dataset
	with open(data_file, 'rb') as f:
		data = pickle.load(f, encoding="latin-1")

	X = data['images']
	Y = data['labels']
	# Convert RGB image to BGR and subtract mean
	X = np.array(X, dtype=np.float32)
	X_r, X_g, X_b = np.split(X, 3, axis=3)
	X = np.concatenate((X_b, X_g, X_r), axis=3)
	print(X.min(), X.max())
	# Subtract image mean
	X /= 255.0
	X -= IMG_MEAN
	X /= IMG_STD

	dataset['images'] = X
	dataset['labels'] = Y

	return dataset


def get_train_loader(dataset, batch_size):
	train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=4, pin_memory=True, shuffle=True)
	return train_loader


def get_test_loader(dataset):
	test_loader  = torch.utils.data.DataLoader(dataset,  batch_size=256, shuffle=False, num_workers=4, pin_memory=True)
	return test_loader

def coreset_selection(dataset, coreset_size, dataset_name):
	# Uniformly sample and select data per each class from (x_train, y_train) and add to current coreset (x_coreset, y_coreset)
	if 'mnist' in dataset_name:
		targets = dataset.targets
		sample_per_each_class = coreset_size // len(targets.unique())
	else:
		loader = get_test_loader(dataset)
		inputs, targets = iterate_data(loader)
		sample_per_each_class = coreset_size

		
	idx = []
	for cl in targets.unique():
		cl = cl.item()
		cls_idx = torch.where(targets == cl)[0]
		if len(cls_idx) == 0:
			continue
		cls_idx = cls_idx[torch.randperm(len(cls_idx))[:sample_per_each_class]]
		idx = idx + cls_idx.tolist()

	remained_item = list(set(torch.arange(0, len(targets)).tolist()) - set(idx))
	if 'mnist' in dataset_name:
		train_dataset    = torch.utils.data.dataset.Subset(dataset, remained_item)
		coreset_dataset  = torch.utils.data.dataset.Subset(dataset, idx)
	else:
		train_dataset    = torch.utils.data.dataset.TensorDataset(inputs[remained_item], targets[remained_item])
		coreset_dataset  = torch.utils.data.dataset.TensorDataset(inputs[idx], targets[idx])
	return train_dataset, coreset_dataset

def iterate_data(loader):
    xs,ys = [], []
    for x,y in loader:
        xs.append(x)
        ys.append(y)
    xs = torch.cat(xs, dim=0).squeeze(1)
    ys = torch.cat(ys, dim=0)
    return xs, ys
	

# if __name__ == "__main__":
# 	dataset = get_mini_imagenet_tasks(20)



