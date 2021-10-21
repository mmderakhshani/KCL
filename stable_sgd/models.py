import torch
import torch.nn as nn
from torch.nn.functional import relu, avg_pool2d, adaptive_avg_pool2d
from stable_sgd.utils import DEVICE

from torch.nn.functional import normalize


class InferenceBlock(nn.Module):
    def __init__(self, input_units, d_theta, output_units):
        """
        :param d_theta: dimensionality of the intermediate hidden layers.
        :param output_units: dimensionality of the output.
        :return: batch of outputs.
        """
        super(InferenceBlock, self).__init__()
        self.module = nn.Sequential(
            nn.Linear(input_units, d_theta, bias=True),
            nn.ELU(inplace=True),
            nn.Linear(input_units, d_theta, bias=True),
            nn.ELU(inplace=True),
            nn.Linear(d_theta, output_units, bias=True)
        )

    def forward(self, inps):
        out = self.module(inps)
        return out

class Amortized(nn.Module):
    def __init__(self, input_units=400, d_theta=400, output_units=400):
        super(Amortized, self).__init__()
        self.output_units = output_units
        self.weight_mean         = InferenceBlock(input_units, d_theta, output_units)
        self.weight_log_variance = InferenceBlock(input_units, d_theta, output_units)

    def forward(self, inps):
        weight_mean         = self.weight_mean(inps)
        weight_log_variance = self.weight_log_variance(inps)
        return weight_mean, weight_log_variance


class MLP(nn.Module):
	"""
	Two layer MLP for MNIST benchmarks.
	"""
	def __init__(self, hiddens, config):
		super(MLP, self).__init__()
		self.W1 = nn.Linear(784, hiddens)
		self.relu = nn.ReLU(inplace=True)
		self.dropout_1 = nn.Dropout(p=config['dropout'])
		self.W2 = nn.Linear(hiddens, hiddens)
		self.dropout_2 = nn.Dropout(p=config['dropout'])
		# self.W3 = nn.Linear(hiddens, 10)
		self.post  = Amortized(hiddens, hiddens, hiddens)
		self.prior = Amortized(hiddens, hiddens, hiddens)

	def forward(self, x, task_id=None, post = False, predefined = False):
		x = x.view(-1, 784)
		out = self.W1(x)
		out = self.relu(out)
		out = self.dropout_1(out)
		out = self.W2(out)
		out = self.relu(out)
		out = self.dropout_2(out)
		out_features = self.normalize(out)
		out_mean = torch.mean(out_features, dim=0, keepdim=True)
		if post:
			mu, logvar = self.post(out_mean)
		else:
			mu, logvar = self.prior(out_mean)
		return out_features, mu, logvar

	def normalize(self, x):
		max_val = x.max()
		min_val = x.min()
		return (x - min_val) / (max_val - min_val)

def conv3x3(in_planes, out_planes, stride=1):
	return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
					 padding=1, bias=False)


class BasicBlock(nn.Module):
	expansion = 1

	def __init__(self, in_planes, planes, stride=1, config={}):
		super(BasicBlock, self).__init__()
		self.conv1 = conv3x3(in_planes, planes, stride)
		self.conv2 = conv3x3(planes, planes)

		self.shortcut = nn.Sequential()
		if stride != 1 or in_planes != self.expansion * planes:
			self.shortcut = nn.Sequential(
				nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1,
						  stride=stride, bias=False),
			)
		self.IC1 = nn.Sequential(
			nn.BatchNorm2d(planes),
			nn.Dropout(p=config['dropout'])
			)

		self.IC2 = nn.Sequential(
			nn.BatchNorm2d(planes),
			nn.Dropout(p=config['dropout'])
			)

	def forward(self, x):
		out = self.conv1(x)
		out = relu(out)
		out = self.IC1(out)

		out += self.shortcut(x)
		out = relu(out)
		out = self.IC2(out)
		return out


class ResNet(nn.Module):
	def __init__(self, block, num_blocks, num_classes, nf, config={}):
		super(ResNet, self).__init__()
		self.in_planes = nf

		self.conv1 = conv3x3(3, nf * 1)
		self.bn1 = nn.BatchNorm2d(nf * 1)
		self.layer1 = self._make_layer(block, nf * 1, num_blocks[0], stride=1, config=config)
		self.layer2 = self._make_layer(block, nf * 2, num_blocks[1], stride=2, config=config)
		self.layer3 = self._make_layer(block, nf * 4, num_blocks[2], stride=2, config=config)
		self.layer4 = self._make_layer(block, nf * 8, num_blocks[3], stride=2, config=config)
		# self.linear = nn.Linear(nf * 8 * block.expansion, num_classes)
		self.hiddens = nf * 8 * block.expansion
		self.post  = Amortized(self.hiddens, self.hiddens, self.hiddens)
		self.prior = Amortized(self.hiddens, self.hiddens, self.hiddens)

	def _make_layer(self, block, planes, num_blocks, stride, config):
		strides = [stride] + [1] * (num_blocks - 1)
		layers = []
		for stride in strides:
			layers.append(block(self.in_planes, planes, stride, config=config))
			self.in_planes = planes * block.expansion
		return nn.Sequential(*layers)

	def forward(self, x, task_id=None, post = False, kernel="rff", predefined=False):
		bsz = x.size(0)
		out = relu(self.bn1(self.conv1(x.view(bsz, 3, 32, 32))))
		out = self.layer1(out)
		out = self.layer2(out)
		out = self.layer3(out)
		out = self.layer4(out)
		out = adaptive_avg_pool2d(out, 1)
		out = out.view(out.size(0), -1)
		out_features = normalize(out)
		out_mean = torch.mean(out_features, dim=0, keepdim=True)
		if kernel == "rff":
			if post:
				mu, logvar = self.post(out_mean)
			else:
				# if predefined == False:
				mu, logvar = self.prior(out_mean)
				# else:
			
				# 	mu     = torch.zeros(1, self.hiddens).to(DEVICE)
				# 	logvar = torch.ones(1, self.hiddens).to(DEVICE)

			return out_features, mu, logvar
		else:
			return out_features


class ResNetMiniImagenet(nn.Module):
	def __init__(self, block, num_blocks, nf, s, config={}):
		super(ResNetMiniImagenet, self).__init__()
		self.in_planes = nf[0]

		self.conv1 = self._conv7x7(3, nf[0], stride=s[0])
		self.bn1 = nn.BatchNorm2d(nf[0])
		self.layer1 = self._make_layer(block, nf[1], num_blocks[0], stride=s[1], config=config)
		self.layer2 = self._make_layer(block, nf[2], num_blocks[1], stride=s[2], config=config)
		self.layer3 = self._make_layer(block, nf[3], num_blocks[2], stride=s[3], config=config)
		self.layer4 = self._make_layer(block, nf[4], num_blocks[3], stride=s[4], config=config)
		# self.linear = nn.Linear(nf * 8 * block.expansion, num_classes)
		self.hiddens = nf[4]
		self.post  = Amortized(self.hiddens, self.hiddens, self.hiddens)
		self.prior = Amortized(self.hiddens, self.hiddens, self.hiddens)

	def _conv7x7(self, in_planes, out_planes, stride=1):
		return nn.Conv2d(in_planes, out_planes, kernel_size=7, stride=stride,
						padding=1, bias=False)
	def _make_layer(self, block, planes, num_blocks, stride, config):
		strides = [stride] + [1] * (num_blocks - 1)
		layers = []
		for stride in strides:
			layers.append(block(self.in_planes, planes, stride, config=config))
			self.in_planes = planes * block.expansion
		return nn.Sequential(*layers)

	def forward(self, x, task_id=None, post = False, kernel="rff", predefined=False):

		x = x.permute(0, 3, 1, 2)
		bsz, ch, h , w = x.size()
		out = relu(self.bn1(self.conv1(x.view(bsz, ch, h, w))))
		out = self.layer1(out)
		out = self.layer2(out)
		out = self.layer3(out)
		out = self.layer4(out)
		out = adaptive_avg_pool2d(out, 1)
		out = out.view(out.size(0), -1)
		out_features = normalize(out, dim=1, p=2)
		out_mean = torch.mean(out_features, dim=0, keepdim=True)
		if kernel == "rff":
			if post:
				mu, logvar = self.post(out_mean)
			else:
				mu, logvar = self.prior(out_mean)
			
			return out_features, mu, logvar
		else:
			return out_features


def ResNet18(nclasses=100, nf=20, config={}):
	net = ResNet(BasicBlock, [2, 2, 2, 2], nclasses, nf, config=config)
	return net

def ResNet18MiniImagenet(nclasses=100, config={}):
	net = ResNetMiniImagenet(BasicBlock, [2, 2, 2, 2], 
					nf=[64, 64, 128, 128, 512], 
					s=[2, 2, 2, 2, 2], 
					config=config)
	return net


# if __name__ == "__main__":
# 	net = ResNetMiniImagenet(BasicBlock, [2, 2, 2, 2], nf=[64, 64, 128, 128, 512], s=[2, 2, 2, 2, 2], config={"dropout":0.5})
# 	net = net.cuda()
# 	t = torch.randn(1, 3, 224, 224).cuda()
# 	print(net(t))