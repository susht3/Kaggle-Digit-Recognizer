import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np


class baseCNN(nn.Module):
	def __init__(self, param):
		super(baseCNN, self).__init__()
		self.x_size = param.x_size
		self.label_size = param.label_size
		self.hidden_size = 32
		self.conv1 = nn.Conv2d(1, 32, (3, 3), stride=(1,1))
		self.conv2 = nn.Conv2d(32, 48, (5, 5), stride=(1,1))

		self.pooling1 = nn.MaxPool2d((2,2), stride=(2,2))
		self.pooling2 = nn.MaxPool2d((3,3), stride=(3,3))

		self.norm1 = nn.BatchNorm1d(32)
		self.norm2 = nn.BatchNorm1d(48)

		self.linear = nn.Linear(432, 128)
		self.linear2 = nn.Linear(128, self.label_size)
		#self.linear = nn.Linear(640, self.label_size)
		self.drop = nn.Dropout(p=0.2)

		#self.weight = torch.FloatTensor([1, 1.5]) 
		self.loss_func = nn.NLLLoss()


	def forward(self, x):
		x = x.unsqueeze(1)
		conv1 = F.relu(self.conv1(x))
		#print('conv1: ', conv1.size())
		p1 = self.pooling1(conv1)
		#print('p1: ', p1.size())
		n1 = self.norm1(p1)
		#print('n1: ', n1.size())
        
		conv2 = F.relu(self.conv2(n1))
		#print('conv2: ', conv2.size())
		p2 = self.pooling2(conv2)
		#print('p2: ', p2.size())
		n2 = self.norm2(p2)

		#print('n2: ', n2.size())
		flat = n2.view(n2.size(0), n2.size(1)*n2.size(2)*n2.size(3))
		#print('flat: ', flat.size())
		logit = self.linear(flat)
		logit = self.linear2(logit)
		logit = F.log_softmax(logit)
		return logit

	def get_loss(self, x, y):
		logit = self.forward(x)
		loss = self.loss_func(logit, y)
		return loss


	def get_tags(self, x):
		logit = self.forward(x)
		_, tags = torch.max(logit, 1)
		return tags.data.cpu().tolist()

