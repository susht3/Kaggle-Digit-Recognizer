import torch
import os
import time
from datetime import datetime
import numpy as np 
import pandas as pd
from cnn import baseCNN
from loader import loadTrainDataset, loadTestDataset
from train import train, train_epoch, test

class Hyperparameters:
	nb_epoch = 1000
	batch_size = 64
	x_size = 28
	label_size = 10

	learning_rate = 0.001
	model_dir = ''
	res_path = '../model/submit.csv'
	#model_path = '../model/baseCnn_2017-11-02/loss0.0003_acc1.0_80'
	model_path = '../model/baseCnn_2017-11-02/loss2e-06_acc1.0_477'

	train_path = '../data/train.csv'
	test_path = '../data/test.csv'

	train_h5py_path = '../data/train.h5py'
	test_h5py_path = '../data/test.h5py'


def train_model(param):
	print('Loading train dataset...')
	train_dataset = loadTrainDataset(param.train_path)
	train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = param.batch_size, num_workers = 1, shuffle = True)
    
	param.model_dir = '../model/baseCnn2_' + str(datetime.now()).split('.')[0].split()[0] + '/'
	if os.path.exists(param.model_dir) == False:
		os.mkdir(param.model_dir)

	model = baseCNN(param)
	if torch.cuda.is_available():
		model = model.cuda()
	train(model, train_loader, param)


def test_model(param):
	print('Loading test dataset...')
	test_dataset = loadTestDataset(param.test_path)
	test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = param.batch_size, num_workers = 1, shuffle = False)
   
	model = torch.load(param.model_path)
	test(model, test_loader, param)




if __name__ == '__main__':
	param = Hyperparameters() 
	print('Biu ~ ~  ~ ~ ~ Give you buffs ~ \n')
    
	train_model(param)  

	#test_model(test_loader, param)

