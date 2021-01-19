from model import Discriminator, Generator, initialize_weights

import torch
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import numpy as np
import cv2 as cv

def prepare_generator(config, device):
	gen = Generator(config['z_dim'], config['img_channels'], config['gen_features_cnt']).to(device)
	initialize_weights(gen)
	gen.train()
	return gen

def prepare_discriminator(config, device):
	disc = Discriminator(config['img_channels'], config['disc_features_cnt']).to(device)
	initialize_weights(disc)
	disc.train()
	return disc

def get_dataLoader(config):
	transformation = transforms.Compose(
		[
			transforms.Resize(config['image_size']),
			transforms.ToTensor(),
			transforms.Normalize(
				[0.5 for _ in range(config['img_channels'])], [0.5 for _ in range(config['img_channels'])]
			)
		]
	)
	dataset = datasets.MNIST(root="dataset/", train=True, transform=transformation, download=True)
	return DataLoader(dataset, batch_size=config['batch_size'], shuffle=True)

def save_generated_imgs(gen, fixed_noise, id, config):
	with torch.no_grad():
		generated = gen(fixed_noise)

		img_grid = torchvision.utils.make_grid(generated[:32], normalize=True).numpy()
		img_grid = np.moveaxis(img_grid, 0, -1)
		img_grid *= 255

		write_path = config['result_path']+str(id)+".jpg"

		cv.imwrite(write_path, img_grid)
