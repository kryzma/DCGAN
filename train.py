import torch
import torch.nn as nn
import torch.optim as optim


from utils import prepare_generator, prepare_discriminator, get_dataLoader, save_generated_imgs

def get_discriminator_loss(disc, real, fake, criterion):
	disc_real = disc(real).reshape(-1)
	loss_disc_real = criterion(disc_real, torch.ones_like(disc_real))
	disc_fake = disc(fake).reshape(-1)
	loss_disc_fake = criterion(disc_fake, torch.zeros_like(disc_fake))
	disc.zero_grad()
	return loss_disc_real + loss_disc_fake


def get_generator_loss(gen, disc, fake, criterion):
	output = disc(fake).reshape(-1)
	loss_gen = criterion(output, torch.ones_like(output))
	gen.zero_grad()
	return loss_gen


def train(config):
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	gen = utils.prepare_generator(config, device)
	disc = utils.prepare_discriminator(config, device)

	fixed_noise = torch.randn(config['result_size'], config['z_dim'], 1, 1).to(device)

	opt_gen = optim.Adam(gen.parameters(), lr=config['learning_rate'], betas=config['betas'])
	opt_disc = optim.Adam(disc.parameters(), lr=config['learning_rate'], betas=config['betas'])
	criterion = nn.BCELoss()

	loader = utils.get_dataLoader(config)

	for epoch in range(config['epoch_cnt']):
		for batch_idx, (real, _) in enumerate(loader):
			real = real.to(device)
			noise = torch.randn((config['batch_size'], config['z_dim'], 1, 1)).to(device)
			fake = gen(noise)

			# Train Discriminator max log(D(x)) + log(1 - D(G(z)))
			# Train Generator min log(D(x)) + log(1 - D(G(z))) (paper suggested) -> max log(D(G(z)))

			# Train Discriminator
			loss_disc = get_discriminator_loss(disc, real, fake, criterion)
			loss_disc.backward(retain_graph=True) # don't delete forward pass cache 
			opt_disc.step()

			# Train Generator
			loss_gen = get_generator_loss(gen, disc, fake, criterion)
			gen.zero_grad()
			loss_gen.backward()
			opt_gen.step()

			# Save current results
			if batch_idx % config['result_period'] == 0:
				utils.save_generated_imgs(
					gen, fixed_noise, batch_idx//config['result_period']*(epoch+1), config)



if __name__ == "__main__":

	config = dict()
	config['result_path'] = "C:/Users/Mantas/Desktop/GAN/results/"

	config['image_size'] = 64
	config['img_channels'] = 1

	# values suggested in the paper
	config['betas'] = (0.5, 0.999)
	config['learning_rate'] = 2e-4

	config['batch_size'] = 128
	config['iterations'] = 10
	config['epoch_cnt'] = 5

	config['z_dim'] = 100

	config['disc_features_cnt'] = 64
	config['gen_features_cnt'] = 64

	config['result_period'] = 10
	config['result_size'] = 32


	train(config)
