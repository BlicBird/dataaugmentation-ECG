import torch
import torch.optim as optim
import pandas as pd
from torch.utils.data import DataLoader
from WGAN_training import LEARNING_RATE_DISC, LEARNING_RATE_GEN

from dataloader_segment_train import ECGDataset
from DCGAN_Models import *

### Hyperparamters ###

data_path = '/home/danielsantosh/intial_folder/china_signal_challenge'

data = ECGDataset(data_path,10000)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LEARNING_RATE_DISC = 2e-4
LEARNING_RATE_GEN = 2e-4 
BATCH_SIZE = 32
ECG_LEN = 10000
CHANNELS_IMG = 12
NOISE_DIM = 624
NUM_EPOCHS = 200
FEATURES_DISC = 12
FEATURES_GEN = 12
NUM_CLASSES = 10
EMBED_CHANNELS = 1

dataloader = DataLoader(data, batch_size=BATCH_SIZE, shuffle=True)
gen = Generator(CHANNELS_IMG, CHANNELS_IMG, FEATURES_GEN, NUM_CLASSES, NOISE_DIM, EMBED_CHANNELS).to(device)
disc = Discriminator(CHANNELS_IMG, FEATURES_DISC, NUM_CLASSES, ECG_LEN, EMBED_CHANNELS).to(device)
initialize_weights(gen)
initialize_weights(disc)

opt_gen = optim.Adam(gen.parameters(), lr=LEARNING_RATE_GEN, betas=(0.5, 0.999))
opt_disc = optim.Adam(disc.parameters(), lr=LEARNING_RATE_DISC, betas=(0.5, 0.999))
criterion = nn.BCELoss()

fixed_noise = torch.randn(BATCH_SIZE, 1, NOISE_DIM).to(device)

gen.train()
disc.train()

### TRAINING ###

training_history = pd.DataFrame({thing:[] for thing in ['Epoch', 'Critic_Loss', 'Generator_Loss']})

for epoch in range(NUM_EPOCHS):
    
    for batch_idx, (real, labels) in enumerate(dataloader):
        real = real.to(device)
        labels = labels.to(device)
        batch_size = real.shape[0]

        ### Train Discriminator
        noise = torch.randn(batch_size, 12, NOISE_DIM).to(device)
        fake = gen(noise,labels)
        disc_real = disc(real, labels)
        lossD_real = criterion(disc_real , torch.ones_like(disc_real)) # log(D(real))
        disc_fake = disc(fake, labels)
        lossD_fake = criterion(disc_fake , torch.zeros_like(disc_fake))
        lossD = (lossD_real + lossD_fake) / 2
        disc.zero_grad()
        lossD.backward(retain_graph = True)
        opt_disc.step()

        ## Train Generator
        output = disc(fake, labels)
        lossG = criterion(output, torch.ones_like(output))
        gen.zero_grad()
        lossG.backward()
        opt_gen.step()
    
        if batch_idx == 0:

            print(
                f"Epoch [{epoch}/{NUM_EPOCHS}] Batch {batch_idx}/{len(dataloader)} \
                      Loss D: {lossD:.4f}, Loss G: {lossG:.4f}"
            )

    training_history.at[epoch, 'Epoch'] = epoch
    training_history.at[epoch, 'Critic_Loss'] = float(lossD)
    training_history.at[epoch, 'Generator_Loss'] = float(lossG)

    save_path = '/home/danielsantosh/data_augmentation/loss_results/DCGAN_base_losses_200e.pkl'
    training_history.to_pickle(save_path)

    model_path_gen = '/home/danielsantosh/data_augmentation/models/DCGAN_gen_base_200e.pt'
    torch.save(gen,model_path_gen)

    model_path_disc = '/home/danielsantosh/data_augmentation/models/DCGAN_disc_fixed_200e.pt'
    torch.save(disc,model_path_disc)