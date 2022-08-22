import torch
import torch.optim as optim
import pandas as pd
from torch.utils.data import DataLoader
from torchmetrics import F1Score

from dataloader_segment_train import ECGDataset
from WGAN_Models import *

def gradient_penalty(critic, labels, real, fake, device="cpu"):
    BATCH_SIZE, C, L = real.shape
    alpha = torch.rand((BATCH_SIZE, 1, 1)).repeat(1, C, L).to(device)
    interpolated_images = real * alpha + fake * (1 - alpha)

    # Calculate critic scores
    mixed_scores = critic(interpolated_images, labels)

    # Take the gradient of the scores with respect to the images
    gradient = torch.autograd.grad(
        inputs=interpolated_images,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True,
    )[0]
    gradient = gradient.view(gradient.shape[0], -1)
    gradient_norm = gradient.norm(2, dim=1)
    gradient_penalty = torch.mean((gradient_norm - 1) ** 2)
    return gradient_penalty

### Hyperparamters ###

data_path = '/home/danielsantosh/intial_folder/china_signal_challenge'

data = ECGDataset(data_path,10000)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LEARNING_RATE_DISC = 0.0001
LEARNING_RATE_GEN = 0.0001
BATCH_SIZE = 64
ECG_LEN = 10000
CHANNELS_IMG = 12
NOISE_DIM = 624
NUM_EPOCHS = 200
FEATURES_DISC = 12
FEATURES_GEN = 12
CRITIC_ITERATIONS = 5
LAMBDA_GP = 10
NUM_CLASSES = 10
EMBED_CHANNELS = 1

dataloader = DataLoader(data, batch_size=BATCH_SIZE, shuffle=True)
gen = Generator(channels_noise = CHANNELS_IMG, channels_img = CHANNELS_IMG, features_g = FEATURES_GEN, 
num_classes = NUM_CLASSES, noise_dim = NOISE_DIM, embed_channels=EMBED_CHANNELS).to(device)
disc = Discriminator(CHANNELS_IMG, FEATURES_DISC, NUM_CLASSES, ECG_LEN, EMBED_CHANNELS).to(device)
initialize_weights(gen)
initialize_weights(disc)

opt_gen = optim.Adam(gen.parameters(), lr=LEARNING_RATE_GEN, betas=(0.0, 0.9))
opt_disc = optim.Adam(disc.parameters(), lr=LEARNING_RATE_DISC, betas=(0.0, 0.9))

fixed_noise = torch.randn(BATCH_SIZE, 1, NOISE_DIM).to(device)

f1 = F1Score

gen.train()
disc.train()

### TRAINING ###

training_history = pd.DataFrame({thing:[] for thing in ['Epoch', 'Critic_Loss', 'Generator_Loss']})

for epoch in range(NUM_EPOCHS):
    
    for batch_idx, (real, labels) in enumerate(dataloader):
        real = real.to(device)
        cur_batch_size = real.shape[0]
        labels = labels.to(device)

        # Train Critic: max E[critic(real)] - E[critic(fake)]
        # equivalent to minimizing the negative of that
        for _ in range(CRITIC_ITERATIONS):
            noise = torch.randn(cur_batch_size, 12, NOISE_DIM).to(device)
            fake = gen(noise, labels)
            critic_real = disc(real,labels).reshape(-1)
            critic_fake = disc(fake,labels).reshape(-1)
            gp = gradient_penalty(disc, labels, real, fake, device=device)
            loss_critic = (
                -(torch.mean(critic_real) - torch.mean(critic_fake)) + LAMBDA_GP * gp
            )
            disc.zero_grad()
            loss_critic.backward(retain_graph=True)
            opt_disc.step()


        # Train Generator: max E[critic(gen_fake)] <-> min -E[critic(gen_fake)]
        gen_fake = disc(fake, labels).reshape(-1)
        loss_gen = -torch.mean(gen_fake)
        gen.zero_grad()
        loss_gen.backward()
        opt_gen.step()



        if batch_idx == 0:
            print(
                f"Epoch [{epoch}/{NUM_EPOCHS}] \
                      Loss D: {loss_critic:.4f}, loss G: {loss_gen:.4f}"
            )

    training_history.at[epoch, 'Epoch'] = epoch
    training_history.at[epoch, 'Critic_Loss'] = float(loss_critic)
    training_history.at[epoch, 'Generator_Loss'] = float(loss_gen)

save_path = '/home/danielsantosh/data_augmentation/loss_results/WGAN_base_losses_200e.pkl'
training_history.to_pickle(save_path)

model_path_gen = '/home/danielsantosh/data_augmentation/models/WGAN_gen_base_200e.pt'                                                                                                                                                                                                                                                                                                                 
torch.save(gen,model_path_gen)

model_path_disc = '/home/danielsantosh/data_augmentation/models/WGAN_disc_fixed_200e.pt'                                                                                                                                                                                                                                                                                                                 
torch.save(disc,model_path_disc)

